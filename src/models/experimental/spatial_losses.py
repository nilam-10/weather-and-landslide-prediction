"""Differentiable Spatial and Intensity-Weighted Loss Functions for Precipitation Nowcasting.

Implements:
1. Intensity-Weighted L1 / Smooth L1 Loss for mitigating heavy convective underestimation.
2. Differentiable Multiscale Soft-FSS Loss (surrogate for Roberts & Lean 2008 Fractions Skill Score).
3. Compound multi-task objective with element-wise observation masking.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.spatial_metrics import SpatialVerificationEngine as SVE


class IntensityWeightedL1Loss(nn.Module):
    """Intensity-weighted regression loss emphasizing convective rainfall extremes.

    Formulation:
      w(y) = 1.0 + alpha * clamp(y / threshold, 0.0, max_scale)
      w_norm(y) = w(y) / (mean(w(y)) + eps)
      Loss = mean( w_norm(y) * SmoothL1(pred, y) * mask )

    Args:
        threshold: Physical rainfall threshold in mm/hr (default: 5.0).
        alpha: Weight scaling coefficient (default: 0.5).
        max_scale: Upper clamp on intensity scaling (default: 6.0).
        use_smooth_l1: If True, uses Smooth L1; if False, uses standard L1.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        alpha: float = 0.5,
        max_scale: float = 6.0,
        use_smooth_l1: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.max_scale = max_scale
        self.use_smooth_l1 = use_smooth_l1

    def forward(
        self,
        pred_qpf: torch.Tensor,
        target_qpf: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intensity-weighted regression loss.

        Args:
            pred_qpf: (B, Leads, H, W) continuous predictions in mm/hr.
            target_qpf: (B, Leads, H, W) ground truth in mm/hr.
            mask: (B, Leads, H, W) binary validity mask.

        Returns:
            Tuple of (loss_scalar, mean_intensity_weight).
        """
        if mask is None:
            mask = torch.ones_like(target_qpf)
        else:
            mask = (mask > 0).float()

        # Compute physically scaled weights
        y_scaled = torch.clamp(target_qpf / self.threshold, 0.0, self.max_scale)
        weights = 1.0 + self.alpha * y_scaled

        # Normalize weights across valid elements to preserve loss scale
        valid_sum = torch.clamp(mask.sum(), min=1.0)
        mean_w = (weights * mask).sum() / valid_sum
        norm_weights = weights / torch.clamp(mean_w, min=1e-4)

        if self.use_smooth_l1:
            base_loss = F.smooth_l1_loss(pred_qpf, target_qpf, reduction="none")
        else:
            base_loss = F.l1_loss(pred_qpf, target_qpf, reduction="none")

        weighted_loss = norm_weights * base_loss * mask
        loss_val = weighted_loss.sum() / valid_sum

        return loss_val, mean_w


class MultiscaleSoftFSSLoss(nn.Module):
    """Differentiable Multiscale Fractions Skill Score (Soft-FSS) Loss.

    Approximates Roberts & Lean (2008) FSS via smooth sigmoid relaxation of the
    event indicator, followed by multi-scale 2D circular depthwise average pooling.

    Formulation:
      p_pred = sigmoid(kappa * (pred_qpf - threshold))
      p_true = (target_qpf >= threshold).float()
      Fractions F_pred, F_true computed via normalized 2D circular convolutions.
      FBS = sum( (F_pred - F_true)^2 * mask )
      FBS_worst = sum( (F_pred^2 + F_true^2) * mask )
      Loss_k = (FBS + eps) / (FBS_worst + eps)  # Equivalent to 1 - FSS_k
      Total Loss = mean_k( Loss_k )

    Args:
        radii_cells: List of neighborhood radii k in grid cells (default: [1, 2, 3] -> 10, 20, 30 km).
        threshold: Rain rate event threshold in mm/hr (default: 5.0).
        kappa: Sigmoid temperature / steepness parameter (default: 1.0).
    """

    def __init__(
        self,
        radii_cells: List[int] = [1, 2, 3],
        threshold: float = 5.0,
        kappa: float = 1.0,
    ):
        super().__init__()
        self.radii_cells = radii_cells
        self.threshold = threshold
        self.kappa = kappa

        # Precompute 2D circular convolution kernels for each radius
        self.kernels: Dict[int, torch.Tensor] = {}
        for k in radii_cells:
            np_kernel = SVE.create_spatial_kernel(k, kernel_type="circular")
            # Shape for depthwise conv2d: (1, 1, H_k, W_k)
            t_kernel = torch.from_numpy(np_kernel).float().unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"kernel_k{k}", t_kernel)

    def _get_fraction_field(
        self,
        event_prob: torch.Tensor,
        mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial fraction field via 2D convolution with boundary mask normalization."""
        kernel = getattr(self, f"kernel_k{k}")
        pad = k

        # Reshape (B, Leads, H, W) -> (B * Leads, 1, H, W)
        B, Leads, H, W = event_prob.shape
        x_flat = (event_prob * mask).view(B * Leads, 1, H, W)
        m_flat = mask.view(B * Leads, 1, H, W)

        event_sum = F.conv2d(x_flat, kernel, padding=pad)
        mask_sum = F.conv2d(m_flat, kernel, padding=pad)

        denom = torch.clamp(mask_sum, min=1e-4)
        fractions = (event_sum / denom).view(B, Leads, H, W)
        valid_frac = (mask_sum > 0.0).float().view(B, Leads, H, W)

        return fractions, valid_frac

    def forward(
        self,
        pred_qpf: torch.Tensor,
        target_qpf: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multiscale Soft-FSS loss.

        Args:
            pred_qpf: (B, Leads, H, W) continuous predictions in mm/hr.
            target_qpf: (B, Leads, H, W) continuous targets in mm/hr.
            mask: (B, Leads, H, W) binary validity mask.

        Returns:
            Tuple of (total_loss_scalar, dict_of_per_scale_fss_surrogates).
        """
        if mask is None:
            mask = torch.ones_like(target_qpf)
        else:
            mask = (mask > 0).float()

        # Differentiable soft probabilities for predictions
        pred_prob = torch.sigmoid(self.kappa * (pred_qpf - self.threshold))
        true_binary = (target_qpf >= self.threshold).float()

        total_loss = 0.0
        scale_diagnostics: Dict[str, float] = {}

        valid_sum = torch.clamp(mask.sum(), min=1.0)

        for k in self.radii_cells:
            f_pred, v_pred = self._get_fraction_field(pred_prob, mask, k)
            f_true, v_true = self._get_fraction_field(true_binary, mask, k)

            eff_mask = mask * v_pred * v_true

            diff_sq = ((f_pred - f_true) ** 2) * eff_mask
            sum_sq = (f_pred ** 2 + f_true ** 2) * eff_mask

            fbs = diff_sq.sum()
            fbs_worst = sum_sq.sum()

            # Bounded surrogate: (FBS + eps) / (FBS_worst + eps)
            # When perfect match, FBS = 0 -> loss = 0. When zero overlap, FBS = FBS_worst -> loss = 1.
            fss_surrogate = 1.0 - (fbs / torch.clamp(fbs_worst, min=1e-4))
            fss_loss_k = (fbs + 1e-4) / (fbs_worst + 1e-4)
            fss_loss_k = torch.clamp(fss_loss_k, 0.0, 1.0)

            total_loss = total_loss + fss_loss_k
            scale_diagnostics[f"soft_fss_k{k}"] = float(torch.clamp(fss_surrogate, 0.0, 1.0).item())

        mean_loss = total_loss / len(self.radii_cells)
        return mean_loss, scale_diagnostics


class CompoundNowcastingLoss(nn.Module):
    """Compound objective combining Occurrence BCE, Weighted L1, and Multiscale Soft-FSS.

    Loss = w_bce * L_BCE + w_qpf * L_WeightedL1 + w_fss * L_SoftFSS
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_qpf: float = 0.5,
        w_fss: float = 0.0,
        use_weighted_l1: bool = False,
        use_soft_fss: bool = False,
        threshold: float = 5.0,
        radii_cells: List[int] = [1, 2, 3],
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_qpf = w_qpf
        self.w_fss = w_fss
        self.use_weighted_l1 = use_weighted_l1
        self.use_soft_fss = use_soft_fss

        self.bce_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1_criterion = nn.SmoothL1Loss(reduction="none")
        self.weighted_l1_criterion = IntensityWeightedL1Loss(threshold=threshold, alpha=0.5)
        self.soft_fss_criterion = MultiscaleSoftFSSLoss(radii_cells=radii_cells, threshold=threshold)

    def forward(
        self,
        pred_cat: torch.Tensor,
        pred_qpf: torch.Tensor,
        target_bin: torch.Tensor,
        target_cont: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute compound loss and individual components."""
        if mask is None:
            mask = torch.ones_like(target_cont)
        else:
            mask = (mask > 0).float()

        valid_sum = torch.clamp(mask.sum(), min=1.0)

        # 1. Classification BCE loss
        bce_raw = self.bce_criterion(pred_cat, target_bin)
        loss_bce = (bce_raw * mask).sum() / valid_sum

        # 2. Continuous regression loss
        if self.use_weighted_l1:
            loss_qpf, _ = self.weighted_l1_criterion(pred_qpf, target_cont, mask)
        else:
            qpf_raw = self.smooth_l1_criterion(pred_qpf, target_cont)
            loss_qpf = (qpf_raw * mask).sum() / valid_sum

        # 3. Soft-FSS loss
        loss_fss = torch.tensor(0.0, device=pred_qpf.device)
        fss_diags: Dict[str, float] = {}
        if self.use_soft_fss and self.w_fss > 0.0:
            loss_fss, fss_diags = self.soft_fss_criterion(pred_qpf, target_cont, mask)

        total_loss = self.w_bce * loss_bce + self.w_qpf * loss_qpf + self.w_fss * loss_fss

        diagnostics = {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_qpf": float(loss_qpf.item()),
            "loss_fss": float(loss_fss.item()),
            **fss_diags,
        }

        return total_loss, diagnostics
