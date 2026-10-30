"""Unit Tests for Differentiable Spatial & Intensity-Weighted Losses."""
from __future__ import annotations

import pytest
import torch

from src.models.experimental.spatial_losses import (
    CompoundNowcastingLoss,
    IntensityWeightedL1Loss,
    MultiscaleSoftFSSLoss,
)


def test_intensity_weighted_l1_shapes_and_gradients():
    """Verify weighted L1 computes gradients and weights heavy rain higher."""
    criterion = IntensityWeightedL1Loss(threshold=5.0, alpha=0.5, use_smooth_l1=True)

    pred = torch.zeros((2, 5, 25, 40), requires_grad=True)
    target = torch.zeros((2, 5, 25, 40))

    # Place light rain (1 mm/hr) at cell (10, 10) and heavy rain (20 mm/hr) at cell (15, 15)
    target[:, :, 10, 10] = 1.0
    target[:, :, 15, 15] = 20.0

    loss, mean_w = criterion(pred, target)
    assert loss > 0.0
    assert torch.isfinite(loss)
    loss.backward()

    # Verify gradients exist and are finite
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Gradient magnitude at heavy rain cell (15, 15) must be significantly higher than at light rain cell (10, 10)
    grad_light = pred.grad[:, :, 10, 10].abs().mean()
    grad_heavy = pred.grad[:, :, 15, 15].abs().mean()
    assert grad_heavy > grad_light * 2.0


def test_soft_fss_differentiability_and_bounds():
    """Verify Soft-FSS loss is strictly bounded in [0, 1] and provides valid gradients."""
    criterion = MultiscaleSoftFSSLoss(radii_cells=[1, 2, 3], threshold=5.0, kappa=1.0)

    # Identical fields -> Loss should be near 0
    target = torch.zeros((1, 1, 25, 40))
    target[0, 0, 10:15, 15:20] = 10.0
    pred = target.clone().detach().requires_grad_(True)

    loss_match, diags = criterion(pred, target)
    assert 0.0 <= loss_match.item() <= 1.0
    assert loss_match.item() < 0.1  # Very low loss for near-perfect match

    loss_match.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Zero prediction when target has heavy rain -> Loss should be near 1.0
    pred_zero = torch.zeros((1, 1, 25, 40), requires_grad=True)
    loss_miss, _ = criterion(pred_zero, target)
    assert 0.0 <= loss_miss.item() <= 1.0
    assert loss_miss.item() > 0.8  # Very high loss for complete miss


def test_soft_fss_displacement_monotonicity():
    """Verify that larger storm displacement produces higher Soft-FSS loss."""
    criterion = MultiscaleSoftFSSLoss(radii_cells=[1, 2], threshold=5.0, kappa=1.0)

    target = torch.zeros((1, 1, 25, 40))
    target[0, 0, 12, 18] = 10.0

    # Test shifts of 0, 1, 3 cells
    losses = []
    for shift in [0, 1, 3]:
        pred = torch.zeros((1, 1, 25, 40))
        pred[0, 0, 12, 18 + shift] = 10.0
        l_val, _ = criterion(pred, target)
        losses.append(l_val.item())

    # Loss must increase as spatial displacement increases
    assert losses[0] < losses[1] <= losses[2]


def test_mask_gradient_isolation():
    """Verify that masked pixels (mask=0) receive zero loss and zero gradient."""
    criterion = IntensityWeightedL1Loss(threshold=5.0, alpha=0.5)

    pred = torch.zeros((1, 1, 10, 10), requires_grad=True)
    target = torch.ones((1, 1, 10, 10)) * 10.0
    mask = torch.ones((1, 1, 10, 10))
    mask[:, :, 0:5, 0:5] = 0.0  # Top-left quadrant invalid

    loss, _ = criterion(pred, target, mask=mask)
    loss.backward()

    # Gradients in the invalid region must be exactly 0
    assert (pred.grad[:, :, 0:5, 0:5] == 0.0).all()
    # Gradients in the valid region must be non-zero
    assert (pred.grad[:, :, 5:, 5:].abs() > 0.0).all()


def test_compound_loss_ablation_configurations():
    """Verify that all 4 experimental configurations run and log diagnostics."""
    pred_cat = torch.randn((2, 5, 25, 40), requires_grad=True)
    pred_qpf = torch.rand((2, 5, 25, 40), requires_grad=True) * 10.0
    target_bin = (torch.rand((2, 5, 25, 40)) > 0.8).float()
    target_cont = target_bin * 8.0
    mask = torch.ones((2, 5, 25, 40))

    configs = [
        # Exp A: Baseline
        CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.0, use_weighted_l1=False, use_soft_fss=False),
        # Exp B: Intensity-Weighted L1
        CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.0, use_weighted_l1=True, use_soft_fss=False),
        # Exp C: Multiscale Soft-FSS
        CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.5, use_weighted_l1=False, use_soft_fss=True),
        # Exp D: Combined Objective
        CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.5, use_weighted_l1=True, use_soft_fss=True),
    ]

    for idx, loss_fn in enumerate(configs):
        loss_val, diags = loss_fn(pred_cat, pred_qpf, target_bin, target_cont, mask=mask)
        assert torch.isfinite(loss_val)
        assert "loss_total" in diags
        assert "loss_bce" in diags
        assert "loss_qpf" in diags
        assert "loss_fss" in diags
        loss_val.backward(retain_graph=True)
