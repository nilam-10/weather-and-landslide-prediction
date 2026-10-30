"""Integrated Gradients (Sundararajan et al., 2017) for MultiTaskWeatherConvLSTM.

Mathematical Formulation:
  IG_i(x) = (x_i - x'_i) * Integral_0^1 [ dF(x' + alpha*(x - x')) / dx_i ] d(alpha)
Approximated via Riemann summation over m discrete interpolation steps:
  IG_i(x) approx (x_i - x'_i) * (1/m) * Sum_{k=1}^m [ dF(x' + (k/m)*(x - x')) / dx_i ]

Completeness Axiom:
  Sum_i IG_i(x) = F(x) - F(x')

Supported Targets:
1. 'probability' (convective occurrence logit or sigmoid probability at lead_idx, r, c)
2. 'qpf' (continuous precipitation rate in mm/hr at lead_idx, r, c)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.explainability.baselines import build_weather_baseline


def compute_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    baseline_type: str = "dry_clearsky",
    target_type: str = "probability",
    attribution_space: str = "logit",
    lead_idx: int = 0,
    target_cell: Tuple[int, int] = (12, 20),
    m_steps: int = 50,
    batch_size: int = 10,
    stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute spatiotemporal Integrated Gradients for a specific forecast target.

    Args:
        model: Trained MultiTaskWeatherConvLSTM in eval mode.
        x: Input tensor of shape (1, 4, 3, 25, 40) or (4, 3, 25, 40).
        baseline: Optional explicit baseline tensor. If None, built via baseline_type.
        baseline_type: 'dry_clearsky', 'zero', 'climatological', 'uniform_min'.
        target_type: 'probability' (convective occurrence) or 'qpf' (rain rate).
        attribution_space: 'logit' (default, linear gradient flow) or 'probability' (sigmoid output).
        lead_idx: Forecast horizon index (0: +30m, 1: +60m, 2: +120m, 3: +180m, 4: +360m).
        target_cell: Tuple of (row, col) grid coordinates in [0..24] x [0..39].
        m_steps: Number of Riemann interpolation steps (typically 50).
        batch_size: Sub-batch size for gradient computation to bound GPU/CPU memory.
        stats: Normalization parameters dict.

    Returns:
        Structured explanation dictionary with raw arrays, spatial maps, channel shares, and completeness proof.
    """
    model.eval()
    device = next(model.parameters()).device

    if x.ndim == 4:
        x = x.unsqueeze(0)
    x = x.to(device)

    r, c = target_cell
    H, W = x.shape[3], x.shape[4]
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(f"target_cell ({r}, {c}) out of bounds for grid shape ({H}, {W}).")

    if baseline is None:
        baseline, baseline_meta = build_weather_baseline(x, baseline_type=baseline_type, stats=stats)
    else:
        baseline_meta = {"baseline_type": "custom", "description": "User-supplied custom baseline"}

    if baseline.ndim == 4:
        baseline = baseline.unsqueeze(0)
    baseline = baseline.to(device)

    # 1. Forward pass on input x and baseline x'
    with torch.no_grad():
        cat_x, qpf_x = model(x)
        cat_b, qpf_b = model(baseline)

        if target_type == "probability":
            raw_logit_x = float(cat_x[0, lead_idx, r, c].item())
            raw_logit_b = float(cat_b[0, lead_idx, r, c].item())
            prob_x = float(torch.sigmoid(cat_x[0, lead_idx, r, c]).item())
            prob_b = float(torch.sigmoid(cat_b[0, lead_idx, r, c]).item())

            if attribution_space == "probability":
                fx = prob_x
                fb = prob_b
                target_unit = "probability [0-1]"
            else:
                fx = raw_logit_x
                fb = raw_logit_b
                target_unit = "logit"
        elif target_type == "qpf":
            fx = float(qpf_x[0, lead_idx, r, c].item())
            fb = float(qpf_b[0, lead_idx, r, c].item())
            target_unit = "rainfall rate (mm/hr)"
        else:
            raise ValueError(f"Unknown target_type: {target_type}. Supported: 'probability', 'qpf'.")

    # 2. Compute path gradients via Riemann summation
    delta = x - baseline  # (1, 4, 3, 25, 40)
    alphas = torch.linspace(0.0, 1.0, m_steps + 1, device=device)[1:]  # [m_steps]

    grads_list = []
    for i in range(0, m_steps, batch_size):
        sub_alphas = alphas[i : i + batch_size]
        # Shape: (sub_batch, 4, 3, 25, 40)
        interp_x = baseline + sub_alphas.view(-1, 1, 1, 1, 1) * delta
        interp_x.requires_grad_(True)

        cat_out, qpf_out = model(interp_x)

        if target_type == "probability":
            if attribution_space == "probability":
                target_vals = torch.sigmoid(cat_out[:, lead_idx, r, c])
            else:
                target_vals = cat_out[:, lead_idx, r, c]
        else:
            target_vals = qpf_out[:, lead_idx, r, c]

        grad_out = torch.ones_like(target_vals)
        sub_grads = torch.autograd.grad(
            outputs=target_vals,
            inputs=interp_x,
            grad_outputs=grad_out,
            create_graph=False,
            retain_graph=False,
        )[0]
        grads_list.append(sub_grads.detach())

    all_grads = torch.cat(grads_list, dim=0)  # (m_steps, 4, 3, 25, 40)
    avg_grads = torch.mean(all_grads, dim=0, keepdim=True)  # (1, 4, 3, 25, 40)

    # 3. Elementwise attribution
    ig_raw = (delta * avg_grads).squeeze(0)  # (4, 3, 25, 40)
    ig_raw_np = ig_raw.cpu().numpy().astype(np.float32)

    # 4. Completeness validation
    sum_ig = float(ig_raw_np.sum())
    delta_f = float(fx - fb)
    abs_err = abs(sum_ig - delta_f)
    rel_err = abs_err / (abs(delta_f) + 1e-7)

    # 5. Channel and spatiotemporal reductions
    # Channels: 0: IR Tb, 1: Precipitation, 2: Mask
    tb_attr = ig_raw_np[:, 0, :, :]      # (4, 25, 40)
    p_attr = ig_raw_np[:, 1, :, :]       # (4, 25, 40)
    mask_attr = ig_raw_np[:, 2, :, :]    # (4, 25, 40)

    # Spatial attribution maps (summed across 4 input lookback frames)
    spatial_total = np.sum(ig_raw_np, axis=(0, 1))      # (25, 40)
    spatial_tb = np.sum(tb_attr, axis=0)                # (25, 40)
    spatial_p = np.sum(p_attr, axis=0)                  # (25, 40)
    spatial_mask = np.sum(mask_attr, axis=0)            # (25, 40)

    # Frame attribution (summed over channels and space)
    # Frames: [t-3, t-2, t-1, t]
    frame_attr_signed = np.sum(ig_raw_np, axis=(1, 2, 3)).tolist()  # 4 values
    frame_attr_abs = np.sum(np.abs(ig_raw_np), axis=(1, 2, 3)).tolist()

    # Channel attribution (summed over time and space)
    ch_tb_signed = float(np.sum(tb_attr))
    ch_p_signed = float(np.sum(p_attr))
    ch_mask_signed = float(np.sum(mask_attr))

    ch_tb_abs = float(np.sum(np.abs(tb_attr)))
    ch_p_abs = float(np.sum(np.abs(p_attr)))
    ch_mask_abs = float(np.sum(np.abs(mask_attr)))
    total_abs = ch_tb_abs + ch_p_abs + ch_mask_abs + 1e-9

    channel_shares = {
        "infrared_tb_pct": round((ch_tb_abs / total_abs) * 100.0, 2),
        "precipitation_pct": round((ch_p_abs / total_abs) * 100.0, 2),
        "sensor_mask_pct": round((ch_mask_abs / total_abs) * 100.0, 2),
    }

    # Frame shares
    total_frame_abs = sum(frame_attr_abs) + 1e-9
    frame_shares = {
        "frame_t_minus_90m_pct": round((frame_attr_abs[0] / total_frame_abs) * 100.0, 2),
        "frame_t_minus_60m_pct": round((frame_attr_abs[1] / total_frame_abs) * 100.0, 2),
        "frame_t_minus_30m_pct": round((frame_attr_abs[2] / total_frame_abs) * 100.0, 2),
        "frame_t_current_pct": round((frame_attr_abs[3] / total_frame_abs) * 100.0, 2),
    }

    lead_names = ["+30m", "+60m", "+120m", "+180m", "+360m"]
    lead_name = lead_names[lead_idx] if 0 <= lead_idx < len(lead_names) else f"lead_{lead_idx}"

    return {
        "method": "Integrated Gradients",
        "target_type": target_type,
        "attribution_space": attribution_space,
        "target_unit": target_unit,
        "lead_horizon": lead_name,
        "lead_idx": lead_idx,
        "target_cell": {"row": r, "col": c},
        "model_output": {
            "fx": fx,
            "f_baseline": fb,
            "delta_f": delta_f,
        },
        "completeness_proof": {
            "sum_attributions": sum_ig,
            "delta_f": delta_f,
            "absolute_error": abs_err,
            "relative_error_pct": round(rel_err * 100.0, 3),
            "completeness_verified": bool(rel_err < 0.05),
            "riemann_steps": m_steps,
        },
        "baseline_info": baseline_meta,
        "raw_attribution_shape": list(ig_raw_np.shape),
        "raw_attribution": ig_raw_np,
        "spatial_attribution_total": spatial_total,
        "spatial_attribution_tb": spatial_tb,
        "spatial_attribution_p": spatial_p,
        "spatial_attribution_mask": spatial_mask,
        "temporal_attribution_signed": frame_attr_signed,
        "temporal_shares_pct": frame_shares,
        "channel_attribution_signed": {
            "infrared_tb": ch_tb_signed,
            "precipitation": ch_p_signed,
            "sensor_mask": ch_mask_signed,
        },
        "channel_shares_pct": channel_shares,
    }
