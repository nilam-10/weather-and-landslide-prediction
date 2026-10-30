"""Occlusion Sensitivity (Zeiler & Fergus, 2014) for MultiTaskWeatherConvLSTM.

Perturbation-based explainability:
Directly measures the output sensitivity by replacing small spatial patches, historical frames,
or input channels with a documented baseline, then recording the prediction delta:
  Delta F = F(x) - F(x_occluded)

Significance of Delta F:
- Delta F > 0: Feature supported the forecast (occluding it reduces the predicted rainfall/probability).
- Delta F < 0: Feature suppressed the forecast (occluding it increases the predicted rainfall/probability).
- Delta F ~ 0: Feature has negligible influence on the forecast at the target cell.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.explainability.baselines import build_weather_baseline


def compute_occlusion_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    baseline_type: str = "dry_clearsky",
    target_type: str = "probability",
    lead_idx: int = 0,
    target_cell: Tuple[int, int] = (12, 20),
    patch_size: int = 3,
    batch_size: int = 64,
    stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute spatial, temporal, and channel occlusion sensitivity maps.

    Args:
        model: Trained MultiTaskWeatherConvLSTM in eval mode.
        x: Input tensor of shape (1, 4, 3, 25, 40) or (4, 3, 25, 40).
        baseline: Optional baseline tensor. If None, built via baseline_type.
        baseline_type: 'dry_clearsky', 'zero', 'climatological'.
        target_type: 'probability' (convective occurrence logit) or 'qpf' (rain rate).
        lead_idx: Forecast horizon index (0..4).
        target_cell: Tuple of (row, col) grid coordinates.
        patch_size: Square occlusion window dimension (e.g. 3 for 3x3 patch).
        batch_size: Batch size for batched forward passes during spatial patch sweeping.
        stats: Normalization stats.

    Returns:
        Structured dictionary with spatial occlusion map, frame sensitivity, and channel sensitivity.
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

    with torch.no_grad():
        cat_x, qpf_x = model(x)
        if target_type == "probability":
            fx = float(cat_x[0, lead_idx, r, c].item())
            target_unit = "logit"
        else:
            fx = float(qpf_x[0, lead_idx, r, c].item())
            target_unit = "rainfall rate (mm/hr)"

        # 1. Spatial Patch Occlusion
        occ_spatial_map = np.zeros((H, W), dtype=np.float32)
        half = patch_size // 2

        patches = []
        coords = []
        for pr in range(H):
            for pc in range(W):
                r_min, r_max = max(0, pr - half), min(H, pr + half + 1)
                c_min, c_max = max(0, pc - half), min(W, pc + half + 1)
                x_occ = x.clone()
                x_occ[:, :, :, r_min:r_max, c_min:c_max] = baseline[:, :, :, r_min:r_max, c_min:c_max]
                patches.append(x_occ)
                coords.append((pr, pc))

        # Forward in batches
        diffs = []
        for i in range(0, len(patches), batch_size):
            batch_x = torch.cat(patches[i : i + batch_size], dim=0)
            cat_occ, qpf_occ = model(batch_x)
            if target_type == "probability":
                f_occ = cat_occ[:, lead_idx, r, c]
            else:
                f_occ = qpf_occ[:, lead_idx, r, c]
            diff = (fx - f_occ).cpu().numpy()
            diffs.extend(diff)

        for (pr, pc), d in zip(coords, diffs):
            occ_spatial_map[pr, pc] = float(d)

        # 2. Historical Frame Occlusion (t=0..3)
        frame_diffs = []
        frame_predictions = []
        for t in range(4):
            x_occ = x.clone()
            x_occ[:, t] = baseline[:, t]
            cat_occ, qpf_occ = model(x_occ)
            f_occ = float(cat_occ[0, lead_idx, r, c].item() if target_type == "probability" else qpf_occ[0, lead_idx, r, c].item())
            frame_predictions.append(f_occ)
            frame_diffs.append(float(fx - f_occ))

        # 3. Channel Occlusion (ch=0: Tb, 1: Precip, 2: Mask)
        ch_diffs = []
        ch_predictions = []
        for ch in range(3):
            x_occ = x.clone()
            x_occ[:, :, ch] = baseline[:, :, ch]
            cat_occ, qpf_occ = model(x_occ)
            f_occ = float(cat_occ[0, lead_idx, r, c].item() if target_type == "probability" else qpf_occ[0, lead_idx, r, c].item())
            ch_predictions.append(f_occ)
            ch_diffs.append(float(fx - f_occ))

        # 4. Relative channel impact percentage
        ch_abs = [abs(d) for d in ch_diffs]
        total_ch_abs = sum(ch_abs) + 1e-9
        channel_shares = {
            "infrared_tb_pct": round((ch_abs[0] / total_ch_abs) * 100.0, 2),
            "precipitation_pct": round((ch_abs[1] / total_ch_abs) * 100.0, 2),
            "sensor_mask_pct": round((ch_abs[2] / total_ch_abs) * 100.0, 2),
        }

        # Frame shares
        f_abs = [abs(d) for d in frame_diffs]
        total_f_abs = sum(f_abs) + 1e-9
        frame_shares = {
            "frame_t_minus_90m_pct": round((f_abs[0] / total_f_abs) * 100.0, 2),
            "frame_t_minus_60m_pct": round((f_abs[1] / total_f_abs) * 100.0, 2),
            "frame_t_minus_30m_pct": round((f_abs[2] / total_f_abs) * 100.0, 2),
            "frame_t_current_pct": round((f_abs[3] / total_f_abs) * 100.0, 2),
        }

    lead_names = ["+30m", "+60m", "+120m", "+180m", "+360m"]
    lead_name = lead_names[lead_idx] if 0 <= lead_idx < len(lead_names) else f"lead_{lead_idx}"

    return {
        "method": "Occlusion Sensitivity",
        "target_type": target_type,
        "target_unit": target_unit,
        "lead_horizon": lead_name,
        "lead_idx": lead_idx,
        "target_cell": {"row": r, "col": c},
        "patch_size": patch_size,
        "model_output": {
            "original_fx": fx,
            "target_unit": target_unit,
        },
        "baseline_info": baseline_meta,
        "spatial_occlusion_map": occ_spatial_map,
        "temporal_occlusion_diffs": frame_diffs,
        "temporal_occlusion_predictions": frame_predictions,
        "temporal_shares_pct": frame_shares,
        "channel_occlusion_diffs": {
            "infrared_tb": ch_diffs[0],
            "precipitation": ch_diffs[1],
            "sensor_mask": ch_diffs[2],
        },
        "channel_occlusion_predictions": {
            "infrared_tb": ch_predictions[0],
            "precipitation": ch_predictions[1],
            "sensor_mask": ch_predictions[2],
        },
        "channel_shares_pct": channel_shares,
    }
