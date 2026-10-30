"""Scientifically Documented Baseline Inputs for Weather Nowcasting Attribution.

Integrated Gradients and Occlusion Sensitivity require explicit baseline inputs (x').
Attribution represents the change in model output relative to the chosen baseline:
  F(x) - F(x') = sum(Attributions)

Baselines implemented:
1. 'zero': Normalized space zero tensor (corresponds to training dataset mean for Tb and log-P).
2. 'dry_clearsky': Physically-based non-convective clear-sky baseline:
   - Surface/cloud-free Tb = 295.0 K (warm brightness temperature, non-convective)
   - Rain rate = 0.0 mm/hr (completely dry)
   - Sensor mask = 1.0 (valid observation)
3. 'climatological': Mean background non-rainy conditions.
4. 'uniform_min': Minimum normalized inputs.

Scientific Disclosure:
Attribution values are inherently relative to the baseline. Changing the baseline alters the
reference point of 'absence of feature'. Both zero and dry_clearsky baselines are supported,
and baseline choice is explicitly recorded in explanation metadata.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


DEFAULT_WEATHER_STATS = {
    "tb_mean": 258.427,
    "tb_std": 25.964,
    "log_p_mean": 0.3267,
    "log_p_std": 0.5783,
}


def build_weather_baseline(
    x: torch.Tensor,
    baseline_type: str = "dry_clearsky",
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Construct an attribution baseline tensor matching the input shape and device.

    Args:
        x: Input tensor of shape (B, T, C, H, W) or (T, C, H, W).
           Channels: [0: Tb_norm, 1: P_norm, 2: Mask]
        baseline_type: One of 'dry_clearsky', 'zero', 'climatological', 'uniform_min'.
        stats: Normalization statistics dictionary.

    Returns:
        Tuple of (baseline_tensor, baseline_metadata_dict).
    """
    st = stats or DEFAULT_WEATHER_STATS
    baseline = torch.zeros_like(x)

    # Determine dimensions
    has_batch = (x.ndim == 5)
    c_dim = 2 if has_batch else 1

    if baseline_type == "dry_clearsky":
        # Warm, non-convective clear sky (Tb ~ 295 K) and zero rain (P = 0 mm/hr)
        tb_phys = 295.0
        tb_norm_val = (tb_phys - st["tb_mean"]) / (st["tb_std"] + 1e-6)

        p_phys = 0.0
        p_log = np.log1p(p_phys)
        p_norm_val = (p_log - st["log_p_mean"]) / (st["log_p_std"] + 1e-6)

        mask_val = 1.0

        if has_batch:
            baseline[:, :, 0, :, :] = float(tb_norm_val)
            baseline[:, :, 1, :, :] = float(p_norm_val)
            baseline[:, :, 2, :, :] = float(mask_val)
        else:
            baseline[:, 0, :, :] = float(tb_norm_val)
            baseline[:, 1, :, :] = float(p_norm_val)
            baseline[:, 2, :, :] = float(mask_val)

        meta = {
            "baseline_type": "dry_clearsky",
            "description": "Fair-weather clear-sky atmosphere (Tb=295.0 K, Rain=0.0 mm/hr, Mask=1.0)",
            "physical_tb_k": tb_phys,
            "physical_p_mm_hr": p_phys,
            "normalized_tb": float(tb_norm_val),
            "normalized_p": float(p_norm_val),
            "sensor_mask": mask_val,
        }

    elif baseline_type == "zero":
        # Normalized zero (corresponds to training dataset mean)
        mask_val = 1.0
        if has_batch:
            baseline[:, :, 0, :, :] = 0.0
            baseline[:, :, 1, :, :] = 0.0
            baseline[:, :, 2, :, :] = float(mask_val)
        else:
            baseline[:, 0, :, :] = 0.0
            baseline[:, 1, :, :] = 0.0
            baseline[:, 2, :, :] = float(mask_val)

        meta = {
            "baseline_type": "zero",
            "description": "Normalized zero baseline (training mean Tb=258.4 K, log_P=0.327, Mask=1.0)",
            "normalized_tb": 0.0,
            "normalized_p": 0.0,
            "sensor_mask": mask_val,
        }

    elif baseline_type == "climatological":
        # Climatological background with zero rain and mean Tb
        p_norm_val = (0.0 - st["log_p_mean"]) / (st["log_p_std"] + 1e-6)
        if has_batch:
            baseline[:, :, 0, :, :] = 0.0
            baseline[:, :, 1, :, :] = float(p_norm_val)
            baseline[:, :, 2, :, :] = 1.0
        else:
            baseline[:, 0, :, :] = 0.0
            baseline[:, 1, :, :] = float(p_norm_val)
            baseline[:, 2, :, :] = 1.0

        meta = {
            "baseline_type": "climatological",
            "description": "Climatological background (mean Tb, zero rain, Mask=1.0)",
            "normalized_tb": 0.0,
            "normalized_p": float(p_norm_val),
            "sensor_mask": 1.0,
        }

    elif baseline_type == "uniform_min":
        # Minimum observed values
        tb_min_norm = (180.0 - st["tb_mean"]) / (st["tb_std"] + 1e-6)
        p_min_norm = (0.0 - st["log_p_mean"]) / (st["log_p_std"] + 1e-6)
        if has_batch:
            baseline[:, :, 0, :, :] = float(tb_min_norm)
            baseline[:, :, 1, :, :] = float(p_min_norm)
            baseline[:, :, 2, :, :] = 1.0
        else:
            baseline[:, 0, :, :] = float(tb_min_norm)
            baseline[:, 1, :, :] = float(p_min_norm)
            baseline[:, 2, :, :] = 1.0

        meta = {
            "baseline_type": "uniform_min",
            "description": "Uniform minimum baseline (extreme cold, zero rain, Mask=1.0)",
            "normalized_tb": float(tb_min_norm),
            "normalized_p": float(p_min_norm),
            "sensor_mask": 1.0,
        }

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}. Supported: 'dry_clearsky', 'zero', 'climatological', 'uniform_min'.")

    return baseline, meta
