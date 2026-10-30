"""Spatial Verification & Neighborhood-Based Meteorological Metrics.

Implements Fractions Skill Score (FSS) and Neighborhood Contingency Metrics
following Roberts & Lean (2008), Ebert (2008), and Mittermaier (2014).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi


class SpatialVerificationEngine:
    """Rigorous Spatial Verification Engine for Precipitation Nowcasting.

    Grid Specification:
      Meghalaya Domain: 25 rows (Lat 24.55°N to 26.95°N), 40 cols (Lon 89.55°E to 93.45°E)
      Resolution: 0.1° (~11.08 km Lat, ~10.03 km Lon at 25.75°N, Mean = 10.55 km/cell)
    """

    DEFAULT_GRID_KM = 10.55  # Average grid cell dimension in km

    @staticmethod
    def km_to_grid_radius(radius_km: float, grid_km: float = DEFAULT_GRID_KM) -> int:
        """Map physical distance in km to discrete grid cell radius k.

        Radius k defines window size L = 2k + 1.
          10 km -> k = 1 (L = 3 cells, ~31.7 km window)
          20 km -> k = 2 (L = 5 cells, ~52.8 km window)
          30 km -> k = 3 (L = 7 cells, ~73.9 km window)
          50 km -> k = 5 (L = 11 cells, ~116.1 km window)
        """
        if radius_km <= 0.0:
            return 0
        return max(1, int(round(radius_km / grid_km)))

    @staticmethod
    def create_spatial_kernel(radius_cells: int, kernel_type: str = "square") -> np.ndarray:
        """Create 2D spatial convolution kernel.

        Args:
            radius_cells: Radius k in integer grid cells.
            kernel_type: 'square' (Roberts & Lean 2008 box) or 'circular' (isotropic disk).
        """
        if radius_cells == 0:
            return np.ones((1, 1), dtype=np.float32)

        if kernel_type == "square":
            L = 2 * radius_cells + 1
            return np.ones((L, L), dtype=np.float32)
        elif kernel_type == "circular":
            y, x = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
            return (x**2 + y**2 <= radius_cells**2).astype(np.float32)
        else:
            raise ValueError(f"Unknown kernel_type '{kernel_type}'. Supported: 'square', 'circular'")

    @classmethod
    def compute_fraction_field(
        cls,
        binary_field: np.ndarray,
        mask: Optional[np.ndarray] = None,
        radius_cells: int = 1,
        kernel_type: str = "square",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial fraction field with domain boundary and missing-data handling.

        Args:
            binary_field: 2D binary event array (H, W).
            mask: Optional 2D binary validity mask (1=valid, 0=missing/invalid).
            radius_cells: Neighborhood radius k.
            kernel_type: 'square' or 'circular'.

        Returns:
            Tuple of (fraction_field, valid_cells_mask).
        """
        H, W = binary_field.shape
        if mask is None:
            m = np.ones((H, W), dtype=np.float32)
        else:
            m = (mask > 0).astype(np.float32)

        kernel = cls.create_spatial_kernel(radius_cells, kernel_type=kernel_type)

        # Convolve event count and valid sample count (mode='constant', cval=0.0 handles edges)
        event_eff = (binary_field * m).astype(np.float32)
        event_sum = ndi.convolve(event_eff, kernel, mode="constant", cval=0.0)
        mask_sum = ndi.convolve(m, kernel, mode="constant", cval=0.0)

        # Normalize fractions by actual number of valid cells inside the window
        valid_denom = mask_sum > 0.0
        fractions = np.zeros((H, W), dtype=np.float32)
        fractions[valid_denom] = event_sum[valid_denom] / mask_sum[valid_denom]

        return fractions, m > 0.0

    @classmethod
    def compute_single_fss(
        cls,
        obs_field: np.ndarray,
        fcst_field: np.ndarray,
        threshold: float,
        radius_cells: int = 1,
        mask: Optional[np.ndarray] = None,
        kernel_type: str = "square",
    ) -> Dict[str, Any]:
        """Compute Fractions Skill Score (FSS) for a single 2D field pair.

        Definition:
          FSS = 1 - FBS / FBS_worst
          where FBS = (1/N) * sum((P_fcst - P_obs)^2)
          and FBS_worst = (1/N) * sum(P_fcst^2 + P_obs^2)

        Args:
            obs_field: Continuous observed rainfall array (H, W).
            fcst_field: Continuous forecast rainfall array (H, W).
            threshold: Event precipitation threshold (mm/hr).
            radius_cells: Neighborhood radius k.
            mask: Optional validity mask (H, W).
            kernel_type: 'square' or 'circular'.

        Returns:
            Dictionary containing fss, fbs, fbs_worst, is_zero_event, p_obs_mean, p_fcst_mean.
        """
        obs_bin = (obs_field >= threshold).astype(np.float32)
        fcst_bin = (fcst_field >= threshold).astype(np.float32)

        p_obs, val_mask_obs = cls.compute_fraction_field(obs_bin, mask, radius_cells, kernel_type)
        p_fcst, val_mask_fcst = cls.compute_fraction_field(fcst_bin, mask, radius_cells, kernel_type)

        val_cells = val_mask_obs & val_mask_fcst
        n_valid = int(np.sum(val_cells))

        if n_valid == 0:
            return {
                "fss": 0.0,
                "fbs": 0.0,
                "fbs_worst": 0.0,
                "is_zero_event": True,
                "obs_frequency": 0.0,
                "fcst_frequency": 0.0,
            }

        diff_sq = (p_fcst[val_cells] - p_obs[val_cells]) ** 2
        sum_sq = p_fcst[val_cells] ** 2 + p_obs[val_cells] ** 2

        fbs = float(np.sum(diff_sq) / n_valid)
        fbs_worst = float(np.sum(sum_sq) / n_valid)

        obs_freq = float(np.mean(obs_bin[val_cells]))
        fcst_freq = float(np.mean(fcst_bin[val_cells]))

        # Handle zero-event edge case
        if fbs_worst < 1e-12:
            # Both fields have 0 events everywhere in domain
            return {
                "fss": 1.0,
                "fbs": 0.0,
                "fbs_worst": 0.0,
                "is_zero_event": True,
                "obs_frequency": obs_freq,
                "fcst_frequency": fcst_freq,
            }

        fss = 1.0 - (fbs / fbs_worst)
        fss_clamped = float(np.clip(fss, 0.0, 1.0))

        return {
            "fss": fss_clamped,
            "fbs": fbs,
            "fbs_worst": fbs_worst,
            "is_zero_event": False,
            "obs_frequency": obs_freq,
            "fcst_frequency": fcst_freq,
        }

    @classmethod
    def compute_neighborhood_contingency(
        cls,
        obs_field: np.ndarray,
        fcst_field: np.ndarray,
        threshold: float,
        radius_cells: int = 1,
        mask: Optional[np.ndarray] = None,
        kernel_type: str = "square",
    ) -> Dict[str, float]:
        """Compute Fuzzy Neighborhood Contingency Metrics (Ebert 2008).

        An observed event is counted as a hit if ANY forecast event exists within radius.
        A forecast event is counted as a hit if ANY observed event exists within radius.

        Returns:
            Dictionary of neighborhood csi, pod, far, precision, recall, f1.
        """
        obs_bin = (obs_field >= threshold).astype(bool)
        fcst_bin = (fcst_field >= threshold).astype(bool)

        kernel = cls.create_spatial_kernel(radius_cells, kernel_type=kernel_type)

        # Morphological dilation: checks if event exists within neighborhood window
        obs_dilated = ndi.binary_dilation(obs_bin, structure=kernel)
        fcst_dilated = ndi.binary_dilation(fcst_bin, structure=kernel)

        if mask is not None:
            m = mask > 0
            obs_bin = obs_bin & m
            fcst_bin = fcst_bin & m
            obs_dilated = obs_dilated & m
            fcst_dilated = fcst_dilated & m

        # Hits: Forecast event confirmed by neighborhood observation
        # or Observed event confirmed by neighborhood forecast
        hit_obs = obs_bin & fcst_dilated  # Observed events successfully matched
        hit_fcst = fcst_bin & obs_dilated  # Forecast events successfully matched

        tp = np.sum(hit_obs | hit_fcst)
        fp = np.sum(fcst_bin & ~obs_dilated)
        fn = np.sum(obs_bin & ~fcst_dilated)

        denom_csi = tp + fp + fn
        n_csi = float(tp / denom_csi) if denom_csi > 0 else 0.0

        denom_pod = tp + fn
        n_pod = float(tp / denom_pod) if denom_pod > 0 else 0.0

        denom_far = tp + fp
        n_far = float(fp / denom_far) if denom_far > 0 else 0.0

        denom_f1 = 2 * tp + fp + fn
        n_f1 = float(2 * tp / denom_f1) if denom_f1 > 0 else 0.0

        return {
            "neighborhood_csi": round(n_csi, 4),
            "neighborhood_pod": round(n_pod, 4),
            "neighborhood_far": round(n_far, 4),
            "neighborhood_f1": round(n_f1, 4),
            "neighborhood_tp": int(tp),
            "neighborhood_fp": int(fp),
            "neighborhood_fn": int(fn),
        }

    @classmethod
    def evaluate_sequence_dataset(
        cls,
        obs_sequences: np.ndarray,  # (N_seq, H, W)
        fcst_sequences: np.ndarray,  # (N_seq, H, W)
        threshold: float,
        radius_cells: int = 1,
        mask: Optional[np.ndarray] = None,
        kernel_type: str = "square",
    ) -> Dict[str, Any]:
        """Evaluate FSS across an entire sequence cohort.

        Calculates both:
          1. Accumulated FSS (sum of FBS / sum of FBS_worst across all frames)
             Following Roberts & Lean (2008) and Mittermaier (2014).
          2. Mean frame FSS (arithmetic mean across frames where events exist).
        """
        N_seq = len(obs_sequences)
        total_fbs_num = 0.0
        total_fbs_den = 0.0

        frame_fss_list = []
        zero_event_frames = 0

        for t in range(N_seq):
            m_t = mask[t] if (mask is not None and mask.ndim == 3) else mask
            res = cls.compute_single_fss(
                obs_field=obs_sequences[t],
                fcst_field=fcst_sequences[t],
                threshold=threshold,
                radius_cells=radius_cells,
                mask=m_t,
                kernel_type=kernel_type,
            )
            if res["is_zero_event"]:
                zero_event_frames += 1
            else:
                frame_fss_list.append(res["fss"])
                total_fbs_num += res["fbs"]
                total_fbs_den += res["fbs_worst"]

        if total_fbs_den > 1e-12:
            accumulated_fss = float(np.clip(1.0 - (total_fbs_num / total_fbs_den), 0.0, 1.0))
        else:
            accumulated_fss = 1.0 if zero_event_frames == N_seq else 0.0

        mean_frame_fss = float(np.mean(frame_fss_list)) if frame_fss_list else 1.0

        return {
            "accumulated_fss": round(accumulated_fss, 4),
            "mean_frame_fss": round(mean_frame_fss, 4),
            "active_event_frames": len(frame_fss_list),
            "zero_event_frames": zero_event_frames,
            "total_frames": N_seq,
        }
