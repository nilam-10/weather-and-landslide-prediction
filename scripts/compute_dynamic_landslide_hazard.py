"""Dynamic Landslide Hazard Index Pipeline (NASA LHASA 2.0 Dual Matrix).

Couples static terrain landslide susceptibility (90m LSI from XGBoost) with
dynamic multi-scale antecedent precipitation (IMERG P_24h, P_72h) over Meghalaya.

Follows NASA LHASA 2.0 (Stanley et al., Frontiers in Earth Science 2021) decision matrix:
  - Static LSI tiers: Low (<0.3), Moderate (0.3-0.6), High (0.6-0.8), Very High (>0.8)
  - Antecedent trigger: A = 0.6 * P_24h + 0.4 * P_72h
  - Antecedent tiers: Normal (<50 mm), Elevated (50-100 mm), Extreme (>100 mm)
  - Dynamic Hazard: Low (0), Moderate (1), High (2), Very High (3)

Strictly separates static susceptibility from dynamic rainfall triggering;
avoids invalid multiplicative joint-probability formulas.
Preserves coordinate orientation: lat_90m is descending (North->South) while
lat_weather is ascending (South->North).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

SUSCEPTIBILITY_NC = Path("data/processed/terrain/meghalaya_landslide_susceptibility_90m.nc4")
ANTECEDENT_NC = Path("data/processed/landslide/antecedent_rainfall_july2025.nc4")
TERRAIN_GRID_NC = Path("data/processed/terrain/meghalaya_terrain_aligned_weather_grid.nc4")
OUTPUT_NC = Path("data/processed/landslide/meghalaya_dynamic_hazard_july2025.nc4")
OUTPUT_DIR = Path("outputs/landslide")


def assign_lhasa_hazard_level(lsi: np.ndarray, antecedent_mm: np.ndarray) -> np.ndarray:
    """Compute LHASA 2.0 categorical dynamic hazard level from LSI and antecedent rainfall.
    
    Levels:
      0: Low (Green)
      1: Moderate (Yellow)
      2: High (Orange)
      3: Very High (Red)
    """
    hazard = np.zeros_like(lsi, dtype=np.uint8)

    # Moderate conditions (Level 1)
    cond_mod = (
        ((lsi >= 0.8) & (antecedent_mm < 50.0)) |
        ((lsi >= 0.3) & (lsi < 0.6) & (antecedent_mm >= 50.0) & (antecedent_mm < 100.0)) |
        ((lsi < 0.3) & (antecedent_mm >= 100.0))
    )
    hazard[cond_mod] = 1

    # High conditions (Level 2)
    cond_high = (
        ((lsi >= 0.6) & (lsi < 0.8) & (antecedent_mm >= 50.0) & (antecedent_mm < 100.0)) |
        ((lsi >= 0.3) & (lsi < 0.6) & (antecedent_mm >= 100.0))
    )
    hazard[cond_high] = 2

    # Very High conditions (Level 3)
    cond_vhigh = (
        ((lsi >= 0.6) & (antecedent_mm >= 100.0)) |
        ((lsi >= 0.8) & (antecedent_mm >= 50.0))
    )
    hazard[cond_vhigh] = 3

    return hazard


def compute_dynamic_hazard_series() -> Dict[str, Any]:
    """Compute dynamic hazard across all 336 steps and 90m peak state."""
    print("=" * 70)
    print("COMPUTING DYNAMIC LANDSLIDE HAZARD (NASA LHASA 2.0 PROTOCOL)")
    print("=" * 70)

    # 1. Load 90m susceptibility
    print(f"Loading 90m static susceptibility from: {SUSCEPTIBILITY_NC}")
    with h5py.File(SUSCEPTIBILITY_NC, "r") as f:
        lsi_90m = f["landslide_susceptibility_index"][:]
        lat_90m = f["lat"][:]  # Descending: 26.15 to 25.00
        lon_90m = f["lon"][:]  # Ascending: 89.75 to 92.85
        mask_90m = f["meghalaya_mask"][:]

    # 2. Load antecedent rainfall series
    print(f"Loading antecedent rainfall series from: {ANTECEDENT_NC}")
    with h5py.File(ANTECEDENT_NC, "r") as f:
        p24 = f["accum_24h"][:]  # (336, 25, 40)
        p72 = f["accum_72h"][:]  # (336, 25, 40)
        lat_w = f["lat"][:]      # Ascending: 24.55 to 26.95
        lon_w = f["lon"][:]      # Ascending: 89.55 to 93.45
        ref_times = [
            s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
            for s in f["reference_time_iso"][:]
        ]

    # Load 0.1 deg terrain summary
    with h5py.File(TERRAIN_GRID_NC, "r") as f:
        meghalaya_cov = f["meghalaya_coverage"][:]

    num_steps = len(ref_times)
    print(f"Loaded {num_steps} half-hourly timesteps from July 1 to July 7, 2025.")

    # 3. Resample 90m LSI onto 0.1 deg weather grid using coordinate-aware interpolation
    # lat_90m is descending, so flip to ascending for RegularGridInterpolator
    interp_lsi = RegularGridInterpolator(
        (lat_90m[::-1], lon_90m),
        lsi_90m[::-1, :],
        bounds_error=False,
        fill_value=0.0,
    )
    mesh_w_lat, mesh_w_lon = np.meshgrid(lat_w, lon_w, indexing="ij")
    w_pts = np.stack([mesh_w_lat.ravel(), mesh_w_lon.ravel()], axis=-1)
    lsi_w = interp_lsi(w_pts).reshape((len(lat_w), len(lon_w))).astype(np.float32)
    lsi_w[meghalaya_cov < 0.05] = 0.0

    # 4. Compute dynamic hazard series on 0.1 deg grid for all 336 steps
    dynamic_hazard_series = np.zeros((num_steps, len(lat_w), len(lon_w)), dtype=np.uint8)
    antecedent_series = np.zeros((num_steps, len(lat_w), len(lon_w)), dtype=np.float32)

    max_hazard_step = 0
    max_hazard_count_elevated = -1

    for t in range(num_steps):
        # Extract antecedent values, zeroing NaNs during the warm-up interval
        p24_t = np.nan_to_num(p24[t], nan=0.0)
        p72_t = np.nan_to_num(p72[t], nan=0.0)
        a_idx = (0.6 * p24_t + 0.4 * p72_t).astype(np.float32)
        antecedent_series[t] = a_idx

        # Compute hazard level for cells inside Meghalaya
        haz = assign_lhasa_hazard_level(lsi_w, a_idx)
        haz[meghalaya_cov < 0.05] = 0
        dynamic_hazard_series[t] = haz

        megh_cells = (meghalaya_cov >= 0.05)
        elevated_cells = int(np.sum((haz >= 1) & megh_cells))
        if elevated_cells > max_hazard_count_elevated and t >= 48:
            max_hazard_count_elevated = elevated_cells
            max_hazard_step = t

    print(f"Identified peak dynamic hazard step: t={max_hazard_step} ({ref_times[max_hazard_step]})")
    print(f"Peak elevated/high hazard cells on 0.1 deg grid: {max_hazard_count_elevated}")

    # 5. Project peak antecedent rainfall onto 90m grid using RegularGridInterpolator
    print("Projecting peak antecedent rainfall to 90m grid using RegularGridInterpolator...")
    peak_a_w = antecedent_series[max_hazard_step]
    interp_rain = RegularGridInterpolator(
        (lat_w, lon_w),
        peak_a_w,
        bounds_error=False,
        fill_value=0.0,
    )
    mesh_90m_lat, mesh_90m_lon = np.meshgrid(lat_90m, lon_90m, indexing="ij")
    pts_90m = np.stack([mesh_90m_lat.ravel(), mesh_90m_lon.ravel()], axis=-1)
    peak_a_90m = interp_rain(pts_90m).reshape(lsi_90m.shape).astype(np.float32)

    peak_hazard_90m = assign_lhasa_hazard_level(lsi_90m, peak_a_90m)
    peak_hazard_90m[mask_90m == 0] = 0

    # Calculate 90m area percentages
    valid_90m_count = int(np.sum(mask_90m == 1))
    p_low = float(np.sum((peak_hazard_90m == 0) & (mask_90m == 1)) / valid_90m_count * 100)
    p_mod = float(np.sum((peak_hazard_90m == 1) & (mask_90m == 1)) / valid_90m_count * 100)
    p_high = float(np.sum((peak_hazard_90m == 2) & (mask_90m == 1)) / valid_90m_count * 100)
    p_vhigh = float(np.sum((peak_hazard_90m == 3) & (mask_90m == 1)) / valid_90m_count * 100)

    print(f"Peak 90m Hazard Distribution across Meghalaya ({ref_times[max_hazard_step]}):")
    print(f"  - Low Hazard (0):       {p_low:.2f}%")
    print(f"  - Moderate Hazard (1):  {p_mod:.2f}%")
    print(f"  - High Hazard (2):      {p_high:.2f}%")
    print(f"  - Very High Hazard (3): {p_vhigh:.2f}%")

    # 6. Save output NetCDF
    OUTPUT_NC.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_NC, "w") as f:
        # Attributes
        f.attrs["title"] = "Dynamic Landslide Hazard Index for Meghalaya (NASA LHASA 2.0 Protocol)"
        f.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["framework"] = "NASA LHASA 2.0 (Stanley et al., 2021)"
        f.attrs["disclaimer"] = "Research prototype for situational awareness only. Not an official operational landslide warning."
        f.attrs["peak_hazard_step"] = max_hazard_step
        f.attrs["peak_hazard_time_iso"] = ref_times[max_hazard_step]

        # 0.1 deg time series datasets
        f.create_dataset("dynamic_hazard_series_weather_grid", data=dynamic_hazard_series, compression="gzip", compression_opts=4)
        f.create_dataset("antecedent_trigger_series_mm", data=antecedent_series, compression="gzip", compression_opts=4)
        f.create_dataset("reference_time_iso", data=[s.encode("utf-8") for s in ref_times])
        f.create_dataset("lat_weather_grid", data=lat_w)
        f.create_dataset("lon_weather_grid", data=lon_w)
        f.create_dataset("lsi_weather_grid", data=lsi_w)

        # 90m peak snapshot datasets
        f.create_dataset("peak_dynamic_hazard_90m", data=peak_hazard_90m, compression="gzip", compression_opts=4)
        f.create_dataset("peak_antecedent_rainfall_90m", data=peak_a_90m, compression="gzip", compression_opts=4)
        f.create_dataset("lat_90m", data=lat_90m)
        f.create_dataset("lon_90m", data=lon_90m)
        f.create_dataset("meghalaya_mask_90m", data=mask_90m)

    print(f"Saved dynamic landslide hazard dataset to: {OUTPUT_NC}")

    # 7. Plot visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dynamic_hazard_summary(
        lsi_w, peak_a_w, dynamic_hazard_series[max_hazard_step],
        peak_hazard_90m, mask_90m, ref_times[max_hazard_step],
        p_low, p_mod, p_high, p_vhigh
    )

    return {
        "peak_hazard_step": max_hazard_step,
        "peak_hazard_time_iso": ref_times[max_hazard_step],
        "hazard_90m_distribution_pct": {
            "low_0": round(p_low, 2),
            "moderate_1": round(p_mod, 2),
            "high_2": round(p_high, 2),
            "very_high_3": round(p_vhigh, 2),
        },
    }


def plot_dynamic_hazard_summary(
    lsi_w: np.ndarray,
    peak_a_w: np.ndarray,
    peak_haz_w: np.ndarray,
    peak_haz_90m: np.ndarray,
    mask_90m: np.ndarray,
    peak_time_iso: str,
    p_low: float,
    p_mod: float,
    p_high: float,
    p_vhigh: float,
) -> None:
    """Generate publication-grade summary plot of LHASA 2.0 dynamic hazard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Static LSI (0.1 deg)
    im1 = axes[0, 0].imshow(lsi_w, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Static Landslide Susceptibility Index (LSI, 0.1°)", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label="LSI [0-1]")

    # 2. Peak Antecedent Rainfall (0.1 deg)
    im2 = axes[0, 1].imshow(peak_a_w, cmap="Blues", vmin=0, vmax=max(np.max(peak_a_w), 100.0))
    axes[0, 1].set_title(f"(b) Peak Antecedent Rainfall Index [mm]\n({peak_time_iso})", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Antecedent Rain (mm)")

    # 3. Dynamic Hazard (0.1 deg)
    cmap_haz = matplotlib.colors.ListedColormap(["#2ca02c", "#ffeb3b", "#ff9800", "#f44336"])
    im3 = axes[1, 0].imshow(peak_haz_w, cmap=cmap_haz, vmin=0, vmax=3)
    axes[1, 0].set_title("(c) Dynamic Hazard Level (0.1° Weather Grid)", fontsize=11, fontweight="bold")
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar3.ax.set_yticklabels(["Low", "Moderate", "High", "Very High"])

    # 4. 90m Peak Dynamic Hazard Map
    sub = 2
    haz_sub = peak_haz_90m[::sub, ::sub]
    mask_sub = mask_90m[::sub, ::sub]
    haz_masked = np.where(mask_sub == 1, haz_sub.astype(float), np.nan)

    im4 = axes[1, 1].imshow(haz_masked, cmap=cmap_haz, vmin=0, vmax=3)
    axes[1, 1].set_title(
        f"(d) High-Resolution 90m Peak Dynamic Hazard Map\n"
        f"Low: {p_low:.1f}%, Mod: {p_mod:.1f}%, High: {p_high:.1f}%, V.High: {p_vhigh:.1f}%",
        fontsize=10, fontweight="bold",
    )
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar4.ax.set_yticklabels(["Low", "Moderate", "High", "Very High"])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_fig = OUTPUT_DIR / "meghalaya_dynamic_hazard_lhasa_july2025.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved dynamic hazard figure to: {out_fig}")


if __name__ == "__main__":
    compute_dynamic_hazard_series()
