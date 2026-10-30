import pandas as pd
import numpy as np
import h5py
from pathlib import Path

df = pd.read_csv("outputs/reports/model_research/step1_forensic_audit/DATA_QUALITY_AUDIT.csv")
print("=== DATA QUALITY SUMMARY ===")
print(f"Total dates audited: {len(df)}")
print(f"Complete days (48 slots): {int(df['is_complete_day'].sum())}")
print(f"Incomplete days: {int((~df['is_complete_day']).sum())}")
print(f"Total slots: {df['slots'].sum()}")
print(f"Earliest timestamp: {df['earliest_time'].min()}")
print(f"Latest timestamp: {df['latest_time'].max()}")
print(f"Total pixels: {df['total_pixels'].sum():,}")
print(f"Valid mask pixels: {df['valid_mask_pixels'].sum():,} ({df['valid_mask_pixels'].sum() / df['total_pixels'].sum() * 100:.2f}%)")
print(f"Rainy pixels (>0 mm/hr): {df['rain_pixels_gt0'].sum():,} ({df['rain_pixels_gt0'].sum() / df['total_pixels'].sum() * 100:.2f}%)")
print(f"Rainy pixels (>=0.5 mm/hr): {df['rain_pixels_ge05'].sum():,} ({df['rain_pixels_ge05'].sum() / df['total_pixels'].sum() * 100:.2f}%)")
print(f"Heavy rain (>=5.0 mm/hr): {df['heavy_rain_ge50'].sum():,} ({df['heavy_rain_ge50'].sum() / df['total_pixels'].sum() * 100:.4f}%)")
print(f"Extreme rain (>=10.0 mm/hr): {df['extreme_rain_ge100'].sum():,} ({df['extreme_rain_ge100'].sum() / df['total_pixels'].sum() * 100:.4f}%)")
print(f"Maximum single-pixel rain rate: {df['max_rain_rate_mm_hr'].max():.2f} mm/hr")

# Compute overall percentiles of precipitation across the whole dataset
print("\nComputing precipitation percentiles across all active monsoon days...")
all_p = []
for pfile in sorted(Path("data/processed/weather").glob("aligned_weather_mergir_*.nc4")):
    if "20250201" in pfile.name:
        continue
    with h5py.File(pfile, "r") as h:
        p = h["input_precip"][:]
        valid = np.isfinite(p) & (p >= 0.0)
        all_p.append(p[valid])

all_p = np.concatenate(all_p)
print(f"Total valid pixels sampled: {len(all_p):,}")
for q in [50, 75, 90, 95, 98, 99, 99.5, 99.9]:
    val = np.percentile(all_p, q)
    print(f"  {q}th percentile: {val:.4f} mm/hr")

# Analyze rainy-pixel-only percentiles
rainy_p = all_p[all_p > 0.0]
print(f"\nConditioned on rain (>0 mm/hr) ({len(rainy_p):,} pixels):")
for q in [50, 75, 90, 95, 98, 99, 99.5, 99.9]:
    val = np.percentile(rainy_p, q)
    print(f"  {q}th percentile: {val:.4f} mm/hr")
