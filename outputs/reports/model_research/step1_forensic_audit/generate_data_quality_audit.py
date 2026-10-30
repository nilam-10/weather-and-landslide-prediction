import csv
import re
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np

DATA_DIR = Path("data/processed/weather")
OUT_DIR = Path("outputs/reports/model_research/step1_forensic_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("aligned_weather_mergir_*.nc4"))

date_stats = []
all_precip_vals = []
all_tir_vals = []
all_masks = []
all_timestamps = []

total_slots = 0
total_missing_slots = 0

for f in files:
    m = re.search(r"aligned_weather_mergir_(\d{8})\.nc4", f.name)
    if not m:
        continue
    date_str = m.group(1)
    with h5py.File(f, "r") as h:
        p = h["input_precip"][:]
        tb = h["satellite_tir1_mean"][:]
        mask = h["satellite_mask"][:]
        times = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in h["reference_time_iso"][:]]
        
        n_slots = len(times)
        total_slots += n_slots
        all_timestamps.extend(times)
        
        # Check completeness (nominal 48 slots per day)
        is_complete = (n_slots == 48)
        
        # Check valid data in precipitation
        p_valid = np.isfinite(p) & (p >= 0.0)
        p_missing_count = np.sum(~p_valid)
        p_total_pixels = p.size
        
        # Check mask (0 = missing/invalid, 1 = valid)
        mask_valid_count = np.sum(mask == 1)
        mask_total = mask.size
        
        # Rainfall statistics
        p_clean = np.where(p_valid, p, 0.0)
        rain_pixels = np.sum(p_clean > 0.0)
        rain_gt_05 = np.sum(p_clean >= 0.5)
        rain_gt_5 = np.sum(p_clean >= 5.0)
        rain_gt_10 = np.sum(p_clean >= 10.0)
        max_rain = float(np.max(p_clean)) if p_clean.size > 0 else 0.0
        mean_rain = float(np.mean(p_clean)) if p_clean.size > 0 else 0.0
        
        # IR temperature statistics
        tb_valid = np.isfinite(tb) & (tb > 150.0)
        tb_clean = tb[tb_valid]
        mean_tb = float(np.mean(tb_clean)) if tb_clean.size > 0 else 0.0
        min_tb = float(np.min(tb_clean)) if tb_clean.size > 0 else 0.0
        
        date_stats.append({
            "date": date_str,
            "filename": f.name,
            "slots": n_slots,
            "is_complete_day": is_complete,
            "earliest_time": times[0] if times else "",
            "latest_time": times[-1] if times else "",
            "total_pixels": p_total_pixels,
            "valid_mask_pixels": int(mask_valid_count),
            "mask_valid_pct": round(mask_valid_count / max(mask_total, 1) * 100, 2),
            "rain_pixels_gt0": int(rain_pixels),
            "rain_gt0_pct": round(rain_pixels / max(p_total_pixels, 1) * 100, 2),
            "rain_pixels_ge05": int(rain_gt_05),
            "rain_ge05_pct": round(rain_gt_05 / max(p_total_pixels, 1) * 100, 2),
            "heavy_rain_ge50": int(rain_gt_5),
            "heavy_rain_ge50_pct": round(rain_gt_5 / max(p_total_pixels, 1) * 100, 4),
            "extreme_rain_ge100": int(rain_gt_10),
            "max_rain_rate_mm_hr": round(max_rain, 2),
            "mean_rain_rate_mm_hr": round(mean_rain, 4),
            "mean_tb_k": round(mean_tb, 2),
            "min_tb_k": round(min_tb, 2)
        })

# Write CSV
csv_out = OUT_DIR / "DATA_QUALITY_AUDIT.csv"
fieldnames = list(date_stats[0].keys())
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(date_stats)

print(f"Wrote {len(date_stats)} dates to {csv_out}")
