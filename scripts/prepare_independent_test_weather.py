"""Prepare, Align and Freeze the Independent Weather Test Period (Step 2).

Dates: 2024-08-01, 2024-08-02, 2024-08-03 (3 full days, 144 half-hourly intervals).
Generates immutable frozen test manifest: outputs/reports/frozen_independent_test_manifest.csv.
Computes SHA256 hashes, coordinate bounds, validity checks, and sequence counts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
OUTPUTS_REP = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_REP.mkdir(parents=True, exist_ok=True)

TARGET_DATES = ["20240801", "20240802", "20240803"]
DATE_STRS = ["2024-08-01", "2024-08-02", "2024-08-03"]
LEAD_STEPS = [1, 2, 4, 6, 12]  # +30m, +60m, +120m, +180m, +360m
T_IN = 4


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_and_align_independent_period() -> List[Dict[str, Any]]:
    py_exe = sys.executable

    # Check and run alignment for each date if not already aligned
    for d_str in DATE_STRS:
        d_clean = d_str.replace("-", "")
        aligned_path = WEATHER_DIR / f"aligned_weather_mergir_{d_clean}.nc4"
        if not aligned_path.exists() or aligned_path.stat().st_size < 100000:
            print(f"Aligning {d_str}...")
            cmd = [
                py_exe,
                str(PROJECT_ROOT / "scripts" / "align_weather.py"),
                "--source", "mergir",
                "--date", d_str,
                "--mergir-dir", "data/mergir",
                "--imerg-dir", "data/imerg",
                "--out-dir", "data/processed/weather",
            ]
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if res.returncode != 0:
                raise RuntimeError(f"Alignment failed for date {d_str}")

    manifest_records = []

    # Load all 3 days concatenated to compute exact continuous hermetic sequences
    all_p = []
    all_tb = []
    all_ref_times = []

    for d_clean in TARGET_DATES:
        fpath = WEATHER_DIR / f"aligned_weather_mergir_{d_clean}.nc4"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing expected aligned file: {fpath}")

        sha = compute_sha256(fpath)
        size = fpath.stat().st_size

        with h5py.File(fpath, "r") as f:
            p = f["input_precip"][:]
            tb = f["satellite_tir1_mean"][:]
            mask = f["satellite_mask"][:]
            ref_iso = [
                s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                for s in f["reference_time_iso"][:]
            ]
            all_p.append(p)
            all_tb.append(tb)
            all_ref_times.extend(ref_iso)

            val_pct = float(np.mean(mask) * 100.0)
            valid_p = p[np.isfinite(p) & (p >= 0.0)]
            valid_tb = tb[np.isfinite(tb) & (tb > 100)]

            manifest_records.append({
                "date": f"{d_clean[:4]}-{d_clean[4:6]}-{d_clean[6:]}",
                "aligned_filename": fpath.name,
                "sha256_checksum": sha,
                "filesize_bytes": size,
                "total_timesteps": len(p),
                "satellite_availability_percent": round(val_pct, 2),
                "min_precip_mm_hr": round(float(np.min(valid_p)), 3) if len(valid_p) else 0.0,
                "mean_precip_mm_hr": round(float(np.mean(valid_p)), 3) if len(valid_p) else 0.0,
                "max_precip_mm_hr": round(float(np.max(valid_p)), 3) if len(valid_p) else 0.0,
                "min_tb_kelvin": round(float(np.min(valid_tb)), 1) if len(valid_tb) else 0.0,
                "mean_tb_kelvin": round(float(np.mean(valid_tb)), 1) if len(valid_tb) else 0.0,
                "max_tb_kelvin": round(float(np.max(valid_tb)), 1) if len(valid_tb) else 0.0,
                "first_ref_time_utc": ref_iso[0],
                "last_ref_time_utc": ref_iso[-1],
                "status": "FROZEN_INDEPENDENT_TEST_SET",
            })

    # Now verify continuous sequences across the entire independent test span
    concat_p = np.concatenate(all_p, axis=0)  # (144, 25, 40)
    T_total = len(concat_p)
    max_lead = max(LEAD_STEPS)

    valid_seq_count = 0
    seq_records = []

    for t in range(T_IN - 1, T_total - max_lead):
        # Verify continuous 30-min spacing
        dt_ref = datetime.fromisoformat(all_ref_times[t].replace("Z", "+00:00"))
        dt_t0 = datetime.fromisoformat(all_ref_times[t - T_IN + 1].replace("Z", "+00:00"))
        if (dt_ref - dt_t0).total_seconds() != (T_IN - 1) * 1800:
            continue

        # Verify target intervals
        dt_target_max = datetime.fromisoformat(all_ref_times[t + max_lead].replace("Z", "+00:00"))
        if (dt_target_max - dt_ref).total_seconds() != max_lead * 1800:
            continue

        valid_seq_count += 1
        seq_records.append({
            "sequence_id": f"INDEP_TEST_{valid_seq_count:04d}",
            "ref_time_iso": all_ref_times[t],
            "input_t0_minus90m": all_ref_times[t - 3],
            "input_t3_ref00m": all_ref_times[t],
            "target_plus30m": all_ref_times[t + 1],
            "target_plus60m": all_ref_times[t + 2],
            "target_plus120m": all_ref_times[t + 4],
            "target_plus180m": all_ref_times[t + 6],
            "target_plus360m": all_ref_times[t + 12],
        })

    print(f"Total timesteps: {T_total} (3 days = 72 hours)")
    print(f"Total valid hermetic independent test sequences: {valid_seq_count}")

    # Add sequence counts to manifest
    for r in manifest_records:
        r["independent_test_total_sequences"] = valid_seq_count

    # Save immutable manifest
    out_csv = OUTPUTS_REP / "frozen_independent_test_manifest.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_records)
    print(f"Saved frozen independent test manifest to: {out_csv}")

    return manifest_records


if __name__ == "__main__":
    verify_and_align_independent_period()
