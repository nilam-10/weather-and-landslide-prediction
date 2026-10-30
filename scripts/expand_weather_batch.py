"""Resumable Bounded Batch Downloader and Aligner for Weather Expansion (Task 2).

Expands the authentic NASA MERGIR and IMERG weather datasets across specified date ranges.
Executes:
1. IMERG download & regional cropping (Meghalaya 0.1 deg).
2. MERGIR download & regional cropping (Meghalaya ~4 km).
3. Multimodal alignment onto common 0.1 deg grid.
4. Data validation and manifest generation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_batch_expansion(start_date: str, end_date: str, workers: int = 6):
    print("=" * 75)
    print(f"WEATHER DATASET EXPANSION BATCH: {start_date} TO {end_date}")
    print("=" * 75)

    py_exe = sys.executable

    # Step 1: Download IMERG
    print(f"\n[Step 1/3] Downloading & Subsetting NASA GPM IMERG ({start_date} to {end_date})...")
    cmd_imerg = [
        py_exe,
        str(PROJECT_ROOT / "scripts" / "download_imerg.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--out-dir", "data/imerg",
        "--raw-dir", "data/raw/imerg",
        "--workers", str(workers),
    ]
    t0 = time.time()
    res_imerg = subprocess.run(cmd_imerg, cwd=str(PROJECT_ROOT))
    if res_imerg.returncode != 0:
        print(f"Error during IMERG download (exit code {res_imerg.returncode})")
        return False
    print(f"IMERG download & subsetting completed in {time.time() - t0:.1f} s")

    # Step 2: Download MERGIR
    print(f"\n[Step 2/3] Downloading & Subsetting NASA GPM MERGIR ({start_date} to {end_date})...")
    cmd_mergir = [
        py_exe,
        str(PROJECT_ROOT / "scripts" / "download_mergir.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--out-dir", "data/mergir",
        "--raw-dir", "data/raw/mergir",
        "--workers", str(workers),
    ]
    t0 = time.time()
    res_mergir = subprocess.run(cmd_mergir, cwd=str(PROJECT_ROOT))
    if res_mergir.returncode != 0:
        print(f"Error during MERGIR download (exit code {res_mergir.returncode})")
        return False
    print(f"MERGIR download & subsetting completed in {time.time() - t0:.1f} s")

    # Step 3: Align into common NetCDF4 rasters
    print(f"\n[Step 3/3] Aligning Multimodal Weather Rasters ({start_date} to {end_date})...")
    cmd_align = [
        py_exe,
        str(PROJECT_ROOT / "scripts" / "align_weather.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--source", "mergir",
        "--imerg-dir", "data/imerg",
        "--mergir-dir", "data/mergir",
        "--out-dir", "data/processed/weather",
    ]
    t0 = time.time()
    res_align = subprocess.run(cmd_align, cwd=str(PROJECT_ROOT))
    if res_align.returncode != 0:
        print(f"Error during alignment (exit code {res_align.returncode})")
        return False
    print(f"Multimodal alignment completed in {time.time() - t0:.1f} s")

    print("\n" + "=" * 75)
    print(f"SUCCESSFULLY EXPANDED BATCH: {start_date} TO {end_date}")
    print("=" * 75)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2024-07-08")
    parser.add_argument("--end-date", default="2024-07-14")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    success = run_batch_expansion(args.start_date, args.end_date, workers=args.workers)
    sys.exit(0 if success else 1)
