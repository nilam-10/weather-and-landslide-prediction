"""Batch Monsoon Data Expansion Manager (Phase 2).

Systematically downloads and aligns authentic NASA IMERG V07B and MERGIR V1
observations across multiple target historical monsoon periods in 2024 and 2023.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_PROC_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
WEATHER_PROC_DIR.mkdir(parents=True, exist_ok=True)


def acquire_period(start_date: str, end_date: str, workers: int = 6) -> None:
    """Download IMERG, MERGIR and align for a specific date range."""
    py_exe = sys.executable
    print("=" * 75)
    print(f"ACQUIRING MONSOON PERIOD: {start_date} to {end_date}")
    print("=" * 75)

    # 1. IMERG
    print(f"\n[1/3] Downloading IMERG V07B ({start_date} to {end_date})...")
    subprocess.run([
        py_exe, str(PROJECT_ROOT / "scripts" / "download_imerg.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--workers", str(workers),
    ], check=True)

    # 2. MERGIR
    print(f"\n[2/3] Downloading MERGIR V1 ({start_date} to {end_date})...")
    subprocess.run([
        py_exe, str(PROJECT_ROOT / "scripts" / "download_mergir.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--workers", str(workers),
    ], check=True)

    # 3. Align each day
    print(f"\n[3/3] Aligning datasets onto 0.1 deg study grid...")
    d_curr = datetime.strptime(start_date, "%Y-%m-%d").date()
    d_end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while d_curr <= d_end:
        ds = d_curr.strftime("%Y-%m-%d")
        ds_flat = d_curr.strftime("%Y%m%d")
        out_nc = WEATHER_PROC_DIR / f"aligned_weather_mergir_{ds_flat}.nc4"
        if out_nc.exists() and out_nc.stat().st_size > 50000:
            print(f"  {ds}: Already aligned ({out_nc.name})")
        else:
            print(f"  {ds}: Aligning...")
            subprocess.run([
                py_exe, str(PROJECT_ROOT / "scripts" / "align_weather.py"),
                "--source", "mergir",
                "--date", ds,
            ], check=True)
            print(f"  {ds}: Done -> {out_nc.name}")
        d_curr += timedelta(days=1)
    print(f"\nSuccessfully completed period: {start_date} to {end_date}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch monsoon data expansion manager.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    acquire_period(args.start_date, args.end_date, workers=args.workers)


if __name__ == "__main__":
    main()
