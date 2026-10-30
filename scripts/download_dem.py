"""Download official NASA SRTM 30 m (1 arc-second) NetCDF DEM tiles for Meghalaya.

Dataset: NASA Shuttle Radar Topography Mission Global 1 arc second NetCDF V003 (SRTMGL1_NC)
Provider: NASA LP DAAC (Land Processes Distributed Active Archive Center)
Resolution: ~30 m (1 arc-second, 3601 x 3601 grid per 1x1 degree tile)
Study Area: Meghalaya, India (Lat: 25-27 N, Lon: 89-93 E)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import h5py
import requests

if not os.environ.get("EARTHDATA_NO_DOTENV"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Official 8 tiles covering Meghalaya state and bounding buffer
MEGHALAYA_TILES = [
    "N25E089",
    "N25E090",
    "N25E091",
    "N25E092",
    "N26E089",
    "N26E090",
    "N26E091",
    "N26E092",
]

LPDAAC_BASE_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1_NC.003"


def verify_dem_tile(file_path: Path) -> bool:
    """Verify that a downloaded SRTM tile is valid and readable via HDF5."""
    if not file_path.exists() or file_path.stat().st_size < 1000:
        return False
    try:
        with h5py.File(file_path, "r") as f:
            if "SRTMGL1_DEM" not in f:
                return False
            ds = f["SRTMGL1_DEM"]
            if ds.shape != (3601, 3601):
                return False
        return True
    except Exception:
        return False


def download_tile(tile_name: str, dest_dir: Path, token: str) -> Path:
    """Download a single SRTM NetCDF tile from NASA LP DAAC."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{tile_name}.SRTMGL1_NC.nc"

    if verify_dem_tile(out_path):
        print(f"[{tile_name}] Already downloaded and valid ({out_path.stat().st_size / 1e6:.1f} MB). Skipping.")
        return out_path

    # Check if already present in dem_test scratch
    scratch_path = Path("data/raw/dem_test") / f"{tile_name}.SRTMGL1_NC.nc"
    if verify_dem_tile(scratch_path):
        import shutil
        shutil.copy2(scratch_path, out_path)
        print(f"[{tile_name}] Copied verified file from cache: {out_path}")
        return out_path

    url = f"{LPDAAC_BASE_URL}/{tile_name}.SRTMGL1_NC/{tile_name}.SRTMGL1_NC.nc"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"[{tile_name}] Downloading from NASA LP DAAC...")
    start_t = time.time()
    resp = requests.get(url, headers=headers, stream=True, timeout=120)

    if resp.status_code == 401 or resp.status_code == 403:
        raise PermissionError(
            f"Authentication failed (HTTP {resp.status_code}). Check your NASA Earthdata token."
        )
    resp.raise_for_status()

    temp_path = out_path.with_suffix(".nc.tmp")
    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not verify_dem_tile(temp_path):
        if temp_path.exists():
            temp_path.unlink()
        raise ValueError(f"Downloaded file for {tile_name} failed HDF5 integrity verification.")

    temp_path.replace(out_path)
    elapsed = time.time() - start_t
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[{tile_name}] Successfully downloaded {size_mb:.1f} MB in {elapsed:.1f}s.")
    return out_path


def download_all_meghalaya_dem(dest_dir: Path, tiles: List[str] = MEGHALAYA_TILES) -> List[Path]:
    """Download all required SRTM DEM tiles for Meghalaya."""
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise ValueError(
            "Missing EARTHDATA_TOKEN in environment or .env file. "
            "Please authenticate through official NASA Earthdata Login."
        )

    downloaded: List[Path] = []
    print(f"Starting download of {len(tiles)} official NASA SRTM 30 m tiles...")
    for tile in tiles:
        p = download_tile(tile, dest_dir, token)
        downloaded.append(p)

    print(f"\nAll {len(downloaded)} SRTM DEM tiles successfully verified in: {dest_dir}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official NASA SRTM 30 m DEM for Meghalaya")
    parser.add_argument("--dest-dir", type=str, default="data/raw/dem", help="Target directory for NetCDF files")
    args = parser.parse_args()

    download_all_meghalaya_dem(Path(args.dest_dir))


if __name__ == "__main__":
    main()
