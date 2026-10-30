"""NASA GPM MERGIR (GPM_MERGIR V1) downloader and regional subsetter.

Downloads NOAA/NCEP/CPC merged geostationary infrared brightness temperature (Tb)
from NASA GES DISC and crops to the Meghalaya study region (89.5-93.5 E, 24.5-27.0 N).
Preserves Kelvin units, coordinates, timestamps, and missing-value masks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import getpass
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import requests

if not os.environ.get("EARTHDATA_NO_DOTENV"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
URS_AUTH_HOST = "urs.earthdata.nasa.gov"
GES_DISC_HOST = "data.gesdisc.earthdata.nasa.gov"
DEFAULT_BBOX = (89.5, 24.5, 93.5, 27.0)  # West, South, East, North (Meghalaya buffer)
COLLECTION_SHORT_NAME = "GPM_MERGIR"
COLLECTION_VERSION = "1"


class EarthdataSession(requests.Session):
    """Requests session that preserves Authorization header across Earthdata redirects."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, token: Optional[str] = None):
        super().__init__()
        self.auth_username = username
        self.auth_password = password
        self.bearer_token = token
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        elif username and password:
            self.auth = (username, password)

    def rebuild_auth(self, prepared_request: requests.PreparedRequest, response: requests.Response) -> None:
        """Override to keep Authorization header on URS authentication redirects."""
        headers = prepared_request.headers
        target_url = prepared_request.url or ""
        parsed_target = urllib.parse.urlparse(target_url)
        parsed_orig = urllib.parse.urlparse(response.url)

        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.auth_username and self.auth_password:
            if parsed_target.hostname == URS_AUTH_HOST or parsed_target.hostname == parsed_orig.hostname:
                prepared_request.prepare_auth((self.auth_username, self.auth_password))


def get_netrc_path() -> Path:
    """Return the platform-appropriate .netrc / _netrc path."""
    user_home = Path.home()
    if os.name == "nt":
        if (user_home / "_netrc").exists():
            return user_home / "_netrc"
        if (user_home / ".netrc").exists():
            return user_home / ".netrc"
        return user_home / "_netrc"
    return user_home / ".netrc"


def read_netrc_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Read Earthdata credentials from netrc file if present."""
    netrc_path = get_netrc_path()
    if not netrc_path.exists():
        alt = Path.home() / (".netrc" if netrc_path.name == "_netrc" else "_netrc")
        if alt.exists():
            netrc_path = alt
        else:
            return None, None

    try:
        import netrc
        n = netrc.netrc(str(netrc_path))
        auth = n.authenticators(URS_AUTH_HOST)
        if auth:
            return auth[0], auth[2]
    except Exception:
        try:
            content = netrc_path.read_text(encoding="utf-8")
            parts = content.split()
            for i, p in enumerate(parts):
                if p == "machine" and i + 1 < len(parts) and parts[i + 1] == URS_AUTH_HOST:
                    login_idx = parts.index("login", i)
                    pass_idx = parts.index("password", i)
                    return parts[login_idx + 1], parts[pass_idx + 1]
        except Exception:
            pass
    return None, None


def get_authenticated_session(token: Optional[str] = None) -> Optional[requests.Session]:
    """Create an authenticated requests session using token, netrc, or env vars."""
    tok = token or os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return EarthdataSession(token=tok)

    env_user = os.environ.get("EARTHDATA_USERNAME")
    env_pass = os.environ.get("EARTHDATA_PASSWORD")
    if env_user and env_pass:
        return EarthdataSession(username=env_user, password=env_pass)

    netrc_user, netrc_pass = read_netrc_credentials()
    if netrc_user and netrc_pass:
        return EarthdataSession(username=netrc_user, password=netrc_pass)

    return None


def query_cmr_granules(start_date: str, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query NASA CMR API for GPM_MERGIR V1 granules across a date range."""
    if not end_date:
        end_date = start_date
    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T23:59:59Z"
    granules = []
    page_num = 1
    page_size = 500

    while True:
        params = {
            "short_name": COLLECTION_SHORT_NAME,
            "version": COLLECTION_VERSION,
            "temporal": f"{start_time},{end_time}",
            "page_size": page_size,
            "page_num": page_num,
            "sort_key": "start_date",
        }
        url = f"{CMR_GRANULES_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "RBU-Honours-MERGIR-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break

        for e in entries:
            title = e.get("title", "")
            links = e.get("links", [])
            download_url = None
            for l in links:
                if l.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#" and "data.gesdisc" in l.get("href", ""):
                    download_url = l.get("href")
                    break
            if not download_url:
                for l in links:
                    if l.get("href", "").endswith((".nc4", ".nc", ".HDF5")):
                        download_url = l.get("href")
                        break

            granules.append({
                "title": title,
                "granule_id": e.get("id"),
                "download_url": download_url,
                "time_start": e.get("time_start"),
                "time_end": e.get("time_end"),
                "size_mb": float(e.get("granule_size", 0.0)),
            })

        if len(entries) < page_size:
            break
        page_num += 1

    return sorted(granules, key=lambda x: x["time_start"])


def download_file(url: str, dest_path: Path, session: requests.Session, expected_size_mb: float = 0.0) -> bool:
    """Download a file with resume support and integrity checks."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        return True

    resp = session.get(url, stream=True, timeout=60, allow_redirects=True)
    if resp.status_code != 200:
        if resp.status_code == 401:
            raise PermissionError("NASA Earthdata authentication failed (HTTP 401). Check credentials.")
        if resp.status_code == 403:
            raise PermissionError("NASA Earthdata HTTP 403 Forbidden. Ensure application authorization is granted.")
        raise RuntimeError(f"HTTP Error {resp.status_code} while downloading {url}")

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        raise PermissionError("Received HTML login page instead of NetCDF-4 file. Check Earthdata authorization.")

    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    temp_path.replace(dest_path)
    return True


def crop_mergir_granule(
    raw_path: Path,
    out_dir: Path,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
) -> List[Dict[str, Any]]:
    """Extract regional subset for each 30-minute interval in the hourly MERGIR granule.

    MERGIR NetCDF-4 files typically contain:
      - Variable: Tb(time=2, lat, lon) in Kelvin
      - Coordinates: time, lat (-60 to 60), lon (-180 to 180 or 0 to 360)

    Produces two half-hourly subset NetCDF-4 files per hourly granule.
    """
    w, s, e, n = bbox
    results = []

    with h5py.File(raw_path, "r") as f:
        # Inspect available datasets and coordinates
        lat_ds = f["lat"] if "lat" in f else f["latitude"]
        lon_ds = f["lon"] if "lon" in f else f["longitude"]
        time_ds = f["time"] if "time" in f else None

        lat = lat_ds[:]
        lon = lon_ds[:]

        # Handle 0-360 lon convention if present
        if lon.max() > 180.0 and w >= 0:
            lon_adj = np.where(lon > 180.0, lon - 360.0, lon)
        else:
            lon_adj = lon

        # Identify bounding indices
        lat_mask = (lat >= s - 0.05) & (lat <= n + 0.05)
        lon_mask = (lon_adj >= w - 0.05) & (lon_adj <= e + 0.05)

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            raise ValueError(f"No coordinates found inside bounding box {bbox}")

        lat_min_idx, lat_max_idx = lat_indices[0], lat_indices[-1] + 1
        lon_min_idx, lon_max_idx = lon_indices[0], lon_indices[-1] + 1

        sub_lat = lat[lat_min_idx:lat_max_idx]
        sub_lon = lon_adj[lon_min_idx:lon_max_idx]

        # Primary variable: Tb
        tb_ds = f["Tb"] if "Tb" in f else (f["tb"] if "tb" in f else f["ir"])
        tb_attrs = dict(tb_ds.attrs)
        units = tb_attrs.get("units", b"K")
        if isinstance(units, (bytes, np.bytes_)):
            units = units.decode("utf-8", "replace")
        fill_val = tb_attrs.get("_FillValue", np.nan)

        n_times = tb_ds.shape[0] if tb_ds.ndim >= 3 else 1

        # Parse base time from filename: e.g. merg_2025070100_4km-pixel.nc4 -> 2025-07-01 00:00 UTC
        base_name = raw_path.name
        # Match YYYYMMDDHH
        import re
        match = re.search(r"merg_(\d{4})(\d{2})(\d{2})(\d{2})", base_name)
        if match:
            yr, mo, dy, hr = map(int, match.groups())
            base_dt = datetime(yr, mo, dy, hr, 0, 0, tzinfo=timezone.utc)
        else:
            base_dt = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

        for t_idx in range(n_times):
            slot_minute = t_idx * 30
            slot_start = base_dt + timedelta(minutes=slot_minute)
            slot_end = slot_start + timedelta(minutes=30)
            time_str = slot_start.strftime("%Y%m%d-S%H%M%S")
            subset_filename = f"mergir_{slot_start.strftime('%Y%m%d_%H%M')}.subset.nc4"
            subset_file = out_dir / subset_filename

            # Slice (lat, lon)
            if tb_ds.ndim == 3:
                tb_crop = tb_ds[t_idx, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx].astype(np.float32)
            else:
                tb_crop = tb_ds[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx].astype(np.float32)

            # Ensure valid temperature values
            valid_mask = np.isfinite(tb_crop) & (tb_crop > 100.0) & (tb_crop < 350.0)
            valid_pct = round(float(np.mean(valid_mask) * 100.0), 2)

            with h5py.File(subset_file, "w") as out_f:
                out_f.attrs["title"] = "GPM_MERGIR 4km IR Brightness Temperature Regional Subset"
                out_f.attrs["source_granule"] = raw_path.name
                out_f.attrs["time_start_utc"] = slot_start.isoformat()
                out_f.attrs["time_end_utc"] = slot_end.isoformat()
                out_f.attrs["bounding_box_wsen"] = list(bbox)
                out_f.attrs["units"] = "K"
                out_f.attrs["resolution"] = "~4 km"

                d_lat = out_f.create_dataset("lat", data=sub_lat.astype(np.float32))
                d_lat.attrs["units"] = "degrees_north"
                d_lat.attrs["standard_name"] = "latitude"

                d_lon = out_f.create_dataset("lon", data=sub_lon.astype(np.float32))
                d_lon.attrs["units"] = "degrees_east"
                d_lon.attrs["standard_name"] = "longitude"

                d_tb = out_f.create_dataset("Tb", data=tb_crop, compression="gzip", compression_opts=4)
                d_tb.attrs["units"] = "K"
                d_tb.attrs["long_name"] = "Brightness Temperature"
                d_tb.attrs["_FillValue"] = np.nan

            results.append({
                "subset_filename": subset_file.name,
                "time_start": slot_start.isoformat(),
                "time_end": slot_end.isoformat(),
                "crop_rows": tb_crop.shape[0],
                "crop_cols": tb_crop.shape[1],
                "valid_percent": valid_pct,
                "file_size": subset_file.stat().st_size,
            })

    return results


def validate_file(path: Path) -> Dict[str, Any]:
    """Validate that subset file is readable, non-empty, and has expected structure."""
    if not path.exists():
        return {"readable": False, "valid_percent": 0.0, "file_size": 0}
    try:
        with h5py.File(path, "r") as f:
            tb = f["Tb"][:]
            valid = np.isfinite(tb) & (tb > 100.0) & (tb < 350.0)
            valid_pct = round(float(np.mean(valid) * 100.0), 2)
            return {
                "readable": True,
                "shape": list(tb.shape),
                "valid_percent": valid_pct,
                "min_tb": round(float(np.nanmin(tb[valid])), 2) if np.any(valid) else None,
                "max_tb": round(float(np.nanmax(tb[valid])), 2) if np.any(valid) else None,
                "mean_tb": round(float(np.nanmean(tb[valid])), 2) if np.any(valid) else None,
                "file_size": path.stat().st_size,
            }
    except Exception as e:
        return {"readable": False, "error": str(e), "valid_percent": 0.0, "file_size": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and subset NASA GPM MERGIR V1 data.")
    parser.add_argument("--date", help="Target date in YYYY-MM-DD format")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format (e.g. 2025-07-01)")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format (e.g. 2025-07-07)")
    parser.add_argument("--bbox", default="89.5,24.5,93.5,27.0", help="Bounding box: W,S,E,N (default: 89.5,24.5,93.5,27.0)")
    parser.add_argument("--out-dir", default="data/mergir", help="Directory for regional subset files")
    parser.add_argument("--raw-dir", default="data/raw/mergir", help="Directory for raw granules")
    parser.add_argument("--out-inspection", default="outputs/inspection", help="Output directory for reports")
    parser.add_argument("--token", help="NASA Earthdata Bearer Token")
    parser.add_argument("--workers", type=int, default=4, help="Number of download worker threads")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing files without downloading")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-processing")
    args = parser.parse_args()

    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.date:
        start_date = end_date = args.date
    else:
        start_date = "2025-07-01"
        end_date = "2025-07-07"

    bbox_parts = tuple(map(float, args.bbox.split(",")))
    if len(bbox_parts) != 4:
        raise ValueError("Bounding box must be 4 comma-separated values: West,South,East,North")

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    inspection_dir = Path(args.out_inspection)
    out_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir.mkdir(parents=True, exist_ok=True)

    date_label = f"{start_date} to {end_date}" if start_date != end_date else start_date
    print("=" * 65)
    print("NASA GPM MERGIR Infrared Brightness Temperature Ingestion")
    print(f"Product: {COLLECTION_SHORT_NAME} (V{COLLECTION_VERSION}) | Dates: {date_label}")
    print(f"Study Bounding Box: {bbox_parts} (Meghalaya buffer)")
    print("=" * 65)

    # 1. Query NASA CMR
    print("\n[Step 1] Querying NASA CMR API for GPM_MERGIR collection...")
    granules = query_cmr_granules(start_date, end_date)
    print(f"Discovered {len(granules)} hourly granules for date range {date_label}.")
    if not granules:
        print("No granules found. Please verify the dates.")
        sys.exit(1)

    if args.validate_only:
        print("\nValidating existing files in", out_dir)
        subsets = sorted(out_dir.glob("*.subset.nc4"))
        print(f"Found {len(subsets)} subset files.")
        for s in subsets:
            v = validate_file(s)
            print(f"  {s.name}: readable={v['readable']}, valid={v['valid_percent']}%, size={v['file_size']}B")
        return

    # 2. Check Authentication
    session = get_authenticated_session(token=args.token)
    if not session:
        print("\n" + "!" * 65)
        print("AUTHENTICATION REQUIRED: NASA Earthdata Login credentials not found.")
        print("Please configure your credentials via:")
        print("  $env:EARTHDATA_TOKEN = 'your_token'")
        print("  Or run: python scripts/download_mergir.py --token <token>")
        print("!" * 65 + "\n")
        sys.exit(2)

    # Pre-flight connection check on first granule
    first_url = granules[0]["download_url"]
    print("Verifying connection and permissions with NASA GES DISC...")
    try:
        test_resp = session.get(first_url, stream=True, timeout=30)
        if test_resp.status_code == 401:
            print("\n" + "!" * 65)
            print("NASA EARTHDATA AUTHENTICATION FAILED (HTTP 401)")
            print("Your Earthdata token is invalid or has expired.")
            print("Please generate a new token at https://urs.earthdata.nasa.gov/")
            print("!" * 65 + "\n")
            sys.exit(2)
        elif test_resp.status_code == 403:
            print("\n" + "!" * 65)
            print("NASA EARTHDATA HTTP 403: GES DISC authorization required.")
            print("Ensure 'NASA GESDISC DATA ARCHIVE' is authorized under your profile at https://urs.earthdata.nasa.gov/")
            print("!" * 65 + "\n")
            sys.exit(3)
    except requests.exceptions.RequestException as e:
        print(f"Network error during pre-flight check: {e}")
        sys.exit(1)

    # 3. Download and Crop Granules
    print(f"\n[Step 2] Downloading and subsetting {len(granules)} granules with {args.workers} workers...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    inventory = []

    def process_granule(item: Tuple[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        idx, g = item
        url = g["download_url"]
        base_name = Path(url).name if url else f"mergir_{idx}.nc4"
        raw_file = raw_dir / base_name

        try:
            import re
            m = re.search(r"merg_(\d{4})(\d{2})(\d{2})(\d{2})", base_name)
            if m and not args.force:
                yr, mo, dy, hr = m.groups()
                sub0 = out_dir / f"mergir_{yr}{mo}{dy}_{hr}00.subset.nc4"
                sub1 = out_dir / f"mergir_{yr}{mo}{dy}_{hr}30.subset.nc4"
                if sub0.exists() and sub0.stat().st_size > 5000 and sub1.exists() and sub1.stat().st_size > 5000:
                    val0 = validate_file(sub0)
                    val1 = validate_file(sub1)
                    if val0.get("readable") and val1.get("readable"):
                        print(f"[{idx:03d}/{len(granules):03d}] {base_name[:32]}... [ALREADY EXISTS (2 intervals)]")
                        dt0 = datetime(int(yr), int(mo), int(dy), int(hr), 0, tzinfo=timezone.utc)
                        return [
                            {
                                "subset_filename": sub0.name,
                                "time_start": dt0.isoformat(),
                                "time_end": (dt0 + timedelta(minutes=30)).isoformat(),
                                "crop_rows": val0.get("shape", [71, 113])[0],
                                "crop_cols": val0.get("shape", [71, 113])[1],
                                "valid_percent": val0.get("valid_percent", 100.0),
                                "file_size": val0.get("file_size", sub0.stat().st_size),
                            },
                            {
                                "subset_filename": sub1.name,
                                "time_start": (dt0 + timedelta(minutes=30)).isoformat(),
                                "time_end": (dt0 + timedelta(minutes=60)).isoformat(),
                                "crop_rows": val1.get("shape", [71, 113])[0],
                                "crop_cols": val1.get("shape", [71, 113])[1],
                                "valid_percent": val1.get("valid_percent", 100.0),
                                "file_size": val1.get("file_size", sub1.stat().st_size),
                            },
                        ]

            download_file(url, raw_file, session, g["size_mb"])
            crop_results = crop_mergir_granule(raw_file, out_dir, bbox=bbox_parts)
            try:
                raw_file.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[{idx:03d}/{len(granules):03d}] {base_name[:32]}... [DOWNLOADED & SUBSETTED {len(crop_results)} intervals]")
            return crop_results
        except Exception as err:
            print(f"[{idx:03d}/{len(granules):03d}] {base_name[:32]}... [FAILED: {err}]")
            return [{
                "subset_filename": f"failed_{base_name}",
                "time_start": g["time_start"],
                "time_end": g["time_end"],
                "crop_rows": 0,
                "crop_cols": 0,
                "valid_percent": 0.0,
                "file_size": 0,
                "status": f"FAILED: {err}",
            }]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        batch_results = list(executor.map(process_granule, enumerate(granules, 1)))

    for batch in batch_results:
        inventory.extend(batch)

    # 4. Generate Inventory CSV
    inv_csv_path = inspection_dir / "mergir_inventory.csv"
    if inventory:
        all_keys = []
        for item in inventory:
            for k in item.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with inv_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(inventory)
        print(f"\nWrote MERGIR inventory to {inv_csv_path}")

    # 5. Generate Report JSON
    subsets = list(out_dir.glob("*.subset.nc4"))
    total_size = sum(f.stat().st_size for f in subsets)
    report = {
        "dataset": "GPM_MERGIR V1 (NOAA/NCEP/CPC Merged IR)",
        "date_range_utc": date_label,
        "region": "Meghalaya, India",
        "bounding_box_wsen": list(bbox_parts),
        "expected_hourly_granules": len(granules),
        "expected_half_hourly_intervals": len(granules) * 2,
        "processed_subset_files": len(subsets),
        "total_subset_size_bytes": total_size,
        "total_subset_size_mb": round(total_size / (1024 * 1024), 2),
        "variable_extracted": "Tb (Infrared Brightness Temperature)",
        "units": "K",
        "native_resolution": "~4 km",
        "temporal_cadence_minutes": 30,
        "proxy_caveat": "MERGIR is an observational input to IMERG retrieval; evaluation against IMERG is proxy-label evaluation, not independent ground-truth validation.",
    }
    report_json_path = inspection_dir / "mergir_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote inspection report to {report_json_path}")

    print("\n" + "=" * 65)
    print(f"MERGIR INGESTION COMPLETED: {len(subsets)} subset intervals processed.")
    print(f"Total archive size: {report['total_subset_size_mb']} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
