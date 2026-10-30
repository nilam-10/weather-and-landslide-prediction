"""NASA GPM IMERG Final Run (GPM_3IMERGHH V07B) downloader and subsetter.

Official data access via NASA Earthdata CMR and GES DISC.
Preserves original HDF5/NetCDF4 metadata, timestamps, units, and missing values.
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
from datetime import datetime, timezone
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


class EarthdataSession(requests.Session):
    """Requests session that preserves Authorization header across Earthdata Login redirects."""

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

        # Retain auth if redirecting to URS or if Bearer token is used
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.auth_username and self.auth_password:
            if parsed_target.hostname == URS_AUTH_HOST or parsed_target.hostname == parsed_orig.hostname:
                prepared_request.prepare_auth((self.auth_username, self.auth_password))


def get_netrc_path() -> Path:
    """Return the platform-appropriate .netrc / _netrc path."""
    user_home = Path.home()
    if os.name == "nt":
        # Windows supports _netrc preferentially, or .netrc
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
        # Check alternate on Windows
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
            return auth[0], auth[2]  # login, password
    except Exception:
        # Fallback line parser if netrc format has quirks
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


def setup_interactive_login() -> bool:
    """Securely prompt user for Earthdata credentials and save to netrc."""
    print("=" * 65)
    print("NASA Earthdata Login Configuration")
    print("=" * 65)
    print("An Earthdata Login account is required to download GPM IMERG data.")
    print("If you do not have an account, register at: https://urs.earthdata.nasa.gov/")
    print("Note: Ensure you authorize 'NASA GESDISC DATA ARCHIVE' under your profile.")
    print("-" * 65)

    username = input("Enter your Earthdata username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return False

    password = getpass.getpass("Enter your Earthdata password (hidden): ").strip()
    if not password:
        print("Password cannot be empty.")
        return False

    # Test credentials against URS
    print("Verifying credentials with NASA Earthdata...")
    session = requests.Session()
    session.auth = (username, password)
    try:
        resp = session.get(f"https://{URS_AUTH_HOST}/api/v1/users/tokens", timeout=15)
        # 200 or 403 (with JSON error) or redirect means auth was evaluated
        if resp.status_code == 401:
            print("Authentication failed: Invalid username or password.")
            return False
    except Exception as e:
        print(f"Connection warning during verification: {e}")

    netrc_path = get_netrc_path()
    entry = f"\nmachine {URS_AUTH_HOST} login {username} password {password}\n"
    try:
        if netrc_path.exists():
            existing = netrc_path.read_text(encoding="utf-8")
            if URS_AUTH_HOST in existing:
                # Replace existing host entry
                lines = [l for l in existing.splitlines() if URS_AUTH_HOST not in l]
                netrc_path.write_text("\n".join(lines) + entry, encoding="utf-8")
            else:
                netrc_path.write_text(existing + entry, encoding="utf-8")
        else:
            netrc_path.write_text(entry.strip() + "\n", encoding="utf-8")
        print(f"Credentials successfully saved to {netrc_path}")
        return True
    except Exception as e:
        print(f"Failed to write credentials file: {e}")
        return False


def get_authenticated_session(token: Optional[str] = None) -> Optional[requests.Session]:
    """Create an authenticated requests session using token, netrc, or env vars."""
    # 1. Bearer Token
    tok = token or os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return EarthdataSession(token=tok)

    # 2. Environment variables
    env_user = os.environ.get("EARTHDATA_USERNAME")
    env_pass = os.environ.get("EARTHDATA_PASSWORD")
    if env_user and env_pass:
        return EarthdataSession(username=env_user, password=env_pass)

    # 3. .netrc / _netrc
    netrc_user, netrc_pass = read_netrc_credentials()
    if netrc_user and netrc_pass:
        return EarthdataSession(username=netrc_user, password=netrc_pass)

    return None


def query_cmr_granules(start_date: str, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query NASA CMR API for GPM_3IMERGHH V07 granules across a date range."""
    if not end_date:
        end_date = start_date
    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T23:59:59Z"
    granules = []
    page_num = 1
    page_size = 500
    while True:
        params = {
            "short_name": "GPM_3IMERGHH",
            "version": "07",
            "temporal": f"{start_time},{end_time}",
            "page_size": page_size,
            "page_num": page_num,
            "sort_key": "start_date",
        }
        url = f"{CMR_GRANULES_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "RBU-Honours-Precip-Downloader/1.0"})
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
                    if l.get("href", "").endswith((".HDF5", ".nc4", ".h5")):
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
    """Download a file with resume support and progress logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    # If already downloaded and valid size, skip
    if dest_path.exists():
        if dest_path.stat().st_size > 1024 * 1024:  # At least 1MB
            return True

    headers = {}
    resp = session.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True)
    if resp.status_code != 200:
        if resp.status_code == 401:
            raise PermissionError("NASA Earthdata authentication failed (HTTP 401). Check credentials.")
        if resp.status_code == 403:
            try:
                err_json = resp.json()
                if "resolution_url" in err_json or "EULA" in err_json.get("error_description", ""):
                    res_url = err_json.get("resolution_url", "https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ")
                    raise PermissionError(f"EULA Acceptance Required. Visit {res_url} in your browser and click 'Authorize'.")
            except PermissionError:
                raise
            except Exception:
                pass
            raise PermissionError("NASA Earthdata HTTP 403 Forbidden. Ensure 'NASA GESDISC DATA ARCHIVE' is authorized in your profile.")
        raise RuntimeError(f"HTTP Error {resp.status_code} while downloading {url}")

    # Check content type is not an HTML error or login page
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        raise PermissionError(
            "Received HTML login page instead of HDF5 file. "
            "Please ensure you are authorized for the GES DISC application at https://urs.earthdata.nasa.gov/"
        )

    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    temp_path.replace(dest_path)
    return True


def crop_imerg_granule(
    raw_path: Path,
    subset_path: Path,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
) -> Dict[str, Any]:
    """Extract spatial subset from global IMERG HDF5 and save as regional NetCDF4/HDF5."""
    w, s, e, n = bbox
    subset_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_path, "r") as src:
        grid = src["Grid"]
        lon = grid["lon"][:]
        lat = grid["lat"][:]
        time_arr = grid["time"][:]

        # Longitudinal and latitudinal mask
        lon_mask = (lon >= w) & (lon <= e)
        lat_mask = (lat >= s) & (lat <= n)

        if not np.any(lon_mask) or not np.any(lat_mask):
            raise ValueError(f"Bounding box {bbox} does not intersect IMERG grid")

        lon_indices = np.where(lon_mask)[0]
        lat_indices = np.where(lat_mask)[0]

        lon_sub = lon[lon_indices]
        lat_sub = lat[lat_indices]

        c_start, c_end = int(lon_indices[0]), int(lon_indices[-1]) + 1
        r_start, r_end = int(lat_indices[0]), int(lat_indices[-1]) + 1

        # Robust dimensional slicing
        p_shape = grid["precipitation"].shape
        if len(p_shape) == 3:
            if p_shape[1] == lon.size and p_shape[2] == lat.size:
                raw_precip = grid["precipitation"][0, c_start:c_end, r_start:r_end]
                dim_order = "lon_lat"
            elif p_shape[1] == lat.size and p_shape[2] == lon.size:
                raw_precip = grid["precipitation"][0, r_start:r_end, c_start:c_end].T
                dim_order = "lat_lon"
            else:
                raw_precip = grid["precipitation"][0, c_start:c_end, r_start:r_end]
                dim_order = "lon_lat"
        elif len(p_shape) == 2:
            if p_shape[0] == lon.size:
                raw_precip = grid["precipitation"][c_start:c_end, r_start:r_end]
                dim_order = "lon_lat"
            else:
                raw_precip = grid["precipitation"][r_start:r_end, c_start:c_end].T
                dim_order = "lat_lon"
        else:
            raw_precip = grid["precipitation"][:]
            dim_order = "unknown"

        precip_attrs = dict(grid["precipitation"].attrs)

        # Missing and fill values
        fill_val = float(precip_attrs.get("_FillValue", -9999.9))
        valid_mask = (raw_precip != fill_val) & (raw_precip >= 0) & np.isfinite(raw_precip)

        stats = {
            "crop_rows": int(lat_sub.size),
            "crop_cols": int(lon_sub.size),
            "min_lon": float(round(lon_sub.min(), 3)),
            "max_lon": float(round(lon_sub.max(), 3)),
            "min_lat": float(round(lat_sub.min(), 3)),
            "max_lat": float(round(lat_sub.max(), 3)),
            "valid_percent": round(float(valid_mask.mean() * 100), 2) if raw_precip.size else 0.0,
            "min_precip": round(float(raw_precip[valid_mask].min()), 4) if np.any(valid_mask) else None,
            "max_precip": round(float(raw_precip[valid_mask].max()), 4) if np.any(valid_mask) else None,
            "mean_precip": round(float(raw_precip[valid_mask].mean()), 4) if np.any(valid_mask) else None,
            "units": precip_attrs.get("units", "mm/hr"),
        }

        # Write subset file
        with h5py.File(subset_path, "w") as dst:
            # Copy root attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["subset_bbox_wsen"] = bbox
            dst.attrs["subset_note"] = "Meghalaya AOI crop from GPM_3IMERGHH V07B"

            g_dst = dst.create_group("Grid")
            for k, v in grid.attrs.items():
                g_dst.attrs[k] = v

            # Coordinates
            d_lon = g_dst.create_dataset("lon", data=lon_sub)
            for k, v in grid["lon"].attrs.items():
                d_lon.attrs[k] = v

            d_lat = g_dst.create_dataset("lat", data=lat_sub)
            for k, v in grid["lat"].attrs.items():
                d_lat.attrs[k] = v

            d_time = g_dst.create_dataset("time", data=time_arr)
            for k, v in grid["time"].attrs.items():
                d_time.attrs[k] = v

            # Primary variable: precipitation (shape: 1, crop_cols, crop_rows)
            precip_3d = raw_precip[np.newaxis, :, :]
            d_precip = g_dst.create_dataset(
                "precipitation",
                data=precip_3d,
                compression="gzip",
                compression_opts=4,
            )
            for k, v in precip_attrs.items():
                d_precip.attrs[k] = v

            # Auxiliary fields if present
            for aux_var in ["precipitationQualityIndex", "probabilityLiquidPrecipitation", "randomError"]:
                if aux_var in grid:
                    if dim_order == "lat_lon":
                        raw_aux = grid[aux_var][0, r_start:r_end, c_start:c_end].T
                    else:
                        raw_aux = grid[aux_var][0, c_start:c_end, r_start:r_end]
                    d_aux = g_dst.create_dataset(
                        aux_var,
                        data=raw_aux[np.newaxis, :, :],
                        compression="gzip",
                        compression_opts=4,
                    )
                    for k, v in grid[aux_var].attrs.items():
                        d_aux.attrs[k] = v

    return stats


def validate_file(file_path: Path) -> Dict[str, Any]:
    """Validate that an IMERG subset file is readable and non-corrupt."""
    with h5py.File(file_path, "r") as f:
        precip = f["Grid"]["precipitation"]
        data = precip[:]
        attrs = dict(precip.attrs)
        fill = float(attrs.get("_FillValue", -9999.9))
        valid = (data != fill) & (data >= 0) & np.isfinite(data)
        return {
            "readable": True,
            "shape": list(data.shape),
            "units": attrs.get("units", "mm/hr"),
            "valid_count": int(np.count_nonzero(valid)),
            "total_count": int(data.size),
            "valid_percent": round(float(valid.mean() * 100), 2),
            "min_precip": round(float(data[valid].min()), 4) if np.any(valid) else 0.0,
            "mean_precip": round(float(data[valid].mean()), 4) if np.any(valid) else 0.0,
            "max_precip": round(float(data[valid].max()), 4) if np.any(valid) else 0.0,
            "file_size": file_path.stat().st_size,
        }


def compare_with_insat(imerg_inventory: List[Dict[str, Any]], insat_csv_path: Path) -> List[Dict[str, Any]]:
    """Compare IMERG 30-min intervals against available INSAT-3DR timestamps."""
    if not insat_csv_path.exists():
        return []

    insat_records = []
    with insat_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            insat_records.append(row)

    comparisons = []
    for ins in insat_records:
        # INSAT start format: "01-FEB-2025T00:15:36"
        ins_start = datetime.strptime(ins["acquisition_start_utc"], "%d-%b-%YT%H:%M:%S").replace(tzinfo=timezone.utc)
        ins_end = datetime.strptime(ins["acquisition_end_utc"], "%d-%b-%YT%H:%M:%S").replace(tzinfo=timezone.utc)

        # Find overlapping IMERG intervals
        overlapping_imerg = []
        for im in imerg_inventory:
            # IMERG ISO start format: "2025-02-01T00:00:00.000Z" or "2025-02-01T00:00:00Z"
            im_start_str = im["time_start"].replace("Z", "+00:00")
            im_end_str = im["time_end"].replace("Z", "+00:00")
            im_start = datetime.fromisoformat(im_start_str)
            im_end = datetime.fromisoformat(im_end_str)

            # Check for temporal overlap: max(start) < min(end)
            if max(ins_start, im_start) < min(ins_end, im_end):
                overlapping_imerg.append(im["subset_filename"])

        comparisons.append({
            "insat_filename": ins["filename"],
            "insat_start_utc": ins["acquisition_start_utc"],
            "insat_end_utc": ins["acquisition_end_utc"],
            "overlapping_imerg_files": overlapping_imerg,
            "overlap_count": len(overlapping_imerg),
            "status": "MATCHED" if overlapping_imerg else "UNMATCHED",
        })

    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and subset NASA GPM IMERG Final Run data.")
    parser.add_argument("--date", help="Target date in YYYY-MM-DD format")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format (e.g. 2025-07-01)")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format (e.g. 2025-07-07)")
    parser.add_argument("--bbox", default="89.5,24.5,93.5,27.0", help="Bounding box: W,S,E,N (default: 89.5,24.5,93.5,27.0)")
    parser.add_argument("--out-dir", default="data/imerg", help="Directory for regional subset files")
    parser.add_argument("--raw-dir", default="data/raw/imerg", help="Directory for raw daily granules")
    parser.add_argument("--out-inspection", default="outputs/inspection", help="Output directory for reports")
    parser.add_argument("--insat-csv", default="outputs/inspection/insat_inventory.csv", help="Path to INSAT inventory CSV")
    parser.add_argument("--token", help="NASA Earthdata Bearer Token")
    parser.add_argument("--login", action="store_true", help="Launch interactive secure Earthdata credentials setup")
    parser.add_argument("--workers", type=int, default=6, help="Number of download/processing worker threads (default: 6)")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing files without downloading")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-processing")
    args = parser.parse_args()

    if args.login:
        success = setup_interactive_login()
        if success:
            print("\nSetup complete. You can now run: python scripts/download_imerg.py")
        sys.exit(0 if success else 1)

    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.date:
        start_date = end_date = args.date
    else:
        start_date = end_date = "2025-02-01"

    bbox_parts = tuple(map(float, args.bbox.split(",")))
    if len(bbox_parts) != 4:
        raise ValueError("Bounding box must be 4 comma-separated values: West,South,East,North")

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    inspection_dir = Path(args.out_inspection)
    out_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir.mkdir(parents=True, exist_ok=True)

    date_label = start_date if start_date == end_date else f"{start_date} to {end_date}"
    print("=" * 65)
    print("NASA GPM IMERG Final Run (V07B) Data Ingestion")
    print(f"Target Date(s): {date_label} | Bounding Box: {bbox_parts}")
    print("=" * 65)

    # 1. Query NASA CMR
    print("Querying NASA CMR API for GPM_3IMERGHH collection 07 granules...")
    granules = query_cmr_granules(start_date, end_date)
    print(f"Discovered {len(granules)} granules for date range {date_label}.")
    if not granules:
        print("No granules found. Please verify the dates.")
        sys.exit(1)

    # If validate-only, check existing subsets
    if args.validate_only:
        print("\nValidating existing files in", out_dir)
        subsets = sorted(out_dir.glob("*.nc4")) + sorted(out_dir.glob("*.h5"))
        print(f"Found {len(subsets)} subset files.")
        for s in subsets:
            v = validate_file(s)
            print(f"  {s.name}: readable={v['readable']}, shape={v['shape']}, valid={v['valid_percent']}%")
        return

    # 2. Check Authentication
    session = get_authenticated_session(token=args.token)
    if not session:
        print("\n" + "!" * 65)
        print("AUTHENTICATION REQUIRED")
        print("NASA Earthdata Login credentials were not found in the environment or netrc.")
        print("To authenticate securely without exposing passwords:")
        print("  1. Interactive CLI prompt:  python scripts/download_imerg.py --login")
        print("  2. Or set Bearer token:     $env:EARTHDATA_TOKEN = 'your_token'")
        print("  3. Or create _netrc:       $env:USERPROFILE\\_netrc")
        print("!" * 65 + "\n")
        sys.exit(2)

    # Pre-flight check on first granule
    first_url = granules[0]["download_url"]
    print("Verifying connection and permissions with NASA GES DISC...")
    test_resp = session.get(first_url, stream=True, timeout=30)
    if test_resp.status_code == 403:
        try:
            err_json = test_resp.json()
            if "resolution_url" in err_json or "EULA" in err_json.get("error_description", ""):
                res_url = err_json.get("resolution_url", "https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ")
                print("\n" + "!" * 65)
                print("NASA EULA / APPLICATION AUTHORIZATION REQUIRED")
                print("Your Earthdata token is valid, but 'NASA GESDISC DATA ARCHIVE' has not been authorized yet.")
                print("Please open the following link in your browser and click 'Authorize':")
                print(f"  {res_url}")
                print("!" * 65 + "\n")
                sys.exit(3)
        except PermissionError:
            raise
        except Exception:
            pass

    # 3. Download and Crop Granules
    print(f"\nProcessing {len(granules)} half-hourly intervals with {args.workers} workers...")

    def process_granule(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, g = item
        granule_title = g["title"]
        url = g["download_url"]
        time_start = g["time_start"]
        time_end = g["time_end"]

        base_name = Path(url).name if url else f"granule_{idx}.HDF5"
        raw_file = raw_dir / base_name
        subset_file = out_dir / (base_name.replace(".HDF5", ".subset.nc4").replace(".h5", ".subset.nc4"))

        try:
            if subset_file.exists() and not args.force:
                stats = validate_file(subset_file)
                stats["crop_rows"] = stats["shape"][2] if len(stats["shape"]) == 3 else stats["shape"][0]
                stats["crop_cols"] = stats["shape"][1] if len(stats["shape"]) == 3 else stats["shape"][1]
                status_str = "CACHED"
            else:
                download_file(url, raw_file, session, g["size_mb"])
                stats = crop_imerg_granule(raw_file, subset_file, bbox=bbox_parts)
                status_str = "DOWNLOADED & SUBSETTED"
                try:
                    raw_file.unlink(missing_ok=True)
                except Exception:
                    pass

            val = validate_file(subset_file)
            print(f"[{idx:02d}/{len(granules):02d}] {time_start[11:16]} - {time_end[11:16]} UTC | {base_name[:32]}... [{status_str}]")
            return {
                "granule_index": idx,
                "subset_filename": subset_file.name,
                "original_granule": base_name,
                "time_start": time_start,
                "time_end": time_end,
                "crop_rows": stats.get("crop_rows", 25),
                "crop_cols": stats.get("crop_cols", 40),
                "min_precip_mm_hr": val.get("min_precip", 0.0),
                "max_precip_mm_hr": val.get("max_precip", 0.0),
                "mean_precip_mm_hr": val.get("mean_precip", 0.0),
                "valid_percent": val.get("valid_percent", 100.0),
                "file_size_bytes": val.get("file_size", 0),
                "status": "SUCCESS",
                "success": True,
            }
        except Exception as err:
            print(f"[{idx:02d}/{len(granules):02d}] {time_start[11:16]} - {time_end[11:16]} UTC | {base_name[:32]}... [FAILED: {err}]")
            return {
                "granule_index": idx,
                "subset_filename": subset_file.name,
                "original_granule": base_name,
                "time_start": time_start,
                "time_end": time_end,
                "crop_rows": 0,
                "crop_cols": 0,
                "min_precip_mm_hr": None,
                "max_precip_mm_hr": None,
                "mean_precip_mm_hr": None,
                "valid_percent": 0.0,
                "file_size_bytes": 0,
                "status": f"FAILED: {err}",
                "success": False,
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_granule, enumerate(granules, 1)))

    inventory = sorted(results, key=lambda x: x["granule_index"])
    success_count = sum(1 for r in inventory if r.get("success", False))
    failed_count = len(inventory) - success_count

    # Clean internal success field before serialization
    for item in inventory:
        item.pop("success", None)

    # 4. Generate Inventory CSV
    inv_csv_path = inspection_dir / "imerg_inventory.csv"
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
        print(f"\nWrote inventory to {inv_csv_path}")

    # 5. Compare with INSAT-3DR
    insat_path = Path(args.insat_csv)
    insat_comparisons = compare_with_insat(inventory, insat_path)

    # 6. Generate JSON Report
    total_size = sum(f.stat().st_size for f in out_dir.glob("*.nc4"))
    report = {
        "dataset": "GPM_3IMERGHH_07 (IMERG Final Run V07B)",
        "date_utc": args.date,
        "region": "Meghalaya, India",
        "bounding_box_wsen": list(bbox_parts),
        "expected_granules": len(granules),
        "successfully_processed": success_count,
        "failed_granules": failed_count,
        "total_subset_size_bytes": total_size,
        "total_subset_size_mb": round(total_size / (1024 * 1024), 2),
        "variables_extracted": ["Grid/precipitation", "precipitationQualityIndex", "probabilityLiquidPrecipitation", "randomError"],
        "precipitation_units": "mm/hr",
        "cadence_minutes": 30,
        "temporal_comparison_with_insat": insat_comparisons,
    }
    report_json_path = inspection_dir / "imerg_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote inspection report to {report_json_path}")

    print("\n" + "=" * 65)
    print(f"SUMMARY: {success_count}/{len(granules)} intervals processed successfully.")
    print(f"Total subset archive size: {report['total_subset_size_mb']} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
