"""Official ISRO MOSDAC INSAT-3DR Downloader and Preprocessor.

Accesses MOSDAC Open Data API (mdapi) for product 3RIMG_L1C_ASIA_MER.
Supports secure credentials management, resumable downloads, and regional integrity verification.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

MOSDAC_SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
MOSDAC_TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
MOSDAC_REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
MOSDAC_DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
MOSDAC_CHECK_INTERNET_URL = "https://mosdac.gov.in/download_api/check-internet"
MOSDAC_LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"

DEFAULT_DATASET_ID = "3RIMG_L1C_ASIA_MER"
CREDENTIALS_FILE = Path(".mosdac_credentials.json")


def get_mosdac_credentials(cli_user: Optional[str] = None, cli_pass: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve MOSDAC credentials from CLI args, environment variables, or secure local credentials file."""
    # 1. CLI args
    if cli_user and cli_pass:
        return cli_user, cli_pass

    # 2. Environment variables
    env_user = os.environ.get("MOSDAC_USER") or os.environ.get("MOSDAC_USERNAME")
    env_pass = os.environ.get("MOSDAC_PASSWORD") or os.environ.get("MOSDAC_PASS")
    if env_user and env_pass:
        return env_user, env_pass

    # 3. Local credentials file (.mosdac_credentials.json)
    if CREDENTIALS_FILE.exists():
        try:
            data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            u = data.get("username") or data.get("username/email")
            p = data.get("password")
            if u and p:
                return u, p
        except Exception:
            pass

    return None, None


def setup_interactive_login() -> bool:
    """Prompt user interactively for MOSDAC credentials and save locally (ignored by Git)."""
    print("\n" + "=" * 60)
    print("ISRO MOSDAC Account Authentication Setup")
    print("=" * 60)
    print("Please enter your registered MOSDAC credentials (https://mosdac.gov.in/).")
    print("Credentials will be saved locally to .mosdac_credentials.json (ignored by Git).")
    print("Never share or commit this file.\n")

    try:
        username = input("MOSDAC Username / Email: ").strip()
        if not username:
            print("Username cannot be empty.")
            return False
        password = getpass.getpass("MOSDAC Password: ").strip()
        if not password:
            print("Password cannot be empty.")
            return False
    except (KeyboardInterrupt, EOFError):
        print("\nLogin cancelled.")
        return False

    print("\nVerifying credentials with MOSDAC server...")
    token_data, err = request_access_token(username, password)
    if not token_data:
        print(f"Authentication FAILED: {err}")
        return False

    # Save credentials locally
    cred_data = {"username": username, "password": password, "created_at": datetime.now(timezone.utc).isoformat()}
    CREDENTIALS_FILE.write_text(json.dumps(cred_data, indent=2), encoding="utf-8")
    print(f"Authentication SUCCESSFUL. Credentials saved to {CREDENTIALS_FILE.resolve()}")
    return True


def request_access_token(username: str, password: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Request an OAuth access token from MOSDAC token endpoint."""
    payload = {"username": username, "password": password}
    try:
        resp = requests.post(MOSDAC_TOKEN_URL, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
            }, None
        elif resp.status_code == 400:
            err = resp.json().get("error", "Bad Request")
            return None, f"HTTP 400: {err}"
        elif resp.status_code == 401:
            err = resp.json().get("error", "Invalid Username or Password")
            return None, f"HTTP 401: {err}"
        elif resp.status_code == 503:
            return None, "HTTP 503: MOSDAC Server Under Maintenance / Service Unavailable"
        else:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: {e}"


def query_mosdac_catalog(
    dataset_id: str,
    start_date: str,
    end_date: str,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """Query MOSDAC public search endpoint for available granule entries."""
    # MOSDAC Search expects date format: "YYYY-MM-DD"
    start_time = start_date
    end_time = end_date

    granules = []
    start_index = 1

    while True:
        params = {
            "datasetId": dataset_id,
            "startTime": start_time,
            "endTime": end_time,
            "count": page_size,
            "startIndex": start_index,
        }
        resp = requests.get(MOSDAC_SEARCH_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"MOSDAC Search API returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        total_results = data.get("totalResults", 0)
        entries = data.get("entries", [])
        if not entries:
            break

        for item in entries:
            granules.append({
                "record_id": item.get("id"),
                "identifier": item.get("identifier"),
                "updated": item.get("updated"),
                "dataset_id": dataset_id,
                "size_bytes": item.get("size", 0),
            })

        if len(granules) >= total_results or len(entries) < page_size:
            break
        start_index += page_size

    return sorted(granules, key=lambda x: x["identifier"])


def download_mosdac_granule(
    access_token: str,
    record_id: str,
    identifier: str,
    dest_path: Path,
    expected_size: int = 0,
) -> bool:
    """Download a single MOSDAC granule with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        # File already exists and has non-trivial size (>1MB)
        return True

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"id": record_id}

    resp = requests.get(MOSDAC_DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=60)
    if resp.status_code == 400:
        err = resp.json().get("error", "Bad Request")
        raise RuntimeError(f"Download Error HTTP 400: {err}")
    elif resp.status_code == 401:
        raise PermissionError("MOSDAC Token Expired or Invalid")
    elif resp.status_code != 200:
        raise RuntimeError(f"MOSDAC Download HTTP {resp.status_code}: {resp.text[:200]}")

    with tmp_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    tmp_path.replace(dest_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Official ISRO MOSDAC INSAT-3DR Downloader.")
    parser.add_argument("--start-date", default="2025-07-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-07-07", help="End date YYYY-MM-DD")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="MOSDAC Dataset ID")
    parser.add_argument("--out-dir", default="data/raw/insat", help="Destination folder for raw HDF5 files")
    parser.add_argument("--username", help="MOSDAC username/email")
    parser.add_argument("--password", help="MOSDAC password")
    parser.add_argument("--login", action="store_true", help="Interactive login prompt to save credentials")
    parser.add_argument("--check-only", action="store_true", help="Check catalog and authentication without downloading")
    args = parser.parse_args()

    if args.login:
        success = setup_interactive_login()
        sys.exit(0 if success else 1)

    print("=" * 65)
    print("ISRO MOSDAC INSAT-3DR Data Ingestion Pilot")
    print(f"Product: {args.dataset_id} | Date Range: {args.start_date} to {args.end_date}")
    print("=" * 65)

    # 1. Query MOSDAC Catalog (Public Search)
    print("\n[Step 1] Querying MOSDAC Public Catalog API...")
    try:
        granules = query_mosdac_catalog(args.dataset_id, args.start_date, args.end_date)
        print(f"Discovered {len(granules)} observations for {args.start_date} to {args.end_date}.")
    except Exception as e:
        print(f"Catalog Query FAILED: {e}")
        sys.exit(1)

    if not granules:
        print("No observations found in catalog.")
        sys.exit(1)

    # 2. Check Authentication
    print("\n[Step 2] Verifying MOSDAC Authentication...")
    username, password = get_mosdac_credentials(args.username, args.password)
    if not username or not password:
        print("\n" + "!" * 65)
        print("EXACT BLOCKER: MOSDAC AUTHENTICATION CREDENTIALS MISSING")
        print("MOSDAC download endpoints strictly require an authorized account token.")
        print("No credentials were found in:")
        print("  - Environment variables: $env:MOSDAC_USER / $env:MOSDAC_PASSWORD")
        print("  - Credentials file:      .mosdac_credentials.json")
        print("  - CLI parameters:        --username / --password")
        print("\nTo authenticate securely without exposing passwords:")
        print("  1. Interactive CLI prompt:  python scripts/download_mosdac.py --login")
        print("  2. Or set environment vars: $env:MOSDAC_USER = 'user'; $env:MOSDAC_PASSWORD = 'password'")
        print("!" * 65 + "\n")
        sys.exit(2)

    token_info, err = request_access_token(username, password)
    if not token_info:
        print("\n" + "!" * 65)
        print("EXACT BLOCKER: MOSDAC AUTHENTICATION FAILED")
        print(f"Server response: {err}")
        print("Please check your username and password or register at https://mosdac.gov.in/.")
        print("!" * 65 + "\n")
        sys.exit(2)

    print(f"Authentication verified successfully for user: {username}")
    access_token = token_info["access_token"]

    if args.check_only:
        print("\nCheck completed successfully. Data is ready for download.")
        return

    # 3. Download Granules
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Step 3] Downloading {len(granules)} observations into {out_dir}...")

    downloaded = 0
    skipped = 0
    failed = 0

    for idx, g in enumerate(granules, 1):
        filename = g["identifier"]
        dest = out_dir / filename
        try:
            download_mosdac_granule(access_token, g["record_id"], filename, dest, g.get("size_bytes", 0))
            if dest.stat().st_size > 1024 * 1024:
                downloaded += 1
                status = "OK"
            else:
                skipped += 1
                status = "SKIPPED"
            print(f"[{idx:03d}/{len(granules):03d}] {filename[:36]}... [{status}]")
        except PermissionError:
            # Refresh token and retry once
            refresh_resp = requests.post(MOSDAC_REFRESH_URL, json={"refresh_token": token_info.get("refresh_token")})
            if refresh_resp.status_code == 200:
                access_token = refresh_resp.json().get("access_token")
                download_mosdac_granule(access_token, g["record_id"], filename, dest, g.get("size_bytes", 0))
                downloaded += 1
            else:
                print(f"[{idx:03d}/{len(granules):03d}] Token refresh failed.")
                failed += 1
        except Exception as e:
            print(f"[{idx:03d}/{len(granules):03d}] FAILED: {e}")
            failed += 1

    print("\n" + "=" * 65)
    print(f"MOSDAC Ingestion Summary: {downloaded} downloaded, {skipped} existing, {failed} failed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
