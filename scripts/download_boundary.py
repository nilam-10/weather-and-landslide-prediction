"""Download and extract authoritative Meghalaya boundary GeoJSON from geoBoundaries."""
import json
import os
import sys
from pathlib import Path
import requests

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"


def download_meghalaya_boundary(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "meghalaya_boundary.geojson"

    if out_file.exists():
        print(f"Meghalaya boundary already exists at: {out_file}")
        return out_file

    print(f"Fetching metadata from {GEOBOUNDARIES_API}...")
    resp = requests.get(GEOBOUNDARIES_API, timeout=20)
    resp.raise_for_status()
    meta = resp.json()

    dl_url = meta["gjDownloadURL"]
    print(f"Downloading India ADM1 GeoJSON from: {dl_url}...")
    data_resp = requests.get(dl_url, timeout=60)
    data_resp.raise_for_status()
    geojson_data = data_resp.json()

    meghalaya_feature = None
    for f in geojson_data.get("features", []):
        props = f.get("properties", {})
        shape_name = props.get("shapeName", "")
        # Handle variations: Meghalaya, Meghālaya
        if "Megh" in shape_name or "megh" in shape_name.lower():
            meghalaya_feature = f
            print("Found Meghalaya feature in GeoJSON.")
            break

    if not meghalaya_feature:
        raise RuntimeError("Failed to locate Meghalaya in India ADM1 GeoJSON")

    meghalaya_collection = {
        "type": "FeatureCollection",
        "name": "Meghalaya_State_Boundary",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": [meghalaya_feature]
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(meghalaya_collection, f, indent=2)

    print(f"Authoritative Meghalaya boundary successfully saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    download_meghalaya_boundary(Path("data/raw/boundary"))
