"""Download and audit official historical landslide events for Meghalaya."""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import requests

NASA_GLC_URL = "https://raw.githubusercontent.com/Dinesh2030/landslide-size-prediction/main/nasa_global_landslide_catalog_point.csv"

# Authoritative Meghalaya state bounding box
MEGHALAYA_BBOX = {
    "min_lat": 25.02,
    "max_lat": 26.12,
    "min_lon": 89.81,
    "max_lon": 92.81,
}


def download_nasa_glc(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "nasa_global_landslide_catalog_point.csv"
    if not csv_path.exists():
        print(f"Downloading NASA Global Landslide Catalog from {NASA_GLC_URL}...")
        resp = requests.get(NASA_GLC_URL, timeout=60)
        resp.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded NASA GLC: {csv_path.stat().st_size} bytes")
    else:
        print(f"NASA GLC already cached at: {csv_path}")
    return csv_path


def audit_landslide_inventory(csv_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_india: List[Dict[str, str]] = []
    meghalaya_records: List[Dict[str, str]] = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row.get("country_name", "").strip()
            admin = row.get("admin_division_name", "").strip()
            loc_desc = row.get("location_description", "").strip()
            title = row.get("event_title", "").strip()
            lat_str = row.get("latitude", "").strip()
            lon_str = row.get("longitude", "").strip()

            is_india = ("India" in country) or ("IND" in row.get("country_code", ""))
            if is_india:
                all_india.append(row)

            try:
                lat = float(lat_str)
                lon = float(lon_str)
                in_bbox = (
                    MEGHALAYA_BBOX["min_lat"] <= lat <= MEGHALAYA_BBOX["max_lat"]
                    and MEGHALAYA_BBOX["min_lon"] <= lon <= MEGHALAYA_BBOX["max_lon"]
                )
            except ValueError:
                in_bbox = False

            is_megh_admin = (
                "Meghalaya" in admin
                or "Meghalaya" in loc_desc
                or "Meghalaya" in title
            )

            if in_bbox or is_megh_admin:
                meghalaya_records.append(row)

    # Classify event date reliability
    dated_events: List[Dict[str, str]] = []
    undated_events: List[Dict[str, str]] = []

    for r in meghalaya_records:
        ev_date = r.get("event_date", "").strip()
        if ev_date and ev_date != "None" and ev_date != "null":
            dated_events.append(r)
        else:
            undated_events.append(r)

    # Save Meghalaya inventory subset CSV
    megh_csv_path = output_dir / "meghalaya_landslide_events.csv"
    if meghalaya_records:
        fieldnames = list(meghalaya_records[0].keys())
        with open(megh_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(meghalaya_records)

    # Prepare audit report summary
    report = {
        "dataset_name": "NASA Global Landslide Catalog (GLC) / COOLR",
        "publisher": "NASA Goddard Space Flight Center (GSFC)",
        "source_url": NASA_GLC_URL,
        "total_india_events": len(all_india),
        "total_meghalaya_events": len(meghalaya_records),
        "reliably_dated_events": len(dated_events),
        "undated_or_imprecise_events": len(undated_events),
        "temporal_range": {
            "earliest_event": min([r["event_date"] for r in dated_events]) if dated_events else None,
            "latest_event": max([r["event_date"] for r in dated_events]) if dated_events else None,
        },
        "trigger_breakdown": {},
        "location_accuracy_breakdown": {},
        "scientific_assessment": {
            "can_train_24h_forecasting": False,
            "justification": (
                f"Only {len(dated_events)} reliably dated historical landslide events exist in the authoritative "
                f"NASA catalog within Meghalaya. A supervised 24-hour binary prediction model across a 25x40 or "
                f"fine-resolution daily grid requires hundreds of temporally distributed positive events paired "
                f"with continuous antecedent rainfall records to avoid severe overfitting and unresolvable class imbalance. "
                f"Therefore, scientific honesty dictates that we construct a Static Landslide Susceptibility Model "
                f"(using terrain factors: elevation, slope, aspect, curvature, and lithological/topographic wetness) "
                f"coupled with antecedent rainfall thresholding (heuristic/empirical trigger curves), rather than claiming "
                f"to have trained an unvalidated supervised next-day event probability model."
            ),
            "recommendation": "Build static geomorphic landslide susceptibility mapping first; formulate dynamic hazard as empirical Rainfall-Triggered Susceptibility index."
        },
        "meghalaya_events_list": [
            {
                "event_id": r.get("event_id"),
                "date": r.get("event_date"),
                "time": r.get("event_time"),
                "lat": r.get("latitude"),
                "lon": r.get("longitude"),
                "title": r.get("event_title"),
                "trigger": r.get("landslide_trigger"),
                "accuracy": r.get("location_accuracy"),
                "source": r.get("source_name"),
            }
            for r in meghalaya_records
        ]
    }

    # Populate breakdowns
    for r in meghalaya_records:
        trig = r.get("landslide_trigger", "Unknown") or "Unknown"
        report["trigger_breakdown"][trig] = report["trigger_breakdown"].get(trig, 0) + 1
        acc = r.get("location_accuracy", "Unknown") or "Unknown"
        report["location_accuracy_breakdown"][acc] = report["location_accuracy_breakdown"].get(acc, 0) + 1

    report_path = output_dir / "landslide_inventory_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit completed: {len(meghalaya_records)} events found ({len(dated_events)} dated).")
    print(f"Saved inventory to: {megh_csv_path}")
    print(f"Saved audit report to: {report_path}")
    return report


if __name__ == "__main__":
    csv_file = download_nasa_glc(Path("data/raw/landslides/nasa_glc"))
    audit_landslide_inventory(csv_file, Path("outputs/landslide"))
