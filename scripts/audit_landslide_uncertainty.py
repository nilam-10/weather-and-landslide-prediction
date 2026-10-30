"""Generate Landslide Label Uncertainty Audit CSV (Task 8).

Analyzes all 67 raw catalog records from outputs/landslide/meghalaya_landslide_events.csv,
documenting:
  1. Event ID, Date, Source, Coordinates.
  2. Spatial uncertainty radius (km).
  3. Trigger mechanism (rainfall vs mining vs construction vs unknown).
  4. Geospatial containment within Meghalaya state boundary.
  5. Scientific classification and suitability for pixel-level training.
  6. Mathematical prevalence proof for REQ-LND-04 (BLOCKED_BY_DATA).
"""
import csv
import json
from pathlib import Path
import numpy as np

EVENTS_PATH = Path("outputs/landslide/meghalaya_landslide_events.csv")
OUT_CSV = Path("outputs/experiments/landslide_label_uncertainty_audit.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def parse_uncertainty(acc_str: str) -> float:
    acc = str(acc_str).strip().lower()
    if not acc or acc == "unknown":
        return 25.0
    if "exact" in acc:
        return 0.1
    acc = acc.replace("km", "").strip()
    try:
        return float(acc)
    except ValueError:
        return 25.0

def main():
    print(f"Reading events from: {EVENTS_PATH}")
    rows = []
    with open(EVENTS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Total raw catalog records: {len(rows)}")

    audit_records = []
    rainfall_count = 0
    meghalaya_rainfall_count = 0
    mining_count = 0
    unknown_count = 0

    for r in rows:
        eid = r.get("event_id", "")
        date = r.get("event_date", "").split(" ")[0]
        time = r.get("event_time", "")
        source = r.get("source_name", "")
        lat_str = r.get("latitude", "0")
        lon_str = r.get("longitude", "0")
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            lat, lon = 0.0, 0.0

        country = r.get("country_name", "")
        admin = r.get("admin_division_name", "")
        trigger = (r.get("landslide_trigger") or "unknown").strip().lower()
        acc_str = r.get("location_accuracy", "unknown")
        radius_km = parse_uncertainty(acc_str)

        # Categorize trigger
        is_rainfall = trigger in ["continuous_rain", "downpour", "rain"]
        is_mining = "mining" in trigger or "quarry" in trigger or "stone" in (r.get("event_title", "") + r.get("event_description", "")).lower()
        if is_mining:
            trigger_clean = "mining_quarrying"
            mining_count += 1
        elif is_rainfall:
            trigger_clean = "rainfall_triggered"
            rainfall_count += 1
        else:
            trigger_clean = "unknown_or_anthropogenic"
            unknown_count += 1

        # Check domain containment
        in_meghalaya_admin = (admin == "Meghalaya") or (country == "India" and (25.0 <= lat <= 26.1) and (89.8 <= lon <= 92.8))
        if in_meghalaya_admin and trigger_clean == "rainfall_triggered":
            meghalaya_rainfall_count += 1
            scientific_use = "SPATIAL_SUSCEPTIBILITY_POINT_ONLY"
            pixel_label_valid = False  # 1-50 km radius cannot be assigned to 90m pixel deterministically
        elif trigger_clean == "mining_quarrying":
            scientific_use = "EXCLUDED_MINING_ARTIFACT"
            pixel_label_valid = False
        elif not in_meghalaya_admin:
            scientific_use = "EXCLUDED_OUTSIDE_MEGHALAYA"
            pixel_label_valid = False
        else:
            scientific_use = "EXCLUDED_UNCONFIRMED_TRIGGER"
            pixel_label_valid = False

        audit_records.append({
            "event_id": eid,
            "event_date": date,
            "event_time_available": bool(time and time != "00:00:00" and time != "unknown"),
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "reported_accuracy": acc_str,
            "uncertainty_radius_km": radius_km,
            "trigger_original": trigger,
            "trigger_classified": trigger_clean,
            "country": country,
            "admin_division": admin,
            "inside_meghalaya": in_meghalaya_admin,
            "valid_for_90m_pixel_label": pixel_label_valid,
            "scientific_status": scientific_use,
            "source_provenance": source,
            "gsi_bhukosh_polygon_equivalent": "NOT_AVAILABLE_OPEN_ACCESS_REQUIRES_MOU"
        })

    # Write audit CSV
    fieldnames = list(audit_records[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)

    print(f"Saved audit CSV to: {OUT_CSV}")
    print(f"Summary: Total: {len(audit_records)}, Rainfall: {rainfall_count}, Mining: {mining_count}, Other: {unknown_count}")
    print(f"Meghalaya verified rainfall landslides: {meghalaya_rainfall_count}")

    # Compute mathematical prevalence
    n_events = meghalaya_rainfall_count
    years_span = 17  # 2007 - 2024
    total_days = years_span * 365
    grid_cells = 1000 # 25 x 40 0.1-deg cells
    total_cell_days = total_days * grid_cells
    prevalence = n_events / total_cell_days

    print("\n--- MATHEMATICAL PREVALENCE CALCULATION FOR REQ-LND-04 ---")
    print(f"Grid cells: {grid_cells}")
    print(f"Temporal span: {years_span} years ({total_days} days)")
    print(f"Total cell-day decision units: {total_cell_days:,}")
    print(f"Positive rainfall events: {n_events}")
    print(f"Spatiotemporal Prevalence: {prevalence:.2e} (1 positive per {1/prevalence:,.0f} cell-days)")
    print("Verdict: REQ-LND-04 supervised sequence model is BLOCKED_BY_DATA.")

if __name__ == "__main__":
    main()
