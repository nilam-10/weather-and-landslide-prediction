"""Inspect landslide event coordinates against 90m terrain features."""
import csv
import json
from pathlib import Path
import h5py
import matplotlib.path as mpath
import numpy as np


def inspect_events() -> None:
    with h5py.File("data/processed/terrain/meghalaya_terrain_features_90m.nc4", "r") as hf:
        lats = hf["lat"][:]
        lons = hf["lon"][:]
        elev = hf["elevation_mean"][:]
        slope = hf["slope_mean"][:]
        slope_max = hf["slope_max"][:]
        relief = hf["relief"][:]
        curv = hf["curvature_mean"][:]
        m_mask = hf["meghalaya_mask"][:]

    with open("data/raw/boundary/meghalaya_boundary.geojson", "r", encoding="utf-8") as f:
        bdata = json.load(f)
    coords = bdata["features"][0]["geometry"]["coordinates"][0]
    poly_path = mpath.Path(coords)

    events = []
    with open("outputs/landslide/meghalaya_landslide_events.csv", "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            events.append(r)

    inside_events = [
        ev for ev in events
        if poly_path.contains_point([float(ev["longitude"]), float(ev["latitude"])])
    ]

    print(f"Total events in bounding box: {len(events)}")
    print(f"Events strictly inside Meghalaya boundary: {len(inside_events)}")

    for i, ev in enumerate(inside_events):
        lat_ev = float(ev["latitude"])
        lon_ev = float(ev["longitude"])
        r_idx = int(np.argmin(np.abs(lats - lat_ev)))
        c_idx = int(np.argmin(np.abs(lons - lon_ev)))

        # 5x5 window around event (~450m radius)
        r_s = max(0, r_idx - 2)
        r_e = min(len(lats), r_idx + 3)
        c_s = max(0, c_idx - 2)
        c_e = min(len(lons), c_idx + 3)

        pt_elev = float(elev[r_idx, c_idx])
        pt_slope = float(slope[r_idx, c_idx])
        buf_max_slope = float(np.max(slope_max[r_s:r_e, c_s:c_e]))
        buf_relief = float(np.max(relief[r_s:r_e, c_s:c_e]))

        ev_id = ev.get("event_id")
        dt = ev.get("event_date", "")[:10]
        trig = ev.get("landslide_trigger", "")
        acc = ev.get("location_accuracy", "")
        title = ev.get("event_title", "")[:28]

        print(
            f"{i+1:02d}. ID:{ev_id} | Date:{dt} | Trig:{trig:15s} | Acc:{acc:6s} | "
            f"PtSlope:{pt_slope:4.1f}deg | BufMaxSlope:{buf_max_slope:4.1f}deg | "
            f"Elev:{pt_elev:4.0f}m | Relief:{buf_relief:4.0f}m | {title}"
        )


if __name__ == "__main__":
    inspect_events()
