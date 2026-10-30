"""Unit and integration tests for Step 9: Landslide Susceptibility Modeling & Spatial Validation."""
import json
import os
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest
import xgboost as xgb

sys.path.append(os.path.abspath("scripts"))
from train_landslide_susceptibility import (
    DISCLAIMER,
    audit_and_filter_events,
    evaluate_binary_predictions,
    extract_features_for_points,
    load_terrain_rasters,
    sample_pseudo_absence_background,
)


@pytest.fixture
def terrain_data():
    nc_path = Path("data/processed/terrain/meghalaya_terrain_features_90m.nc4")
    return load_terrain_rasters(nc_path)


def test_event_filtering_and_trigger_separation():
    """Verify that events are filtered by state boundary and separated by trigger type."""
    events_csv = Path("outputs/landslide/meghalaya_landslide_events.csv")
    boundary_geojson = Path("data/raw/boundary/meghalaya_boundary.geojson")

    rainfall_events, audit_stats = audit_and_filter_events(events_csv, boundary_geojson)

    # 42 events strictly inside Meghalaya polygon
    assert audit_stats["records_inside_meghalaya"] == 42
    # 25 border records excluded
    assert audit_stats["records_outside_meghalaya_border"] == 25
    # Separated non-rainfall events
    assert audit_stats["trigger_breakdown_inside"]["mining_quarry_triggered"] == 1
    assert audit_stats["trigger_breakdown_inside"]["unknown_trigger"] == 3
    # Exactly 38 rainfall-triggered events
    assert len(rainfall_events) == 38
    assert audit_stats["trigger_breakdown_inside"]["rainfall_triggered"] == 38


def test_pseudo_absence_sampling_buffer(terrain_data):
    """Verify that background points are inside Meghalaya and >= 3 km from known landslide sites."""
    events_csv = Path("outputs/landslide/meghalaya_landslide_events.csv")
    boundary_geojson = Path("data/raw/boundary/meghalaya_boundary.geojson")
    rainfall_events, _ = audit_and_filter_events(events_csv, boundary_geojson)

    bg_samples = sample_pseudo_absence_background(
        terrain_data, rainfall_events, n_samples=50, buffer_radius_pixels=33, seed=42
    )

    assert len(bg_samples) == 50
    lats = terrain_data["lat"]
    lons = terrain_data["lon"]
    m_mask = terrain_data["meghalaya_mask"]

    for lat, lon, block in bg_samples:
        r = int(np.argmin(np.abs(lats - lat)))
        c = int(np.argmin(np.abs(lons - lon)))
        assert m_mask[r, c] == 1, "Background sample outside Meghalaya mask"

        # Check Euclidean distance in pixels to all known events
        for ev in rainfall_events:
            ev_r = int(np.argmin(np.abs(lats - float(ev["latitude"]))))
            ev_c = int(np.argmin(np.abs(lons - float(ev["longitude"]))))
            dist_px = np.sqrt((r - ev_r)**2 + (c - ev_c)**2)
            assert dist_px >= 33, f"Background point too close to event {ev['event_id']} (dist={dist_px:.1f} px)"


def test_feature_extraction_and_spatial_blocks(terrain_data):
    """Verify feature matrix dimensions, blocks, and uncertainty buffer attribution."""
    events_csv = Path("outputs/landslide/meghalaya_landslide_events.csv")
    boundary_geojson = Path("data/raw/boundary/meghalaya_boundary.geojson")
    rainfall_events, _ = audit_and_filter_events(events_csv, boundary_geojson)

    points = []
    for ev in rainfall_events[:5]:
        lon = float(ev["longitude"])
        lat = float(ev["latitude"])
        block = "West" if lon < 91.0 else ("Central" if lon < 92.0 else "East")
        points.append((lat, lon, 1, block))

    points.append((25.5, 91.5, 0, "Central"))  # Dummy control

    X, y, blocks, f_names = extract_features_for_points(points, terrain_data)
    assert X.shape == (6, 8)
    assert len(y) == 6
    assert len(blocks) == 6
    assert len(f_names) == 8
    assert np.all(np.isfinite(X))


def test_susceptibility_model_artifacts_exist():
    """Verify that all Step 9 model checkpoints, metrics, and configs exist."""
    out_dir = Path("outputs/models/landslide_susceptibility")
    assert out_dir.exists()

    required_files = [
        "logistic_regression_lsm.joblib",
        "xgboost_lsm.json",
        "susceptibility_config.json",
        "susceptibility_metrics.json",
        "landslide_susceptibility_evaluation.png",
    ]
    for rf in required_files:
        p = out_dir / rf
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0

    with open(out_dir / "susceptibility_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "spatial_cross_validation_results" in metrics
    assert "xgboost" in metrics["spatial_cross_validation_results"]
    assert metrics["spatial_cross_validation_results"]["xgboost"]["summary"]["mean_roc_auc"] > 0.80
    assert metrics["scientific_assessment_on_24h_objective"]["feasible_with_current_data"] is False


def test_statewide_susceptibility_raster_integrity():
    """Verify statewide 90 m susceptibility raster geometry, values, and disclaimer."""
    nc_path = Path("data/processed/terrain/meghalaya_landslide_susceptibility_90m.nc4")
    assert nc_path.exists()
    assert nc_path.stat().st_size > 1_000_000

    with h5py.File(nc_path, "r") as hf:
        assert hf["lat"].shape == (1380,)
        assert hf["lon"].shape == (3720,)
        assert hf["landslide_susceptibility_index"].shape == (1380, 3720)

        # Check disclaimer attribute
        attrs = dict(hf["landslide_susceptibility_index"].attrs)
        assert "disclaimer" in attrs
        assert "NOT A NEXT-24-HOUR FORECAST" in attrs["disclaimer"]

        lsi = hf["landslide_susceptibility_index"][:]
        mask = hf["meghalaya_mask"][:]
        valid_lsi = lsi[mask == 1]
        assert np.all((valid_lsi >= 0.0) & (valid_lsi <= 1.0))
