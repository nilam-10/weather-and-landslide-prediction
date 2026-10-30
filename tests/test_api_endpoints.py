"""Automated tests for FastAPI Full-Stack Web GIS Endpoints."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify /health returns 200, system online, hardware details, and scientific disclaimer."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert "hardware" in data
    assert "cuda_available" in data["hardware"]
    assert data["datasets"]["weather_steps_available"] == 336
    assert data["datasets"]["historical_landslide_events_verified"] == 38
    assert "RESEARCH USE ONLY" in data["disclaimer"]


def test_api_metadata_endpoint():
    """Verify /metadata returns 200, bounding box, 336 steps, and model registry."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()

    assert data["bounding_box_wsen"] == [89.5, 24.5, 93.5, 27.0]
    assert data["grids"]["weather_grid"]["rows"] == 25
    assert data["grids"]["weather_grid"]["cols"] == 40
    assert data["temporal_timeline"]["total_slots"] == 336
    assert data["temporal_timeline"]["peak_hazard_step"] == 316
    assert "convlstm" in data["supported_models"]["weather"]
    assert "30m" in data["supported_lead_times"]
    assert "360m" in data["supported_lead_times"]
    assert data["probability_calibration_supported"] is True


def test_api_layers_weather_endpoint():
    """Verify /layers/weather returns 200, valid grid, and non-empty probabilities."""
    # Test ConvLSTM +30m uncalibrated
    response = client.get("/layers/weather?step=316&model=convlstm&lead=30m&calibrated=false")
    assert response.status_code == 200
    data = response.json()

    assert data["layer_type"] == "weather_nowcast"
    assert data["model"] == "convlstm"
    assert data["lead_time"] == "30m"
    assert data["calibrated"] is False
    assert len(data["grid"]["latitudes"]) == 25
    assert len(data["grid"]["longitudes"]) == 40
    assert len(data["grid"]["values"]) == 25
    assert len(data["grid"]["values"][0]) == 40

    # Values must be bounded in [0, 1]
    vals = [v for row in data["grid"]["values"] for v in row]
    assert min(vals) >= 0.0
    assert max(vals) <= 1.0

    # Test ConvLSTM +360m calibrated
    res_360 = client.get("/layers/weather?step=316&model=convlstm&lead=360m&calibrated=true")
    assert res_360.status_code == 200
    d_360 = res_360.json()
    assert d_360["calibrated"] is True
    assert "physical_limit_notice" in d_360
    assert len(d_360["grid"]["values"]) == 25


def test_api_layers_landslide_susceptibility_endpoint():
    """Verify /layers/landslide/susceptibility returns 200 and valid downsampled LSI grid."""
    response = client.get("/layers/landslide/susceptibility")
    assert response.status_code == 200
    data = response.json()

    assert data["layer_type"] == "landslide_static_susceptibility"
    assert data["model"].startswith("XGBoost")
    assert len(data["grid"]["latitudes"]) > 100
    assert len(data["grid"]["longitudes"]) > 200

    # Check that valid LSI values are in [0, 1]
    non_null_vals = [v for row in data["grid"]["values"] for v in row if v is not None]
    assert len(non_null_vals) > 0
    assert min(non_null_vals) >= 0.0
    assert max(non_null_vals) <= 1.0


def test_api_layers_landslide_dynamic_endpoint():
    """Verify /layers/landslide/dynamic returns 200 and LHASA 2.0 categorical levels 0..3."""
    response = client.get("/layers/landslide/dynamic?step=316")
    assert response.status_code == 200
    data = response.json()

    assert data["layer_type"] == "landslide_dynamic_hazard"
    assert "category_counts" in data
    assert len(data["grid"]["hazard_levels"]) == 25
    assert len(data["grid"]["hazard_levels"][0]) == 40

    all_levels = set([v for row in data["grid"]["hazard_levels"] for v in row])
    assert all_levels.issubset({0, 1, 2, 3})


def test_api_layers_landslide_events_endpoint():
    """Verify /layers/landslide/events returns valid GeoJSON with 38 verified events."""
    response = client.get("/layers/landslide/events")
    assert response.status_code == 200
    data = response.json()

    assert data["type"] == "FeatureCollection"
    assert data["total_features"] == 38
    assert len(data["features"]) == 38

    # Inspect first feature properties
    first_feat = data["features"][0]
    assert "geometry" in first_feat
    assert "coordinates" in first_feat["geometry"]
    assert len(first_feat["geometry"]["coordinates"]) == 2
    props = first_feat["properties"]
    assert "date" in props
    assert "trigger" in props
    assert "accuracy" in props


def test_api_layers_colocation_endpoint():
    """Verify /layers/colocation returns 200 and includes prominent anti-fallacy notice."""
    response = client.get("/layers/colocation?step=316&weather_thresh=0.5&susceptibility_thresh=0.4")
    assert response.status_code == 200
    data = response.json()

    assert data["layer_type"] == "spatial_colocation_indicator"
    assert "INDEPENDENT CO-LOCATION ONLY — NOT A JOINT PROBABILITY" in data["critical_scientific_notice"]
    assert len(data["grid"]["indicator"]) == 25
    assert len(data["grid"]["indicator"][0]) == 40


def test_api_cell_inspect_endpoint():
    """Verify /cell_inspect returns multimodal status at queried coordinates."""
    response = client.get("/cell_inspect?lat=25.55&lon=91.88&step=316")
    assert response.status_code == 200
    data = response.json()

    assert "meteorological_state" in data
    assert "landslide_hazard_state" in data
    assert data["matched_weather_cell"]["center_lat"] > 24.5
    assert data["meteorological_state"]["instantaneous_precip_mm_hr"] >= 0.0
    assert 0.0 <= data["landslide_hazard_state"]["static_susceptibility_index"] <= 1.0
    assert data["landslide_hazard_state"]["lhasa_dynamic_hazard_level"] in [0, 1, 2, 3]


def test_api_provenance_endpoint():
    """Verify /provenance returns 200, citations, and product DOIs."""
    response = client.get("/provenance")
    assert response.status_code == 200
    data = response.json()

    assert len(data["data_sources"]) >= 4
    products = [ds["product"] for ds in data["data_sources"]]
    assert any("MERGIR" in p for p in products)
    assert any("IMERG" in p for p in products)
    assert any("SRTM" in p for p in products)
    assert any("GLC" in p for p in products)
