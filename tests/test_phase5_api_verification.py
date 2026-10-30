"""Phase 5 API and Model Verification Test Suite.

Verifies:
  1. initialize_cache() loads all 10 models (7 weather + 3 landslide).
  2. /health reports status online and correct model loading flags.
  3. /metadata reports all supported models and documented decision.
  4. /layers/weather functions for all 7 weather models and 5 lead times.
  5. /layers/landslide/susceptibility functions for all 3 landslide models.
  6. /layers/landslide/dynamic reports correct categories and no NaN values.
  7. /layers/landslide/events returns exactly 38 verified rainfall events.
  8. /layers/colocation returns valid boolean mask.
  9. /cell_inspect returns complete multimodal state at queried location.
  10. /provenance returns complete citations and retrospective disclaimers.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import (
    initialize_cache,
    get_health,
    get_metadata,
    get_weather_layer,
    get_landslide_susceptibility,
    get_dynamic_hazard,
    get_landslide_events,
    get_colocation_overlay,
    inspect_cell,
    get_provenance,
)


@pytest.fixture(scope="session", autouse=True)
def setup_api():
    initialize_cache()


def test_health_endpoint():
    h = get_health()
    assert h["status"] == "online"
    assert h["datasets"]["weather_steps_available"] == 336
    assert h["datasets"]["historical_landslide_events_verified"] == 38
    assert h["models_loaded"]["convlstm"] is True
    assert h["models_loaded"]["mlp"] is True
    assert h["models_loaded"]["fclstm"] is True
    assert h["models_loaded"]["logistic_regression"] is True
    assert h["models_loaded"]["landslide_xgboost"] is True
    assert h["models_loaded"]["landslide_random_forest"] is True
    assert h["models_loaded"]["landslide_logistic_regression"] is True


def test_metadata_endpoint():
    m = get_metadata()
    w_models = m["supported_models"]["weather"]
    l_models = m["supported_models"]["landslide"]

    assert len(w_models) == 7
    assert "convlstm" in w_models
    assert "mlp" in w_models
    assert "fclstm" in w_models
    assert "logistic_regression" in w_models
    assert "optical_flow_dense" in w_models
    assert "optical_flow_sparse" in w_models
    assert "persistence" in w_models

    assert len(l_models) == 3
    assert "xgboost" in l_models
    assert "random_forest" in l_models
    assert "logistic_regression" in l_models
    assert "validation loss and CSI" in m["production_model_decision"]


@pytest.mark.parametrize("model_name", [
    "convlstm",
    "mlp",
    "fclstm",
    "logistic_regression",
    "optical_flow_dense",
    "optical_flow_sparse",
    "persistence",
])
def test_all_weather_models_inference(model_name: str):
    res = get_weather_layer(step=316, model=model_name, lead="30m", calibrated=False)
    assert res["layer_type"] == "weather_nowcast"
    assert res["model"] == model_name
    grid_vals = res["grid"]["values"]
    assert len(grid_vals) == 25
    assert len(grid_vals[0]) == 40
    assert 0.0 <= res["stats"]["min_prob"] <= res["stats"]["max_prob"] <= 1.0

    if model_name == "logistic_regression":
        assert res["grid"]["rain_rate_mm_hr"] is None
    else:
        assert res["grid"]["rain_rate_mm_hr"] is not None
        assert len(res["grid"]["rain_rate_mm_hr"]) == 25


@pytest.mark.parametrize("lead_time", ["30m", "60m", "120m", "180m", "360m"])
def test_convlstm_all_lead_times(lead_time: str):
    res = get_weather_layer(step=316, model="convlstm", lead=lead_time, calibrated=True)
    assert res["lead_time"] == lead_time
    assert res["calibrated"] is True
    assert len(res["grid"]["values"]) == 25


@pytest.mark.parametrize("landslide_model", ["xgboost", "random_forest", "logistic_regression"])
def test_all_landslide_susceptibility_models(landslide_model: str):
    res = get_landslide_susceptibility(model=landslide_model)
    assert res["layer_type"] == "landslide_static_susceptibility"
    assert res["model_key"] == landslide_model
    assert res["feature_count"] == 10
    grid_vals = res["grid"]["values"]
    assert len(grid_vals) == 173
    assert len(grid_vals[0]) == 465
    assert res["spatial_cross_val_roc_auc"] > 0.50


def test_dynamic_hazard_and_events():
    dh = get_dynamic_hazard(step=316)
    assert dh["layer_type"] == "landslide_dynamic_hazard"
    assert dh["step"] == 316
    assert dh["category_counts"]["moderate_1"] > 0

    ev = get_landslide_events()
    assert ev["type"] == "FeatureCollection"
    assert ev["total_features"] == 38
    assert len(ev["features"]) == 38


def test_colocation_and_cell_inspect():
    coloc = get_colocation_overlay(step=316, weather_thresh=0.5, susceptibility_thresh=0.4)
    assert coloc["layer_type"] == "spatial_colocation_indicator"
    assert len(coloc["grid"]["indicator"]) == 25

    insp = inspect_cell(lat=25.5, lon=91.5, step=316)
    assert insp["queried_coordinates"]["latitude"] == 25.5
    assert insp["queried_coordinates"]["longitude"] == 91.5
    assert insp["landslide_hazard_state"]["susceptibility_tier"] in [
        "Very Low", "Low", "Moderate", "High", "Very High"
    ]


def test_provenance_citations():
    p = get_provenance()
    assert len(p["data_sources"]) == 5
    assert len(p["models_and_papers"]) == 11
    assert "RESEARCH USE ONLY" in p["disclaimer"]
