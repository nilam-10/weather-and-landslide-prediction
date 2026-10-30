"""Automated Test Suite for Phase 4: Landslide Susceptibility & Dynamic Hazard Research.

Verifies:
  1. Landslide inventory audit (67 total, 38 verified rainfall, 25 border, 4 non-rainfall).
  2. Environmental feature registry (10 verified morphometric features, 5 audited external).
  3. Spatial block cross-validation metrics across all 3 models.
  4. Geocoding coordinate uncertainty sensitivity trends.
  5. Empirical dynamic hazard validation outputs.
  6. Model checkpoint loading and inference functionality.
  7. Publication figures and reports integrity.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REP_DIR = PROJECT_ROOT / "outputs" / "reports"
EXP_DIR = PROJECT_ROOT / "outputs" / "experiments"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase4"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "landslide_susceptibility"


def test_landslide_inventory_audit() -> None:
    """Verify inventory audit reconciling 67 raw events to 38 verified rainfall events."""
    csv_path = REP_DIR / "PHASE4_LANDSLIDE_DATASET_INVENTORY.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 67, f"Expected 67 total rows, got {len(rows)}"

    verified = [r for r in rows if r["audit_status"] == "VERIFIED_RAINFALL_EVENT"]
    border_excluded = [r for r in rows if r["audit_status"] == "EXCLUDED_CROSS_BORDER"]
    non_rain_excluded = [r for r in rows if r["audit_status"] == "EXCLUDED_NON_RAINFALL"]

    assert len(verified) == 38, f"Expected 38 verified rainfall events, got {len(verified)}"
    assert len(border_excluded) == 25, f"Expected 25 border exclusions, got {len(border_excluded)}"
    assert len(non_rain_excluded) == 4, f"Expected 4 non-rainfall exclusions, got {len(non_rain_excluded)}"

    # Check spatial block distribution among verified
    blocks = [r["spatial_block"] for r in verified]
    assert blocks.count("West Garo Hills") == 11
    assert blocks.count("Central Khasi Hills") == 17
    assert blocks.count("East Jaintia Hills") == 10


def test_environmental_feature_registry() -> None:
    """Verify environmental feature registry covering 10 terrain + 5 external features."""
    csv_path = REP_DIR / "PHASE4_ENVIRONMENTAL_FEATURE_REGISTRY.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 15, f"Expected 15 features, got {len(rows)}"

    active_features = [r for r in rows if r["status"] == "VERIFIED_ACTIVE_PREDICTOR"]
    assert len(active_features) == 10, f"Expected 10 active terrain features, got {len(active_features)}"

    # Verify TRI and TPI are included
    feature_names = [r["feature_name"] for r in active_features]
    assert "tri" in feature_names
    assert "tpi" in feature_names
    assert "elevation_mean" in feature_names
    assert "slope_mean" in feature_names

    for r in active_features:
        assert r["missing_value_rate"] == "0.0%"


def test_model_comparison_spatial_cv() -> None:
    """Verify overall model comparison metrics under 3-fold spatial cross-validation."""
    csv_path = EXP_DIR / "PHASE4_MODEL_COMPARISON.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3, f"Expected 3 model summaries, got {len(rows)}"
    model_names = {r["model_name"] for r in rows}
    assert model_names == {"Logistic Regression", "Random Forest", "XGBoost"}

    for r in rows:
        assert int(r["total_samples"]) == 228
        assert int(r["total_presences"]) == 38
        assert int(r["total_background"]) == 190
        roc = float(r["roc_auc"])
        pr = float(r["pr_auc"])
        bal_acc = float(r["balanced_accuracy"])
        assert roc >= 0.90, f"Expected spatial ROC-AUC >= 0.90, got {roc} for {r['model_name']}"
        assert pr >= 0.80, f"Expected spatial PR-AUC >= 0.80, got {pr} for {r['model_name']}"
        assert bal_acc >= 0.80, f"Expected balanced accuracy >= 0.80, got {bal_acc}"


def test_spatial_cv_fold_balance() -> None:
    """Verify fold-by-fold results across the 3 geomorphic blocks."""
    csv_path = EXP_DIR / "PHASE4_SPATIAL_CV_RESULTS.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 9, f"Expected 9 fold records (3 models x 3 folds), got {len(rows)}"

    for r in rows:
        assert int(r["test_presences"]) in [10, 11, 17]
        assert int(r["test_background"]) > 0
        assert 0.0 <= float(r["roc_auc"]) <= 1.0


def test_coordinate_uncertainty_sensitivity() -> None:
    """Verify coordinate uncertainty sensitivity shows expected degradation at nominal 0m."""
    csv_path = EXP_DIR / "PHASE4_EVENT_UNCERTAINTY_ANALYSIS.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12, f"Expected 12 sensitivity records (4 modes x 3 models), got {len(rows)}"

    # Nominal 0m must show severe performance degradation compared to 450m buffer
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        nom_row = next(r for r in rows if r["model_name"] == m_name and r["uncertainty_mode"] == "Nominal_Point_0m")
        buf_row = next(r for r in rows if r["model_name"] == m_name and r["uncertainty_mode"] == "Buffer_450m_5x5")

        nom_auc = float(nom_row["roc_auc"])
        buf_auc = float(buf_row["roc_auc"])

        assert nom_auc <= 0.60, f"Expected nominal AUC <= 0.60, got {nom_auc} for {m_name}"
        assert buf_auc >= 0.90, f"Expected 450m buffer AUC >= 0.90, got {buf_auc} for {m_name}"
        assert buf_auc > nom_auc + 0.35, f"Expected buffer improvement > 0.35 AUC, got {buf_auc - nom_auc}"


def test_dynamic_hazard_validation() -> None:
    """Verify dynamic rainfall hazard decision matrix validation records."""
    csv_path = EXP_DIR / "PHASE4_DYNAMIC_HAZARD_VALIDATION.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8, f"Expected 8 time-slice records, got {len(rows)}"
    for r in rows:
        assert "NOT a calibrated probability" in r["scientific_interpretation"]
        tot_pct = (
            float(r["low_hazard_cells_pct"])
            + float(r["moderate_hazard_cells_pct"])
            + float(r["high_hazard_cells_pct"])
            + float(r["very_high_hazard_cells_pct"])
        )
        assert abs(tot_pct - 100.0) < 0.2


def test_model_checkpoints_and_inference() -> None:
    """Verify saved model checkpoints can be loaded and perform inference."""
    lr_ckpt = MODELS_DIR / "logistic_regression_spatial_cv.joblib"
    rf_ckpt = MODELS_DIR / "random_forest_spatial_cv.joblib"
    xgb_ckpt = MODELS_DIR / "xgboost_spatial_cv.joblib"

    assert lr_ckpt.exists() and lr_ckpt.stat().st_size > 0
    assert rf_ckpt.exists() and rf_ckpt.stat().st_size > 0
    assert xgb_ckpt.exists() and xgb_ckpt.stat().st_size > 0

    dummy_X = np.random.randn(5, 10).astype(np.float32)

    # Test Logistic Regression
    lr_data = joblib.load(lr_ckpt)
    assert "scaler" in lr_data and "model" in lr_data
    lr_scaled = lr_data["scaler"].transform(dummy_X)
    lr_probs = lr_data["model"].predict_proba(lr_scaled)[:, 1]
    assert lr_probs.shape == (5,)
    assert np.all((lr_probs >= 0.0) & (lr_probs <= 1.0))

    # Test Random Forest
    rf_data = joblib.load(rf_ckpt)
    assert "model" in rf_data
    rf_probs = rf_data["model"].predict_proba(dummy_X)[:, 1]
    assert rf_probs.shape == (5,)
    assert np.all((rf_probs >= 0.0) & (rf_probs <= 1.0))

    # Test XGBoost
    xgb_data = joblib.load(xgb_ckpt)
    assert "model" in xgb_data
    xgb_probs = xgb_data["model"].predict_proba(dummy_X)[:, 1]
    assert xgb_probs.shape == (5,)
    assert np.all((xgb_probs >= 0.0) & (xgb_probs <= 1.0))


def test_publication_figures_and_reports() -> None:
    """Verify all 6 publication figures and completion reports exist and are non-empty."""
    for fig_id in range(1, 7):
        matches = list(FIG_DIR.glob(f"fig0{fig_id}_*.png"))
        assert len(matches) == 1, f"Missing figure {fig_id} in {FIG_DIR}"
        assert matches[0].stat().st_size > 5000, f"Figure {matches[0]} suspiciously small"

    feasibility_md = REP_DIR / "PHASE4_NEXT24H_FEASIBILITY.md"
    completion_md = REP_DIR / "PHASE4_COMPLETION.md"

    assert feasibility_md.exists() and feasibility_md.stat().st_size > 500
    assert completion_md.exists() and completion_md.stat().st_size > 500
