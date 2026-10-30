"""Automated tests for Experiment Registry CSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

EXP_DIR = Path("outputs/experiments")


def test_training_history_csv():
    """Verify training_history.csv exists, has 15 epochs, and decreasing train loss."""
    path = EXP_DIR / "training_history.csv"
    assert path.exists(), f"Missing {path}"

    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 15
    epochs = [int(r["epoch"]) for r in reader]
    assert epochs == list(range(1, 16))

    train_losses = [float(r["train_loss"]) for r in reader]
    # General downward trend: final loss < initial loss
    assert train_losses[-1] < train_losses[0]

    for r in reader:
        assert float(r["val_csi_30min"]) > 0.3
        assert float(r["val_csi_60min"]) > 0.2


def test_model_comparison_csv():
    """Verify model_comparison.csv contains all benchmark models and valid metrics."""
    path = EXP_DIR / "model_comparison.csv"
    assert path.exists(), f"Missing {path}"

    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 10
    models = set(r["model_name"] for r in reader)
    assert "Persistence Baseline" in models
    assert "Optical Flow (Lucas-Kanade)" in models
    assert "Logistic Regression" in models
    assert "PyTorch 3-Layer MLP" in models
    assert "Spatiotemporal 2D ConvLSTM" in models

    for r in reader:
        assert float(r["csi"]) >= 0.0
        assert float(r["pod_recall"]) >= 0.0
        assert float(r["far"]) >= 0.0
        assert float(r["accuracy"]) >= 0.0


def test_test_predictions_csv():
    """Verify test_predictions.csv exists and has authentic sample forecast columns."""
    path = EXP_DIR / "test_predictions.csv"
    assert path.exists(), f"Missing {path}"

    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 50
    for r in reader:
        assert float(r["latitude"]) > 24.5
        assert float(r["longitude"]) > 89.5
        assert float(r["current_rain_mm_hr"]) >= 0.0
        assert float(r["ground_truth_lead30_mm_hr"]) >= 0.0
        assert 0.0 <= float(r["convlstm_prob_lead30"]) <= 1.0


def test_experiment_registry_csv():
    """Verify experiment_registry.csv records all 8 model runs and artifacts."""
    path = EXP_DIR / "experiment_registry.csv"
    assert path.exists(), f"Missing {path}"

    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 8
    run_ids = [r["run_id"] for r in reader]
    assert "EXP-WTH-001" in run_ids
    assert "EXP-WTH-005" in run_ids
    assert "EXP-LND-002" in run_ids

    for r in reader:
        assert r["verification_status"] == "VERIFIED_TEST_PASSED"
        assert len(r["target_horizons"]) > 0


def test_landslide_registries_csv():
    """Verify landslide spatial CV, feature importance, and sensitivity CSVs."""
    cv_path = EXP_DIR / "landslide_spatial_cv_results.csv"
    feat_path = EXP_DIR / "landslide_feature_importance.csv"
    sens_path = EXP_DIR / "landslide_sensitivity_analysis.csv"

    assert cv_path.exists()
    assert feat_path.exists()
    assert sens_path.exists()

    with cv_path.open("r", encoding="utf-8") as f:
        cv_reader = list(csv.DictReader(f))
    assert len(cv_reader) == 6  # 3 folds x 2 models
    for r in cv_reader:
        assert float(r["roc_auc"]) > 0.90
        assert float(r["pr_auc"]) > 0.75

    with feat_path.open("r", encoding="utf-8") as f:
        feat_reader = list(csv.DictReader(f))
    assert len(feat_reader) == 8
    ranks = [int(r["rank"]) for r in feat_reader]
    assert ranks == list(range(1, 9))

    with sens_path.open("r", encoding="utf-8") as f:
        sens_reader = list(csv.DictReader(f))
    assert len(sens_reader) == 5
