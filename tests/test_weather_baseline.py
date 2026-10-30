"""Unit and integration tests for Step 6 Weather Baseline ML pipeline and GPU verification."""
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest
import torch


BASELINE_DIR = Path("outputs/models/weather_baseline")


def test_gpu_hardware_detection():
    """Verify that NVIDIA GPU and CUDA are detected and accessible."""
    assert torch.cuda.is_available() is True, "CUDA should be available on this host."
    device_name = torch.cuda.get_device_name(0)
    assert "RTX 4050" in device_name, f"Expected RTX 4050, got {device_name}"
    total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    assert total_mem_mb > 5000, f"Expected >5GB VRAM, got {total_mem_mb} MB"


def test_baseline_artifacts_exist():
    """Verify that all required baseline artifacts are generated and non-empty."""
    expected_files = [
        "model_config.json",
        "baseline_metrics.json",
        "imputer.joblib",
        "scaler.joblib",
        "logistic_regression_model.joblib",
        "pytorch_baseline_model.pt",
        "weather_baseline_evaluation.png",
    ]
    for fname in expected_files:
        fpath = BASELINE_DIR / fname
        assert fpath.exists(), f"Artifact {fpath} does not exist."
        assert fpath.stat().st_size > 0, f"Artifact {fpath} is empty."


def test_baseline_metrics_integrity():
    """Verify metrics dictionary structure and values for persistence, logistic regression, and PyTorch."""
    metrics_path = BASELINE_DIR / "baseline_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    for model_key in ["persistence", "logistic_regression", "pytorch_mlp"]:
        assert model_key in data, f"Model {model_key} missing in metrics."
        m = data[model_key]

        for split in ["train", "val", "test"]:
            assert split in m, f"Split {split} missing for {model_key}"
            s = m[split]
            assert 0.0 <= s["precision"] <= 1.0
            assert 0.0 <= s["recall_pod"] <= 1.0
            assert 0.0 <= s["f1_score"] <= 1.0
            assert 0.0 <= s["csi_threat_score"] <= 1.0
            assert 0.0 <= s["false_alarm_ratio"] <= 1.0

            cm = s["confusion_matrix"]
            total_cm = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
            assert total_cm == s["total_samples"], f"CM sum {total_cm} != total {s['total_samples']}"

    # Verify PyTorch training ran on CUDA
    pt_info = data["pytorch_mlp"]["training_info"]
    assert pt_info["device"] == "cuda"
    assert pt_info["epochs"] == 8
    assert pt_info["total_training_time_seconds"] > 0.0


def test_baseline_config_and_anti_leakage():
    """Verify that configuration records strict chronological splits, anti-leakage audit, and proxy caveat."""
    config_path = BASELINE_DIR / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["heavy_rain_threshold_mm_hr"] == 5.0
    assert config["training_device"] == "cuda"
    assert config["anti_leakage_audit"]["strict_chronological_splits"] is True
    assert config["anti_leakage_audit"]["preprocessing_fit_on_train_only"] is True
    assert config["anti_leakage_audit"]["zero_future_rainfall_in_features"] is True

    # Splits verification
    splits = config["chronological_splits"]
    assert splits["train"]["dates"] == ["20250701", "20250702", "20250703", "20250704"]
    assert splits["validation"]["dates"] == ["20250705"]
    assert splits["test"]["dates"] == ["20250706", "20250707"]

    # Proxy caveat and disclaimer presence
    assert "proxy label" in config["proxy_label_caveat"]
    assert "Preliminary pipeline test only" in config["disclaimer"]
