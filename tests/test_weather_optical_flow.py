"""Automated tests for Optical Flow Weather Nowcasting Baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.abspath("scripts"))
sys.path.insert(0, os.path.abspath("."))

from weather_optical_flow import (
    calculate_contingency_metrics,
    compute_lucas_kanade_flow,
    semi_lagrangian_extrapolate,
)

METRICS_PATH = Path("outputs/models/weather_optical_flow/optical_flow_metrics.json")


def test_lucas_kanade_flow_computation():
    """Verify that optical flow produces finite velocity vectors with expected dimensions."""
    h, w = 25, 40
    # Synthetic translating Gaussian storm cell
    y, x = np.mgrid[0:h, 0:w]
    i1 = 20.0 * np.exp(-((x - 15)**2 + (y - 12)**2) / 10.0)
    i2 = 20.0 * np.exp(-((x - 17)**2 + (y - 12)**2) / 10.0)  # translated +2 pixels in x

    u, v = compute_lucas_kanade_flow(i1, i2, sigma=0.5, win_size=5, reg=1e-3, max_vel=5.0)

    assert u.shape == (h, w)
    assert v.shape == (h, w)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))

    # Mean u inside the storm core should be positive (moving East)
    core = i2 > 5.0
    assert np.mean(u[core]) > 0.0, f"Expected positive eastward motion, got {np.mean(u[core])}"
    assert abs(np.mean(v[core])) < 1.0, f"Expected near-zero vertical motion, got {np.mean(v[core])}"


def test_semi_lagrangian_extrapolation():
    """Verify forward advection preserves shape and respects non-negativity."""
    h, w = 25, 40
    y, x = np.mgrid[0:h, 0:w]
    img = 15.0 * np.exp(-((x - 20)**2 + (y - 12)**2) / 8.0)
    u = np.ones((h, w), dtype=np.float32) * 1.5  # 1.5 pixels per step eastward
    v = np.zeros((h, w), dtype=np.float32)

    adv_1 = semi_lagrangian_extrapolate(img, u, v, lead_steps=1)
    adv_2 = semi_lagrangian_extrapolate(img, u, v, lead_steps=2)

    assert adv_1.shape == (h, w)
    assert adv_2.shape == (h, w)
    assert np.all(adv_1 >= 0.0), "Extrapolated precipitation must be non-negative"
    assert np.all(adv_2 >= 0.0)

    # Peak of adv_2 should be further right than adv_1
    peak_orig = np.argmax(np.sum(img, axis=0))
    peak_adv1 = np.argmax(np.sum(adv_1, axis=0))
    peak_adv2 = np.argmax(np.sum(adv_2, axis=0))

    assert peak_adv1 >= peak_orig
    assert peak_adv2 >= peak_adv1


def test_contingency_metrics_calculation():
    """Verify contingency calculation against known analytical truth."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    # TP = 3 (idx 0, 1, 6)
    # FP = 1 (idx 3)
    # FN = 2 (idx 2, 8)
    # TN = 4 (idx 4, 5, 7, 9)
    # Total = 10, Positives = 5

    res = calculate_contingency_metrics(y_true, y_pred)

    assert res["confusion_matrix"]["tp"] == 3
    assert res["confusion_matrix"]["fp"] == 1
    assert res["confusion_matrix"]["fn"] == 2
    assert res["confusion_matrix"]["tn"] == 4

    # CSI = 3 / (3 + 1 + 2) = 3/6 = 0.5
    assert res["csi"] == 0.5
    # POD = 3 / (3 + 2) = 0.6
    assert res["pod_recall"] == 0.6
    # FAR = 1 / (3 + 1) = 0.25
    assert res["far"] == 0.25
    # Accuracy = (3 + 4) / 10 = 0.7
    assert res["accuracy"] == 0.7


def test_optical_flow_metrics_artifact_integrity():
    """Verify optical flow metrics artifact exists and has all 4 lead horizons."""
    assert METRICS_PATH.exists(), f"Missing optical flow metrics at {METRICS_PATH}"

    with METRICS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "optical_flow" in data
    assert "persistence" in data
    assert "test" in data["optical_flow"]

    test_of = data["optical_flow"]["test"]
    for lead in ["lead_30min", "lead_60min", "lead_120min", "lead_180min"]:
        assert lead in test_of
        assert 0.0 <= test_of[lead]["csi"] <= 1.0
        assert 0.0 <= test_of[lead]["pod_recall"] <= 1.0
        assert 0.0 <= test_of[lead]["far"] <= 1.0
        assert test_of[lead]["total_pixels"] > 0

    # Verify optical flow outperforms persistence at +60m
    of_csi_60 = test_of["lead_60min"]["csi"]
    p_csi_60 = data["persistence"]["test"]["lead_60min"]["csi"]
    assert of_csi_60 > p_csi_60, f"Expected OF ({of_csi_60}) > Persistence ({p_csi_60})"
