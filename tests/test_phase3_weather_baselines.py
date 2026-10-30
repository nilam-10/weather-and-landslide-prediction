"""Automated Tests for Phase 3 Research Weather Baselines and Models."""
from __future__ import annotations

import csv
from pathlib import Path
import joblib
import numpy as np
import pytest
import torch

from src.models.convlstm import MultiTaskWeatherConvLSTM
from scripts.run_phase3_research_experiments import (
    MultiTaskWeatherFCLSTM,
    MultiTaskWeatherMLP,
    compute_dense_horn_schunck_flow,
    compute_sparse_lucas_kanade_flow,
    semi_lagrangian_advect,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REP_DIR = PROJECT_ROOT / "outputs" / "reports"
EXP_DIR = PROJECT_ROOT / "outputs" / "experiments"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "phase3"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase3"


def test_phase3_output_reports_exist():
    """Verify that all Phase 3 required reports and manifest CSVs exist."""
    required_files = [
        REP_DIR / "PHASE3_FINAL_SPLIT_MANIFEST.csv",
        REP_DIR / "PHASE3_MODEL_REGISTRY.csv",
        REP_DIR / "PHASE3_PAPER_BASELINE_MAPPING.csv",
        REP_DIR / "PHASE3_COMPLETION.md",
        EXP_DIR / "PHASE3_ALL_MODEL_METRICS.csv",
        EXP_DIR / "PHASE3_CLASSIFICATION_RESULTS.csv",
        EXP_DIR / "PHASE3_CONTINUOUS_QPF_RESULTS.csv",
        EXP_DIR / "PHASE3_TEST_PREDICTIONS" / "PHASE3_TEST_PREDICTIONS_2023.npz",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required Phase 3 deliverable: {rf}"
        assert rf.stat().st_size > 0, f"File {rf} is empty"


def test_phase3_model_checkpoints_exist():
    """Verify that trained model checkpoints exist for all neural architectures."""
    required_ckpts = [
        MODELS_DIR / "convlstm_seed42.pt",
        MODELS_DIR / "convlstm_seed123.pt",
        MODELS_DIR / "convlstm_seed999.pt",
        MODELS_DIR / "fclstm_seed42.pt",
        MODELS_DIR / "fclstm_seed123.pt",
        MODELS_DIR / "fclstm_seed999.pt",
        MODELS_DIR / "mlp_seed42.pt",
        MODELS_DIR / "mlp_seed123.pt",
        MODELS_DIR / "mlp_seed999.pt",
        MODELS_DIR / "logistic_regression.joblib",
    ]
    for ck in required_ckpts:
        assert ck.exists(), f"Missing checkpoint: {ck}"
        assert ck.stat().st_size > 1000, f"Checkpoint {ck} unexpectedly small"


def test_phase3_optical_flow_kinematics():
    """Verify that Sparse and Dense optical flow algorithms compute valid non-diverging fields."""
    i1 = np.zeros((25, 40), dtype=np.float32)
    i2 = np.zeros((25, 40), dtype=np.float32)
    i1[10:15, 15:20] = 10.0
    i2[10:15, 17:22] = 10.0  # +2 pixels eastward advection

    u_sparse, v_sparse = compute_sparse_lucas_kanade_flow(i1, i2)
    assert u_sparse.shape == (25, 40)
    assert v_sparse.shape == (25, 40)
    assert np.all(np.isfinite(u_sparse))

    u_dense, v_dense = compute_dense_horn_schunck_flow(i1, i2, alpha=1.0, num_iter=20)
    assert u_dense.shape == (25, 40)
    assert v_dense.shape == (25, 40)
    assert np.all(np.isfinite(u_dense))

    warped = semi_lagrangian_advect(i2, u_dense, v_dense, steps=1)
    assert warped.shape == (25, 40)
    assert np.min(warped) >= 0.0


def test_phase3_neural_forward_shapes():
    """Verify that neural baseline forward passes output exact multi-horizon shapes."""
    B, T, C, H, W = 2, 4, 3, 25, 40
    x_seq = torch.randn(B, T, C, H, W)
    feat = torch.randn(B, 16, H, W)

    convlstm = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=5)
    c_cat, c_qpf = convlstm(x_seq)
    assert c_cat.shape == (B, 5, H, W)
    assert c_qpf.shape == (B, 5, H, W)

    fclstm = MultiTaskWeatherFCLSTM()
    fc_cat, fc_qpf = fclstm(x_seq)
    assert fc_cat.shape == (B, 5, H, W)
    assert fc_qpf.shape == (B, 5, H, W)

    mlp = MultiTaskWeatherMLP()
    m_cat, m_qpf = mlp(feat)
    assert m_cat.shape == (B, 5, H, W)
    assert m_qpf.shape == (B, 5, H, W)


def test_phase3_stored_predictions_npz():
    """Verify that the stored predictions NPZ contains all evaluated models."""
    npz_path = EXP_DIR / "PHASE3_TEST_PREDICTIONS" / "PHASE3_TEST_PREDICTIONS_2023.npz"
    with np.load(npz_path) as data:
        keys = set(data.files)

    expected_keys = {
        "ground_truth_binary",
        "ground_truth_continuous",
        "reference_times",
        "persistence_binary",
        "persistence_continuous",
        "sparse_lk_binary",
        "sparse_lk_continuous",
        "dense_flow_binary",
        "dense_flow_continuous",
        "logistic_regression_probs",
        "logistic_regression_binary",
        "mlp_probs",
        "mlp_qpf",
        "mlp_bin",
        "fclstm_probs",
        "fclstm_qpf",
        "fclstm_bin",
        "convlstm_probs",
        "convlstm_qpf",
        "convlstm_bin",
    }
    assert expected_keys.issubset(keys), f"Missing keys in test predictions: {expected_keys - keys}"


def test_phase3_publication_figures_exist():
    """Verify that all 300 DPI publication figures are generated."""
    expected_figs = [
        "fig01_training_validation_loss_curves.png",
        "fig02_validation_csi_epochs.png",
        "fig03_csi_vs_lead_horizon.png",
        "fig04_f1_vs_lead_horizon.png",
        "fig05_roc_curves_30m_60m.png",
        "fig06_calibration_reliability_diagrams.png",
        "fig06_precision_recall_curves_30m_60m.png",
        "fig07_confusion_matrices_30m.png",
        "fig08_continuous_qpf_error_comparison.png",
        "fig09_spatial_nowcasting_comparison_maps.png",
        "fig10_convlstm_dataset_scaling_comparison.png",
    ]
    for fn in expected_figs:
        fig_path = FIG_DIR / fn
        assert fig_path.exists(), f"Figure {fn} missing"
        assert fig_path.stat().st_size > 10000, f"Figure {fn} unexpectedly small"
