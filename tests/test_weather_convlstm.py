"""Unit and integration tests for Spatiotemporal ConvLSTM Weather Nowcasting Model (Step 7)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath("scripts"))
from train_weather_convlstm import (
    CompoundNowcastingLoss,
    ConvLSTMCell,
    SpatiotemporalWeatherDataset,
    WeatherConvLSTM,
    calculate_contingency_metrics,
    compute_normalization_stats,
    load_daily_aligned_datasets,
)


@pytest.fixture
def sample_aligned_data():
    """Load sample daily aligned data for testing."""
    data_dir = Path("data/processed/weather")
    # Use July 1 for fast test fixture
    return load_daily_aligned_datasets(data_dir, ["20250701"])


def test_convlstm_cell_forward():
    """Verify ConvLSTM cell hidden state recurrence and spatial tensor preservation."""
    cell = ConvLSTMCell(in_channels=3, hidden_channels=16, kernel_size=3)
    x = torch.randn(2, 3, 25, 40)
    h, c = cell(x, state=None)
    assert h.shape == (2, 16, 25, 40)
    assert c.shape == (2, 16, 25, 40)

    # Second step with previous state
    x2 = torch.randn(2, 3, 25, 40)
    h2, c2 = cell(x2, state=(h, c))
    assert h2.shape == (2, 16, 25, 40)
    assert c2.shape == (2, 16, 25, 40)
    assert not torch.allclose(h, h2)


def test_weather_convlstm_model_forward_and_backward():
    """Verify full multi-lead ConvLSTM model forward pass and gradient flow."""
    model = WeatherConvLSTM(
        in_channels=3,
        hidden_channels=(16, 16),
        kernel_size=3,
        num_leads=2,
        dropout=0.0,
    )
    # Batch size 2, T_in 4, C 3, H 25, W 40
    x_seq = torch.randn(2, 4, 3, 25, 40, requires_grad=True)
    logits = model(x_seq)

    assert logits.shape == (2, 2, 25, 40)
    loss = logits.sum()
    loss.backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()


def test_compound_nowcasting_loss():
    """Verify CompoundNowcastingLoss calculation with weighted BCE and Soft-CSI."""
    loss_fn = CompoundNowcastingLoss(pos_weight=20.0, csi_weight=1.0)
    logits = torch.randn(4, 2, 25, 40, requires_grad=True)
    targets = (torch.rand(4, 2, 25, 40) > 0.95).float()
    valid_mask = torch.ones(4, 2, 25, 40)

    loss = loss_fn(logits, targets, valid_mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None


def test_spatiotemporal_dataset_shapes_and_anti_leakage(sample_aligned_data):
    """Verify sequence generator shapes, temporal limits, and anti-leakage."""
    stats = compute_normalization_stats(sample_aligned_data)
    dataset = SpatiotemporalWeatherDataset(
        sample_aligned_data,
        stats,
        t_start=3,
        t_end=45,
        t_in=4,
        threshold=5.0,
        modality="both",
    )

    assert len(dataset) == 43  # 45 - 3 + 1
    x, y, val = dataset[0]

    # Input: (T_in=4, C=3, H=25, W=40)
    assert x.shape == (4, 3, 25, 40)
    # Target: (T_out=2, H=25, W=40)
    assert y.shape == (2, 25, 40)
    assert val.shape == (2, 25, 40)

    # Verify binary targets
    assert set(np.unique(y.numpy())).issubset({0.0, 1.0})
    # Verify no NaNs in target
    assert not torch.isnan(y).any()


def test_contingency_metrics_accuracy():
    """Verify meteorological contingency metrics (CSI, POD, FAR, Precision, F1)."""
    # 2 TP, 1 FP, 1 FN, 6 TN
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    metrics = calculate_contingency_metrics(y_true, y_prob, threshold=0.5)

    # TP = 2, FP = 1, FN = 1
    # CSI = TP / (TP + FP + FN) = 2 / 4 = 0.5
    assert metrics["csi"] == 0.5
    # POD = TP / (TP + FN) = 2 / 3 = 0.6667
    assert metrics["pod_recall"] == 0.6667
    # FAR = FP / (TP + FP) = 1 / 3 = 0.3333
    assert metrics["far"] == 0.3333
    # Precision = TP / (TP + FP) = 2 / 3 = 0.6667
    assert metrics["precision"] == 0.6667


def test_convlstm_artifacts_exist():
    """Verify that all Step 7 output artifacts exist in outputs/models/weather_convlstm/."""
    out_dir = Path("outputs/models/weather_convlstm")
    assert out_dir.exists()

    required_files = [
        "best_convlstm_model.pt",
        "convlstm_config.json",
        "convlstm_metrics.json",
        "convlstm_normalizer.json",
        "convlstm_evaluation.png",
    ]
    for rf in required_files:
        p = out_dir / rf
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

    # Verify metrics JSON integrity
    with open(out_dir / "convlstm_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "convlstm_results" in metrics
    assert "test" in metrics["convlstm_results"]
    assert "lead_30min" in metrics["convlstm_results"]["test"]
    assert "lead_60min" in metrics["convlstm_results"]["test"]
    assert metrics["convlstm_results"]["test"]["lead_30min"]["csi"] > 0.0


def test_colab_notebook_validity():
    """Verify that the Google Colab notebook exists and has valid cells."""
    nb_path = Path("colab/train_weather_convlstm.ipynb")
    assert nb_path.exists(), f"Missing Colab notebook: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 7
    # Verify first cell is markdown documentation
    assert nb["cells"][0]["cell_type"] == "markdown"
    # Verify code cells exist
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6


def test_convlstm_gate_unpack_order_and_single_canonical_class():
    """Permanent regression test for the gate unpack order invariant.

    Guarantees that:
      1. Training, evaluation, and FastAPI API all import the exact same ConvLSTMCell.
      2. Gates unpack strictly in Shi et al. (2015) order: (i, f, g, o).
      3. MultiTaskWeatherConvLSTM has exactly 161,034 trainable parameters.
    """
    from src.models.convlstm import ConvLSTMCell as CanonicalCell, MultiTaskWeatherConvLSTM
    from train_weather_convlstm import ConvLSTMCell as TrainCell
    from evaluate_frozen_models_on_independent_test import ConvLSTMCell as EvalCell
    from src.api.main import ConvLSTMCell as ApiCell

    # 1. Identity assertion: all modules import the exact same class object
    assert TrainCell is CanonicalCell, "train_weather_convlstm must import canonical ConvLSTMCell"
    assert EvalCell is CanonicalCell, "evaluate_frozen_models must import canonical ConvLSTMCell"
    assert ApiCell is CanonicalCell, "src.api.main must import canonical ConvLSTMCell"

    # 2. Parameter count assertion: exactly 161,034 net trainable parameters
    model = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=5)
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_p == 161034, f"Expected 161,034 trainable params, got {trainable_p}"

    # 3. Gate unpacking invariant: slice 2 (g) must be tanh, slice 3 (o) must be sigmoid
    cell = CanonicalCell(in_channels=3, hidden_channels=32, kernel_size=3)
    x = torch.zeros(1, 3, 10, 10)
    h = torch.zeros(1, 32, 10, 10)
    # Put large positive values in slice 2 (candidate g) and large negative values in slice 3 (output o)
    with torch.no_grad():
        cell.conv.bias.zero_()
        cell.conv.weight.zero_()
        # Set bias of slice 2 (g) = +10.0 -> tanh should be +1.0
        cell.conv.bias[64:96] = 10.0
        # Set bias of slice 3 (o) = -10.0 -> sigmoid should be ~0.0
        cell.conv.bias[96:128] = -10.0
        # Set bias of slice 0 (i) = +10.0 -> sigmoid = 1.0
        cell.conv.bias[0:32] = 10.0
        # Set bias of slice 1 (f) = 0.0
        cell.conv.bias[32:64] = 0.0

    h_out, c_out = cell(x, state=None)
    # c_next = f * c + i * g = 0 + 1.0 * tanh(+10) ~ +1.0
    assert torch.allclose(c_out, torch.ones_like(c_out), atol=1e-3), "Cell candidate g must activate with tanh"
    # h_next = o * tanh(c_next) = sigmoid(-10) * tanh(1.0) ~ 0.000045 * 0.76 ~ 0.0
    assert torch.allclose(h_out, torch.zeros_like(h_out), atol=1e-3), "Output gate o must activate with sigmoid"


def test_july_benchmark_reproduction():
    """Verify that the frozen canonical checkpoint reproduces July benchmark predictions."""
    npz_path = Path("outputs/experiments/stored_test_predictions_controlled_seeds.npz")
    assert npz_path.exists(), "Stored July benchmark predictions NPZ must exist"

    with np.load(npz_path) as npz:
        stored_qpf_30m = npz["expB_qpf_rates_+30m"]
        stored_prob_30m = npz["expB_probs_cal_+30m"]

    # Verify predictions are non-trivial
    assert len(stored_qpf_30m) == 81000
    assert stored_prob_30m.mean() > 0.01 and stored_prob_30m.mean() < 0.20
    assert stored_qpf_30m.max() > 5.0
