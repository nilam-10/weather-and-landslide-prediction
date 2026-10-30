"""Unit tests for experimental spatiotemporal models (SimVP and UNet-ConvLSTM)."""
from __future__ import annotations

import pytest
import torch

from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.unet_convlstm import UNetConvLSTMWeather


def test_simvp_weather_shapes_and_backward():
    """Verify SimVP input/output shape invariants and gradient backpropagation."""
    model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 150_000 <= params <= 500_000, f"Unexpected parameter count: {params}"

    # Forward pass with batch size 2
    dummy_input = torch.randn(2, 4, 3, 25, 40)
    cat_logits, qpf_rates = model(dummy_input)

    assert cat_logits.shape == (2, 5, 25, 40)
    assert qpf_rates.shape == (2, 5, 25, 40)
    assert torch.all(qpf_rates >= 0.0), "QPF rain rates must be non-negative (Softplus invariant)"

    # Backward pass verification
    model.train()
    cat_logits, qpf_rates = model(dummy_input)
    loss = cat_logits.mean() + qpf_rates.mean()
    loss.backward()

    # Check that gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_unet_convlstm_weather_shapes_and_backward():
    """Verify UNet-ConvLSTM input/output shape invariants and multi-scale skip flow."""
    model = UNetConvLSTMWeather(in_channels=3, base_channels=24, num_leads=5)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 150_000 <= params <= 450_000, f"Unexpected parameter count: {params}"

    # Forward pass with batch size 2
    dummy_input = torch.randn(2, 4, 3, 25, 40)
    cat_logits, qpf_rates = model(dummy_input)

    assert cat_logits.shape == (2, 5, 25, 40)
    assert qpf_rates.shape == (2, 5, 25, 40)
    assert torch.all(qpf_rates >= 0.0), "QPF rain rates must be non-negative (Softplus invariant)"

    # Backward pass verification
    model.train()
    cat_logits, qpf_rates = model(dummy_input)
    loss = cat_logits.mean() + qpf_rates.mean()
    loss.backward()

    # Check that gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
