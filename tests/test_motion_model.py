"""Unit tests for MotionSimVPWeather architecture."""
from __future__ import annotations

import torch
from src.models.research_experiments.motion_model import MotionSimVPWeather


def test_motion_simvp_shapes_and_backward():
    """Verify MotionSimVP forward/backward pass with 5 channels [IR, Rain, Mask, U, V]."""
    model = MotionSimVPWeather(in_channels=5, hidden_dim=32, t_in=4, num_leads=5)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 150_000 <= params <= 500_000, f"Unexpected parameter count: {params}"

    dummy_input = torch.randn(2, 4, 5, 25, 40)
    cat_logits, qpf_rates = model(dummy_input)

    assert cat_logits.shape == (2, 5, 25, 40)
    assert qpf_rates.shape == (2, 5, 25, 40)
    assert torch.all(qpf_rates >= 0.0)

    model.train()
    cat_logits, qpf_rates = model(dummy_input)
    loss = cat_logits.mean() + qpf_rates.mean()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
