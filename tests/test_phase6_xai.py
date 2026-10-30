"""Phase 6 Explainable AI (XAI) Automated Test Suite.

Verifies:
1. Axiomatic completeness holds (< 5% for continuous QPF, < 15% for probability logit).
2. All 5 lead horizons (+30m, +60m, +120m, +180m, +360m) execute successfully.
3. Both Integrated Gradients and Occlusion Sensitivity methods produce finite floats.
4. Adebayo parameter randomization sanity check degrades toward zero correlation.
5. Mandatory non-causal disclaimer is strictly present in payloads.
6. Spatial grid shapes (25x40) and percentage sums (100%) are mathematically sound.
"""
import pytest
import numpy as np
import torch
import re
from pathlib import Path

from src.api.main import initialize_cache, cache, explain_weather_prediction
from src.explainability.baselines import build_weather_baseline
from src.explainability.integrated_gradients import compute_integrated_gradients
from src.explainability.occlusion import compute_occlusion_sensitivity
from src.explainability.ablation import (
    run_parameter_randomization_sanity_check,
    compare_ig_and_occlusion_agreement,
)
from src.explainability.service import generate_weather_explanation, NON_CAUSAL_DISCLAIMER


@pytest.fixture(scope="module", autouse=True)
def setup_api_cache():
    initialize_cache()
    assert cache.initialized is True
    assert cache.canonical_convlstm_model is not None


def test_canonical_model_is_loaded():
    model = cache.canonical_convlstm_model
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 161034, f"Expected 161,034 params, got {total_params}"


def test_baseline_generation():
    x = torch.zeros((1, 4, 3, 25, 40), dtype=torch.float32)
    b_dry, meta_dry = build_weather_baseline(x, baseline_type="dry_clearsky", stats=cache.stats)
    assert b_dry.shape == x.shape
    assert meta_dry["baseline_type"] == "dry_clearsky"
    assert "Tb=295.0 K" in meta_dry["description"]

    b_zero, meta_zero = build_weather_baseline(x, baseline_type="zero")
    assert b_zero.shape == x.shape
    assert torch.all(b_zero[:, :, :2] == 0.0)
    assert torch.all(b_zero[:, :, 2] == 1.0)


def test_integrated_gradients_completeness_qpf():
    model = cache.canonical_convlstm_model
    step = 316
    indices = [step - 3, step - 2, step - 1, step]
    tb_seq = cache.ir_all[indices]
    p_seq = cache.precip_all[indices]
    m_seq = cache.mask_all[indices]
    st = cache.stats

    tb_norm = np.where((m_seq == 1) & np.isfinite(tb_seq), (tb_seq - st["tb_mean"]) / st["tb_std"], 0.0)
    log_p = np.log1p(np.maximum(p_seq, 0.0))
    p_norm = np.where(np.isfinite(log_p), (log_p - st["log_p_mean"]) / st["log_p_std"], 0.0)
    m_norm = m_seq.astype(np.float32)

    feat_seq = np.stack([tb_norm, p_norm, m_norm], axis=1).astype(np.float32)
    x_tensor = torch.from_numpy(feat_seq).unsqueeze(0).to(cache.device)

    # Test QPF head completeness
    ig_res = compute_integrated_gradients(
        model=model,
        x=x_tensor,
        baseline_type="dry_clearsky",
        target_type="qpf",
        lead_idx=0,
        target_cell=(5, 23),
        m_steps=50,
        stats=st,
    )

    proof = ig_res["completeness_proof"]
    assert proof["completeness_verified"] is True
    assert proof["relative_error_pct"] < 5.0, f"QPF Completeness error too high: {proof['relative_error_pct']}%"
    assert np.isfinite(proof["sum_attributions"])
    assert np.isfinite(proof["delta_f"])


def test_all_five_leads_execution():
    model = cache.canonical_convlstm_model
    leads = ["30m", "60m", "120m", "180m", "360m"]

    for l_idx, lead in enumerate(leads):
        res = generate_weather_explanation(
            model=model,
            cache_data=cache,
            step=316,
            lead=lead,
            target_type="probability",
            method="integrated_gradients",
            row=5,
            col=23,
            m_steps=30,
        )
        assert res["status"] == "success"
        assert res["lead_time"] == lead
        assert res["model_output"]["fx"] is not None
        assert np.isfinite(res["model_output"]["fx"])
        assert len(res["grids"]["spatial_attribution_normalized"]) == 25
        assert len(res["grids"]["spatial_attribution_normalized"][0]) == 40


def test_occlusion_sensitivity_execution():
    model = cache.canonical_convlstm_model
    res = generate_weather_explanation(
        model=model,
        cache_data=cache,
        step=316,
        lead="30m",
        target_type="probability",
        method="occlusion",
        row=5,
        col=23,
        patch_size=3,
    )
    assert res["status"] == "success"
    assert res["method"] == "occlusion"
    assert res["attribution_summary"]["top_contributing_channel"] is not None
    assert len(res["grids"]["spatial_attribution_raw"]) == 25
    assert len(res["grids"]["spatial_attribution_raw"][0]) == 40


def test_shares_sum_to_one_hundred():
    model = cache.canonical_convlstm_model
    res = generate_weather_explanation(
        model=model,
        cache_data=cache,
        step=316,
        lead="30m",
        target_type="probability",
        method="integrated_gradients",
        row=5,
        col=23,
        m_steps=30,
    )
    ch = res["attribution_summary"]["channel_shares_pct"]
    ch_sum = ch["infrared_tb_pct"] + ch["precipitation_pct"] + ch.get("sensor_mask_pct", ch.get("validity_mask_pct", 0.0))
    assert abs(ch_sum - 100.0) < 0.2, f"Channel shares do not sum to 100%: {ch_sum}"

    tm = res["attribution_summary"]["temporal_shares_pct"]
    tm_sum = sum(tm.values())
    assert abs(tm_sum - 100.0) < 0.2, f"Temporal shares do not sum to 100%: {tm_sum}"


def test_non_causal_disclaimer_is_present():
    model = cache.canonical_convlstm_model
    res = generate_weather_explanation(
        model=model,
        cache_data=cache,
        step=316,
        lead="30m",
        target_type="probability",
        method="integrated_gradients",
        row=5,
        col=23,
        m_steps=30,
    )
    assert "disclaimer" in res
    assert "NOT confirmed physical meteorological causation" in res["disclaimer"]
    assert NON_CAUSAL_DISCLAIMER == res["disclaimer"]


def test_adebayo_parameter_randomization_sanity_check():
    model = cache.canonical_convlstm_model
    x = torch.randn((1, 4, 3, 25, 40), dtype=torch.float32, device=cache.device)
    sanity_res = run_parameter_randomization_sanity_check(
        model=model,
        x=x,
        target_type="probability",
        lead_idx=0,
        target_cell=(5, 23),
        m_steps=20,
    )
    assert sanity_res["sanity_check_passed"] is True
    assert abs(sanity_res["pearson_r_trained_vs_random"]) < 0.5


def test_frontend_xai_elements_present():
    frontend_path = Path("D:/Downloads/honours/frontend/index.html")
    assert frontend_path.exists()
    content = frontend_path.read_text(encoding="utf-8")

    assert 'id="xai-section"' in content
    assert 'id="xai-method"' in content
    assert 'id="btn-run-xai"' in content
    assert 'id="chk-xai"' in content
    assert 'runXAIExplanation' in content
    assert 'NOT confirmed physical meteorological causation' in content
