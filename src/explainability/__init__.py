"""Explainable AI (XAI) Module for Weather Nowcasting.

Exports:
- build_weather_baseline: Baselines for gradient and perturbation attributions.
- compute_integrated_gradients: Axiomatic Integrated Gradients for ConvLSTM.
- compute_occlusion_sensitivity: Independent perturbation-based sensitivity maps.
- generate_weather_explanation: High-level service interface for API and dashboard.
- run_channel_and_temporal_ablations: Systematic ablation experiments.
- run_parameter_randomization_sanity_check: Adebayo et al. (2018) sanity checks.
- compare_ig_and_occlusion_agreement: Cross-method correlation evaluation.
"""
from src.explainability.ablation import (
    compare_ig_and_occlusion_agreement,
    run_channel_and_temporal_ablations,
    run_parameter_randomization_sanity_check,
)
from src.explainability.baselines import build_weather_baseline
from src.explainability.integrated_gradients import compute_integrated_gradients
from src.explainability.occlusion import compute_occlusion_sensitivity
from src.explainability.service import generate_weather_explanation

__all__ = [
    "build_weather_baseline",
    "compute_integrated_gradients",
    "compute_occlusion_sensitivity",
    "generate_weather_explanation",
    "run_channel_and_temporal_ablations",
    "run_parameter_randomization_sanity_check",
    "compare_ig_and_occlusion_agreement",
]
