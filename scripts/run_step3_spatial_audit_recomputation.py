"""Step 3 Spatial Verification Audit Recomputation Script.

Performs:
1. Full inference across all seeds of ConvLSTM, SimVP, and UNet-ConvLSTM.
2. Complete spatial verification across both Square (Step 2) and Circular (Corrected) geometries.
3. Pointwise continuous & contingency metrics (CSI, POD, FAR, RMSE, MAE, Pearson r).
4. Export of CORRECTED_FSS_RESULTS.csv and ORIGINAL_VS_CORRECTED_COMPARISON.csv.
5. In-depth diagnostics of maximum predicted rainfall rates at +180m and +360m.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.convlstm import MultiTaskWeatherConvLSTM
from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.unet_convlstm import UNetConvLSTMWeather
from src.evaluation.spatial_metrics import SpatialVerificationEngine as SVE
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT_ROOT / "outputs/reports/model_research/step3_spatial_verification_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEADS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
RADII_KM = [10.0, 20.0, 30.0, 50.0]
THRESHOLDS = [0.1, 1.0, 5.0]
GRID_KM = 10.55

print("=" * 80)
print("EXECUTING STEP 3 SPATIAL VERIFICATION AUDIT RECOMPUTATION")
print(f"Device: {DEVICE} | Output Directory: {OUT_DIR}")
print("=" * 80)

# 1. Load Primary Test Data
test_dates = ["20230712", "20230713", "20230714"]
te_x, te_y_b, te_y_c, te_m, te_t = build_advanced_sequences(test_dates, t_in=4)

gt_bin = np.stack(te_y_b, axis=0)       # (129, 5, 25, 40)
gt_cont = np.stack(te_y_c, axis=0)      # (129, 5, 25, 40)
mask_stack = np.stack(te_m, axis=0)     # (129, 5, 25, 40)
ref_times = np.array(te_t)

x_tensor = torch.from_numpy(np.stack(te_x, axis=0)).to(DEVICE)
print(f"Loaded test cohort: {len(x_tensor)} sequences from {ref_times[0][:10]} to {ref_times[-1][:10]}.")

# 2. Gather Model Checkpoints
models_qpf: Dict[str, np.ndarray] = {}

# Physical baselines from existing Step 2 cohort
cohort_npz = np.load("outputs/reports/model_research/step2_spatial_verification/test_predictions_cohort.npz")
models_qpf["Persistence"] = cohort_npz["persistence_qpf"]
models_qpf["Dense_Optical_Flow"] = cohort_npz["dense_flow_qpf"]

# ConvLSTM seeds
conv_ckpts = {
    "ConvLSTM_Seed42": "outputs/models/phase3/convlstm_seed42.pt",
    "ConvLSTM_Seed123": "outputs/models/phase3/convlstm_seed123.pt",
    "ConvLSTM_Seed999": "outputs/models/phase3/convlstm_seed999.pt",
}
for name, p in conv_ckpts.items():
    m = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=5).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    with torch.no_grad():
        _, q = m(x_tensor)
        models_qpf[name] = q.cpu().numpy()
models_qpf["ConvLSTM_Mean3Seeds"] = (models_qpf["ConvLSTM_Seed42"] + models_qpf["ConvLSTM_Seed123"] + models_qpf["ConvLSTM_Seed999"]) / 3.0

# SimVP seeds
simvp_ckpts = {
    "SimVP_Seed42": "outputs/experiments/research_models/checkpoints/simvp_seed42.pt",
    "SimVP_Seed123": "outputs/experiments/research_models/checkpoints/simvp_seed123.pt",
    "SimVP_Seed999": "outputs/experiments/research_models/checkpoints/simvp_seed999.pt",
    "SimVP_Optimal": "outputs/experiments/model_research/checkpoints/simvp_optimal_best.pt",
}
for name, p in simvp_ckpts.items():
    m = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    with torch.no_grad():
        _, q = m(x_tensor)
        models_qpf[name] = q.cpu().numpy()
models_qpf["SimVP_Mean3Seeds"] = (models_qpf["SimVP_Seed42"] + models_qpf["SimVP_Seed123"] + models_qpf["SimVP_Seed999"]) / 3.0

# UNet-ConvLSTM seeds
unet_ckpts = {
    "UNet_ConvLSTM_Seed42": "outputs/experiments/research_models/checkpoints/unet_convlstm_seed42.pt",
    "UNet_ConvLSTM_Seed123": "outputs/experiments/research_models/checkpoints/unet_convlstm_seed123.pt",
    "UNet_ConvLSTM_Seed999": "outputs/experiments/research_models/checkpoints/unet_convlstm_seed999.pt",
    "UNet_ConvLSTM_Best": "outputs/experiments/model_research/checkpoints/unet_convlstm_best.pt",
}
for name, p in unet_ckpts.items():
    m = UNetConvLSTMWeather(in_channels=3, base_channels=24, num_leads=5).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    with torch.no_grad():
        _, q = m(x_tensor)
        models_qpf[name] = q.cpu().numpy()
models_qpf["UNet_ConvLSTM_Mean3Seeds"] = (models_qpf["UNet_ConvLSTM_Seed42"] + models_qpf["UNet_ConvLSTM_Seed123"] + models_qpf["UNet_ConvLSTM_Seed999"]) / 3.0

print(f"Successfully assembled {len(models_qpf)} model forecast variants.")

# 3. Comprehensive Evaluation: Pointwise, Square FSS, and Circular FSS
corrected_rows = []
comparison_rows = []
diagnostic_max_rows = []

for m_name, qpf_arr in models_qpf.items():
    print(f"Evaluating model: {m_name}...")
    for h_idx, lead in enumerate(LEADS):
        obs_h = gt_cont[:, h_idx]
        fcst_h = qpf_arr[:, h_idx]
        mask_h = mask_stack[:, h_idx]

        # Continuous Pointwise Metrics
        q_flat = fcst_h.ravel()
        o_flat = obs_h.ravel()
        rmse_val = float(np.sqrt(np.mean((q_flat - o_flat) ** 2)))
        mae_val = float(np.mean(np.abs(q_flat - o_flat)))
        r_val, _ = pearsonr(q_flat, o_flat) if np.std(q_flat) > 1e-6 else (0.0, 1.0)

        # Intensity diagnostics
        obs_max = float(np.max(obs_h))
        fcst_max = float(np.max(fcst_h))
        fcst_p99 = float(np.percentile(fcst_h, 99.0))
        fcst_p95 = float(np.percentile(fcst_h, 95.0))
        diagnostic_max_rows.append({
            "model_name": m_name,
            "horizon": lead,
            "obs_max_mm_hr": round(obs_max, 2),
            "fcst_max_mm_hr": round(fcst_max, 2),
            "fcst_p99_mm_hr": round(fcst_p99, 2),
            "fcst_p95_mm_hr": round(fcst_p95, 2),
            "rmse": round(rmse_val, 4),
            "mae": round(mae_val, 4),
        })

        for th in THRESHOLDS:
            # Pointwise binary contingency
            o_bin = (obs_h >= th).astype(int).ravel()
            f_bin = (fcst_h >= th).astype(int).ravel()
            tp = int(np.sum((o_bin == 1) & (f_bin == 1)))
            fp = int(np.sum((o_bin == 0) & (f_bin == 1)))
            fn = int(np.sum((o_bin == 1) & (f_bin == 0)))
            tn = int(np.sum((o_bin == 0) & (f_bin == 0)))

            csi_pt = float(tp / max(tp + fp + fn, 1))
            pod_pt = float(tp / max(tp + fn, 1))
            far_pt = float(fp / max(tp + fp, 1))

            for r_km in RADII_KM:
                k_cells = SVE.km_to_grid_radius(r_km, GRID_KM)

                # Original Square Kernel Evaluation
                res_sq = SVE.evaluate_sequence_dataset(
                    obs_sequences=obs_h,
                    fcst_sequences=fcst_h,
                    threshold=th,
                    radius_cells=k_cells,
                    mask=mask_h,
                    kernel_type="square",
                )

                # Corrected Circular Kernel Evaluation
                res_cir = SVE.evaluate_sequence_dataset(
                    obs_sequences=obs_h,
                    fcst_sequences=fcst_h,
                    threshold=th,
                    radius_cells=k_cells,
                    mask=mask_h,
                    kernel_type="circular",
                )

                # Circular Neighborhood Contingency across all frames
                n_tp, n_fp, n_fn = 0, 0, 0
                for t in range(len(obs_h)):
                    nc = SVE.compute_neighborhood_contingency(
                        obs_field=obs_h[t],
                        fcst_field=fcst_h[t],
                        threshold=th,
                        radius_cells=k_cells,
                        mask=mask_h[t],
                        kernel_type="circular",
                    )
                    n_tp += nc["neighborhood_tp"]
                    n_fp += nc["neighborhood_fp"]
                    n_fn += nc["neighborhood_fn"]

                denom_n_csi = n_tp + n_fp + n_fn
                n_csi = float(n_tp / denom_n_csi) if denom_n_csi > 0 else 0.0
                denom_n_pod = n_tp + n_fn
                n_pod = float(n_tp / denom_n_pod) if denom_n_pod > 0 else 0.0
                denom_n_far = n_tp + n_fp
                n_far = float(n_fp / denom_n_far) if denom_n_far > 0 else 0.0

                # Geometry details
                k_cir = SVE.create_spatial_kernel(k_cells, "circular")
                k_sq = SVE.create_spatial_kernel(k_cells, "square")
                cir_cells = int(np.sum(k_cir))
                sq_cells = int(np.sum(k_sq))

                fss_diff = round(res_cir["accumulated_fss"] - res_sq["accumulated_fss"], 4)

                # Row for CORRECTED_FSS_RESULTS.csv
                corrected_rows.append({
                    "model_name": m_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "grid_radius_cells": k_cells,
                    "circular_kernel_cells": cir_cells,
                    "effective_circular_area_km2": round(cir_cells * (10.03 * 11.08), 1),
                    "true_circle_area_km2": round(np.pi * (r_km ** 2), 1),
                    "fss_accumulated": res_cir["accumulated_fss"],
                    "fss_mean_frame": res_cir["mean_frame_fss"],
                    "active_event_frames": res_cir["active_event_frames"],
                    "zero_event_frames": res_cir["zero_event_frames"],
                    "pointwise_csi": round(csi_pt, 4),
                    "pointwise_pod": round(pod_pt, 4),
                    "pointwise_far": round(far_pt, 4),
                    "neighborhood_csi": round(n_csi, 4),
                    "neighborhood_pod": round(n_pod, 4),
                    "neighborhood_far": round(n_far, 4),
                    "qpf_rmse": round(rmse_val, 4),
                    "qpf_mae": round(mae_val, 4),
                    "pearson_r": round(r_val, 4),
                })

                # Row for ORIGINAL_VS_CORRECTED_COMPARISON.csv
                comparison_rows.append({
                    "model_name": m_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "grid_radius_cells": k_cells,
                    "square_window_size": f"{2*k_cells+1}x{2*k_cells+1}",
                    "square_kernel_cells": sq_cells,
                    "square_fss_accumulated": res_sq["accumulated_fss"],
                    "circular_kernel_cells": cir_cells,
                    "circular_fss_accumulated": res_cir["accumulated_fss"],
                    "fss_delta_circular_minus_square": fss_diff,
                    "relative_area_reduction_pct": round(100.0 * (1.0 - cir_cells / sq_cells), 1),
                    "pointwise_csi": round(csi_pt, 4),
                    "qpf_rmse": round(rmse_val, 4),
                })

# Save CORRECTED_FSS_RESULTS.csv
p_corr = OUT_DIR / "CORRECTED_FSS_RESULTS.csv"
with open(p_corr, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(corrected_rows[0].keys()))
    writer.writeheader()
    writer.writerows(corrected_rows)
print(f"Saved {len(corrected_rows)} rows to {p_corr.name}")

# Save ORIGINAL_VS_CORRECTED_COMPARISON.csv
p_comp = OUT_DIR / "ORIGINAL_VS_CORRECTED_COMPARISON.csv"
with open(p_comp, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Saved {len(comparison_rows)} rows to {p_comp.name}")

# Save PREDICTION_INTENSITY_DIAGNOSTICS.csv
p_diag = OUT_DIR / "PREDICTION_INTENSITY_DIAGNOSTICS.csv"
with open(p_diag, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(diagnostic_max_rows[0].keys()))
    writer.writeheader()
    writer.writerows(diagnostic_max_rows)
print(f"Saved {len(diagnostic_max_rows)} rows to {p_diag.name}")

print("\n" + "=" * 80)
print("RECOMPUTATION COMPLETE.")
print("=" * 80)
