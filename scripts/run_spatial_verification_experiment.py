"""Spatial Verification Experiment Execution Engine.

Executes:
  1. Controlled Displacement Sensitivity Benchmark
  2. Multi-Model, Multi-Horizon, Multi-Threshold FSS Evaluation
  3. Paired Block Bootstrap Uncertainty Quantification
  4. Publication-Quality Visualization Suite
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "outputs" / ".matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import pearsonr

from src.evaluation.spatial_metrics import SpatialVerificationEngine

OUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "model_research" / "step2_spatial_verification"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = OUT_DIR / "test_predictions_cohort.npz"
if not COHORT_PATH.exists():
    raise FileNotFoundError(f"Missing test predictions cohort: {COHORT_PATH}")

npz = np.load(COHORT_PATH)
gt_cont = npz["ground_truth_continuous"]       # (129, 5, 25, 40)
gt_bin = npz["ground_truth_binary"]           # (129, 5, 25, 40)
mask_stack = npz["observation_mask"]          # (129, 5, 25, 40)
ref_times = npz["reference_times"]

LEADS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
THRESHOLDS = [0.1, 1.0, 5.0]  # mm/hr
RADII_KM = [10.0, 20.0, 30.0, 50.0]
GRID_KM = SpatialVerificationEngine.DEFAULT_GRID_KM  # 10.55 km

eng = SpatialVerificationEngine

print("=" * 80)
print("EXECUTING RIGOROUS SPATIAL VERIFICATION EXPERIMENT (FSS & NEIGHBORHOOD METRICS)")
print("=" * 80)


# ==============================================================================
# SECTION 1: CONTROLLED DISPLACEMENT SENSITIVITY BENCHMARK
# ==============================================================================
print("\n>>> Section 1: Running Controlled Displacement Sensitivity Benchmark...")

# Extract a representative convective case from test set (step 35, lead +30m)
sample_idx = 35
sample_obs = gt_cont[sample_idx, 0]  # (25, 40)
sample_mask = mask_stack[sample_idx, 0]

displacement_rows = []

# Define synthetic shifts in grid cells (0, 1, 2, 3, 5 cells)
shift_scenarios = [
    ("Shift_0_Cells_Perfect", 0, 0, 1.0),
    ("Shift_1_Cell_10km", 0, 1, 1.0),
    ("Shift_2_Cells_20km", 0, 2, 1.0),
    ("Shift_3_Cells_30km", 0, 3, 1.0),
    ("Shift_5_Cells_50km", 0, 5, 1.0),
    ("Shift_2_Cells_HalfIntensity", 0, 2, 0.5),
    ("Shift_Diagonal_2x2_Cells", 2, 2, 1.0),
]

for name, sy, sx, scale in shift_scenarios:
    # Shift field using nearest/constant pad
    fcst_field = ndi.shift(sample_obs, [sy, sx], order=1, mode="constant", cval=0.0) * scale

    # Pointwise contingency at 5.0 mm/hr
    o_b = sample_obs >= 5.0
    f_b = fcst_field >= 5.0
    tp = np.sum(o_b & f_b)
    fp = np.sum(~o_b & f_b)
    fn = np.sum(o_b & ~f_b)
    point_csi = float(tp / max(tp + fp + fn, 1))
    point_pod = float(tp / max(tp + fn, 1))
    point_far = float(fp / max(tp + fp, 1))

    # Evaluate across radii (0, 10, 20, 30, 50 km)
    fss_scores = {}
    n_csi_scores = {}
    for r_km in [0.0, 10.0, 20.0, 30.0, 50.0]:
        k = eng.km_to_grid_radius(r_km, GRID_KM)
        res_fss = eng.compute_single_fss(sample_obs, fcst_field, threshold=5.0, radius_cells=k, mask=sample_mask)
        res_n = eng.compute_neighborhood_contingency(sample_obs, fcst_field, threshold=5.0, radius_cells=k, mask=sample_mask)
        fss_scores[f"fss_{int(r_km)}km"] = round(res_fss["fss"], 4)
        n_csi_scores[f"n_csi_{int(r_km)}km"] = round(res_n["neighborhood_csi"], 4)

    displacement_rows.append({
        "scenario_name": name,
        "shift_cells_y": sy,
        "shift_cells_x": sx,
        "shift_km_approx": round(np.sqrt((sy * 11.08)**2 + (sx * 10.03)**2), 1),
        "intensity_scaling": scale,
        "pointwise_csi": round(point_csi, 4),
        "pointwise_pod": round(point_pod, 4),
        "pointwise_far": round(point_far, 4),
        **fss_scores,
        **n_csi_scores,
    })

# Add extreme edge case scenarios
edge_cases = [
    ("EdgeCase_ZeroForecast_MissAll", np.zeros_like(sample_obs)),
    ("EdgeCase_ZeroObs_FalseAlarmAll", sample_obs.copy()),  # we invert obs/fcst
    ("EdgeCase_BothEmpty_NoRain", np.zeros_like(sample_obs)),
]

for ec_name, f_field in edge_cases:
    o_field = np.zeros_like(sample_obs) if "ZeroObs" in ec_name or "BothEmpty" in ec_name else sample_obs

    o_b = o_field >= 5.0
    f_b = f_field >= 5.0
    tp = np.sum(o_b & f_b)
    fp = np.sum(~o_b & f_b)
    fn = np.sum(o_b & ~f_b)
    point_csi = float(tp / max(tp + fp + fn, 1)) if (tp + fp + fn) > 0 else (1.0 if np.sum(o_b) == 0 and np.sum(f_b) == 0 else 0.0)
    point_pod = float(tp / max(tp + fn, 1)) if (tp + fn) > 0 else 0.0
    point_far = float(fp / max(tp + fp, 1)) if (tp + fp) > 0 else 0.0

    fss_scores = {}
    n_csi_scores = {}
    for r_km in [0.0, 10.0, 20.0, 30.0, 50.0]:
        k = eng.km_to_grid_radius(r_km, GRID_KM)
        res_fss = eng.compute_single_fss(o_field, f_field, threshold=5.0, radius_cells=k, mask=sample_mask)
        res_n = eng.compute_neighborhood_contingency(o_field, f_field, threshold=5.0, radius_cells=k, mask=sample_mask)
        fss_scores[f"fss_{int(r_km)}km"] = round(res_fss["fss"], 4)
        n_csi_scores[f"n_csi_{int(r_km)}km"] = round(res_n["neighborhood_csi"], 4)

    displacement_rows.append({
        "scenario_name": ec_name,
        "shift_cells_y": -1,
        "shift_cells_x": -1,
        "shift_km_approx": -1.0,
        "intensity_scaling": 0.0,
        "pointwise_csi": round(point_csi, 4),
        "pointwise_pod": round(point_pod, 4),
        "pointwise_far": round(point_far, 4),
        **fss_scores,
        **n_csi_scores,
    })

# Save Section 1 CSV
csv_disp = OUT_DIR / "DISPLACEMENT_SENSITIVITY.csv"
with open(csv_disp, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(displacement_rows[0].keys()))
    writer.writeheader()
    writer.writerows(displacement_rows)
print(f"Saved {len(displacement_rows)} displacement sensitivity scenarios to: {csv_disp.name}")


# ==============================================================================
# SECTION 2: MULTI-MODEL, MULTI-HORIZON, MULTI-THRESHOLD FSS EVALUATION
# ==============================================================================
print("\n>>> Section 2: Evaluating All Models Across Horizons, Thresholds, and Radii...")

# Assemble model QPF dictionary
models_qpf = {
    "ConvLSTM_Seed42": npz["convlstm_seed42_qpf"],
    "ConvLSTM_Seed123": npz["convlstm_seed123_qpf"],
    "ConvLSTM_Seed999": npz["convlstm_seed999_qpf"],
    "ConvLSTM_Mean3Seeds": (npz["convlstm_seed42_qpf"] + npz["convlstm_seed123_qpf"] + npz["convlstm_seed999_qpf"]) / 3.0,
    "SimVP_Optimal": npz["simvp_qpf"],
    "UNet_ConvLSTM": npz["unet_convlstm_qpf"],
    "Persistence": npz["persistence_qpf"],
    "Optical_Flow_Dense": npz["dense_flow_qpf"],
}

fss_master_rows = []
neighborhood_master_rows = []

for m_name, qpf_arr in models_qpf.items():
    print(f"  Evaluating {m_name}...")
    for h_idx, lead in enumerate(LEADS):
        obs_h = gt_cont[:, h_idx]  # (129, 25, 40)
        fcst_h = qpf_arr[:, h_idx] # (129, 25, 40)
        mask_h = mask_stack[:, h_idx] # spatial mask (129, 25, 40)

        # Compute Pointwise Continuous Metrics
        q_flat = fcst_h.ravel()
        o_flat = obs_h.ravel()
        rmse_val = float(np.sqrt(np.mean((q_flat - o_flat)**2)))
        mae_val = float(np.mean(np.abs(q_flat - o_flat)))
        r_val, _ = pearsonr(q_flat, o_flat) if np.std(q_flat) > 1e-6 else (0.0, 1.0)

        for th in THRESHOLDS:
            # Pointwise Contingency for this threshold
            o_bin = (obs_h >= th).astype(int).ravel()
            f_bin = (fcst_h >= th).astype(int).ravel()
            tp_pt = int(np.sum((o_bin == 1) & (f_bin == 1)))
            fp_pt = int(np.sum((o_bin == 0) & (f_bin == 1)))
            fn_pt = int(np.sum((o_bin == 1) & (f_bin == 0)))
            tn_pt = int(np.sum((o_bin == 0) & (f_bin == 0)))

            csi_pt = float(tp_pt / max(tp_pt + fp_pt + fn_pt, 1))
            pod_pt = float(tp_pt / max(tp_pt + fn_pt, 1))
            far_pt = float(fp_pt / max(tp_pt + fp_pt, 1))

            for r_km in RADII_KM:
                k_cells = eng.km_to_grid_radius(r_km, GRID_KM)

                # Compute Cohort FSS (Accumulated & Mean)
                fss_res = eng.evaluate_sequence_dataset(
                    obs_sequences=obs_h,
                    fcst_sequences=fcst_h,
                    threshold=th,
                    radius_cells=k_cells,
                    mask=mask_h,
                )

                # Compute Neighborhood Contingency across all frames
                n_tp, n_fp, n_fn = 0, 0, 0
                for t in range(len(obs_h)):
                    nc = eng.compute_neighborhood_contingency(
                        obs_field=obs_h[t],
                        fcst_field=fcst_h[t],
                        threshold=th,
                        radius_cells=k_cells,
                        mask=mask_h[t],
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
                n_f1 = float(2 * n_tp / (2 * n_tp + n_fp + n_fn)) if (2 * n_tp + n_fp + n_fn) > 0 else 0.0

                row_dict = {
                    "model_name": m_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "grid_radius_cells": k_cells,
                    "window_size_cells": 2 * k_cells + 1,
                    "fss_accumulated": fss_res["accumulated_fss"],
                    "fss_mean_frame": fss_res["mean_frame_fss"],
                    "active_event_frames": fss_res["active_event_frames"],
                    "zero_event_frames": fss_res["zero_event_frames"],
                    "pointwise_csi": round(csi_pt, 4),
                    "pointwise_pod": round(pod_pt, 4),
                    "pointwise_far": round(far_pt, 4),
                    "neighborhood_csi": round(n_csi, 4),
                    "neighborhood_pod": round(n_pod, 4),
                    "neighborhood_far": round(n_far, 4),
                    "neighborhood_f1": round(n_f1, 4),
                    "qpf_rmse": round(rmse_val, 4),
                    "qpf_mae": round(mae_val, 4),
                    "pearson_r": round(float(r_val), 4),
                }
                fss_master_rows.append(row_dict)

                # Summary table for neighborhood comparison
                neighborhood_master_rows.append({
                    "model_name": m_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "fss_accumulated": fss_res["accumulated_fss"],
                    "pointwise_csi": round(csi_pt, 4),
                    "neighborhood_csi": round(n_csi, 4),
                    "csi_relative_gain_pct": round(((n_csi - csi_pt) / max(csi_pt, 1e-4)) * 100, 1),
                    "pointwise_far": round(far_pt, 4),
                    "neighborhood_far": round(n_far, 4),
                    "far_relative_reduction_pct": round(((far_pt - n_far) / max(far_pt, 1e-4)) * 100, 1),
                })

# Save Section 2 CSVs
csv_fss = OUT_DIR / "FSS_RESULTS.csv"
with open(csv_fss, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(fss_master_rows[0].keys()))
    writer.writeheader()
    writer.writerows(fss_master_rows)
print(f"Saved {len(fss_master_rows)} full FSS results to: {csv_fss.name}")

csv_neigh = OUT_DIR / "NEIGHBORHOOD_METRICS.csv"
with open(csv_neigh, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(neighborhood_master_rows[0].keys()))
    writer.writeheader()
    writer.writerows(neighborhood_master_rows)
print(f"Saved {len(neighborhood_master_rows)} neighborhood metric comparison rows to: {csv_neigh.name}")


# ==============================================================================
# SECTION 3: PAIRED BLOCK BOOTSTRAP UNCERTAINTY QUANTIFICATION
# ==============================================================================
print("\n>>> Section 3: Performing Paired Block Bootstrap Uncertainty Analysis...")

# The test cohort spans 72 hours (129 sequences)
# Construct 12 non-overlapping 6-hour blocks (each ~10-11 sequences)
N_seq = len(gt_cont)
block_size = 11
num_blocks = 12
block_indices = [list(range(i * block_size, min((i + 1) * block_size, N_seq))) for i in range(num_blocks)]
# Ensure all sequences covered in last block
block_indices[-1].extend(list(range(num_blocks * block_size, N_seq)))

np.random.seed(42)
N_BOOTSTRAPS = 1000

uncertainty_rows = []

# Analyze primary models: ConvLSTM_Seed42 vs SimVP_Optimal at threshold 5.0 mm/hr
for lead_idx, lead in enumerate(LEADS):
    for r_km in [10.0, 30.0]:
        k_cells = eng.km_to_grid_radius(r_km, GRID_KM)
        
        conv_q = models_qpf["ConvLSTM_Seed42"][:, lead_idx]
        simvp_q = models_qpf["SimVP_Optimal"][:, lead_idx]
        obs_h = gt_cont[:, lead_idx]
        mask_h = mask_stack[:, lead_idx]

        # Precompute frame-level FBS and FBS_worst across the 129 test sequences
        fbs_c_list = np.zeros(N_seq, dtype=np.float64)
        fbs_worst_c_list = np.zeros(N_seq, dtype=np.float64)
        fbs_s_list = np.zeros(N_seq, dtype=np.float64)
        fbs_worst_s_list = np.zeros(N_seq, dtype=np.float64)

        for t in range(N_seq):
            rc = eng.compute_single_fss(obs_h[t], conv_q[t], threshold=5.0, radius_cells=k_cells, mask=mask_h[t])
            rs = eng.compute_single_fss(obs_h[t], simvp_q[t], threshold=5.0, radius_cells=k_cells, mask=mask_h[t])
            if not rc["is_zero_event"]:
                fbs_c_list[t] = rc["fbs"]
                fbs_worst_c_list[t] = rc["fbs_worst"]
            if not rs["is_zero_event"]:
                fbs_s_list[t] = rs["fbs"]
                fbs_worst_s_list[t] = rs["fbs_worst"]

        conv_boot_fss = []
        simvp_boot_fss = []
        diff_boot_fss = []

        for _ in range(N_BOOTSTRAPS):
            sampled_block_ids = np.random.choice(num_blocks, size=num_blocks, replace=True)
            sampled_seq_idx = []
            for b_id in sampled_block_ids:
                sampled_seq_idx.extend(block_indices[b_id])
            s_idx = np.array(sampled_seq_idx)

            den_c = np.sum(fbs_worst_c_list[s_idx])
            f_c = float(np.clip(1.0 - (np.sum(fbs_c_list[s_idx]) / den_c), 0.0, 1.0)) if den_c > 1e-12 else 1.0

            den_s = np.sum(fbs_worst_s_list[s_idx])
            f_s = float(np.clip(1.0 - (np.sum(fbs_s_list[s_idx]) / den_s), 0.0, 1.0)) if den_s > 1e-12 else 1.0

            conv_boot_fss.append(f_c)
            simvp_boot_fss.append(f_s)
            diff_boot_fss.append(f_s - f_c)

        c_mean = float(np.mean(conv_boot_fss))
        c_ci_low = float(np.percentile(conv_boot_fss, 2.5))
        c_ci_high = float(np.percentile(conv_boot_fss, 97.5))

        s_mean = float(np.mean(simvp_boot_fss))
        s_ci_low = float(np.percentile(simvp_boot_fss, 2.5))
        s_ci_high = float(np.percentile(simvp_boot_fss, 97.5))

        d_mean = float(np.mean(diff_boot_fss))
        d_ci_low = float(np.percentile(diff_boot_fss, 2.5))
        d_ci_high = float(np.percentile(diff_boot_fss, 97.5))
        p_val = float(np.mean(np.array(diff_boot_fss) <= 0.0))  # one-sided test: SimVP <= ConvLSTM

        uncertainty_rows.append({
            "horizon": lead,
            "threshold_mm_hr": 5.0,
            "neighborhood_radius_km": r_km,
            "convlstm_fss_mean": round(c_mean, 4),
            "convlstm_fss_ci95_low": round(c_ci_low, 4),
            "convlstm_fss_ci95_high": round(c_ci_high, 4),
            "simvp_fss_mean": round(s_mean, 4),
            "simvp_fss_ci95_low": round(s_ci_low, 4),
            "simvp_fss_ci95_high": round(s_ci_high, 4),
            "fss_diff_simvp_minus_conv": round(d_mean, 4),
            "fss_diff_ci95_low": round(d_ci_low, 4),
            "fss_diff_ci95_high": round(d_ci_high, 4),
            "bootstrap_p_value": round(p_val, 4),
            "statistically_significant": bool(p_val < 0.05 or (1 - p_val) < 0.05),
        })

csv_unc = OUT_DIR / "FSS_UNCERTAINTY.csv"
with open(csv_unc, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(uncertainty_rows[0].keys()))
    writer.writeheader()
    writer.writerows(uncertainty_rows)
print(f"Saved paired block bootstrap uncertainty table to: {csv_unc.name}")


# ==============================================================================
# SECTION 4: PUBLICATION-GRADE VISUALIZATIONS
# ==============================================================================
print("\n>>> Section 4: Generating Publication-Grade Visualizations...")
import pandas as pd
df_fss = pd.DataFrame(fss_master_rows)

# ------------------------------------------------------------------------------
# Fig 1: FSS vs Neighborhood Radius for Every Horizon (Threshold = 5.0 mm/hr, SimVP)
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=180)
sub_simvp = df_fss[(df_fss["model_name"] == "SimVP_Optimal") & (df_fss["threshold_mm_hr"] == 5.0)]
palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
markers = ["o", "s", "^", "D", "v"]

for idx, lead in enumerate(LEADS):
    sub = sub_simvp[sub_simvp["horizon"] == lead]
    plt.plot(sub["neighborhood_radius_km"], sub["fss_accumulated"], marker=markers[idx], color=palette[idx], linewidth=2.2, label=f"Lead {lead}")

plt.axhline(0.5, color="black", linestyle="--", alpha=0.7, label="Useful Skill Threshold (FSS = 0.50)")
plt.title("Fractions Skill Score (FSS) vs. Spatial Neighborhood Radius\n(Heavy Convective Precipitation >= 5.0 mm/hr, SimVP Model)", fontsize=12, fontweight="bold")
plt.xlabel("Spatial Neighborhood Radius (km)", fontsize=11)
plt.ylabel("Accumulated Fractions Skill Score (FSS)", fontsize=11)
plt.ylim(-0.02, 1.02)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True, fontsize=9.5, loc="lower right")
plt.tight_layout()
fig1_path = FIG_DIR / "fig_fss_vs_radius.png"
plt.savefig(fig1_path)
plt.close()

# ------------------------------------------------------------------------------
# Fig 2: FSS vs Forecast Horizon for Each Radius (Threshold = 5.0 mm/hr)
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=180)
radii_palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
radii_markers = ["o", "s", "^", "D"]

for idx, r_km in enumerate(RADII_KM):
    sub = sub_simvp[sub_simvp["neighborhood_radius_km"] == r_km]
    plt.plot(sub["horizon"], sub["fss_accumulated"], marker=radii_markers[idx], color=radii_palette[idx], linewidth=2.2, label=f"Radius {int(r_km)} km")

plt.axhline(0.5, color="black", linestyle="--", alpha=0.7, label="Useful Skill Target (0.50)")
plt.title("FSS Decay Over Forecast Horizon by Spatial Neighborhood Scale\n(Heavy Convective Rain >= 5.0 mm/hr, SimVP)", fontsize=12, fontweight="bold")
plt.xlabel("Forecast Horizon", fontsize=11)
plt.ylabel("Accumulated Fractions Skill Score (FSS)", fontsize=11)
plt.ylim(-0.02, 1.02)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True, fontsize=9.5, loc="lower left")
plt.tight_layout()
fig2_path = FIG_DIR / "fig_fss_vs_lead_time.png"
plt.savefig(fig2_path)
plt.close()

# ------------------------------------------------------------------------------
# Fig 3: Model Comparison FSS (ConvLSTM vs SimVP vs UNet-ConvLSTM vs Baselines)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5.5), dpi=180)
eval_models = [
    ("ConvLSTM_Seed42", "ConvLSTM (Seed 42)", "#1f77b4", "o"),
    ("SimVP_Optimal", "SimVP (Optimal)", "#2ca02c", "s"),
    ("UNet_ConvLSTM", "U-Net-ConvLSTM", "#ff7f0e", "^"),
    ("Persistence", "Persistence", "#7f7f7f", "--"),
    ("Optical_Flow_Dense", "Dense Optical Flow", "#8c564b", ":"),
]

for m_key, m_lbl, col, m_style in eval_models:
    sub = df_fss[(df_fss["model_name"] == m_key) & (df_fss["threshold_mm_hr"] == 5.0) & (df_fss["neighborhood_radius_km"] == 30.0)]
    if "--" in m_style or ":" in m_style:
        plt.plot(sub["horizon"], sub["fss_accumulated"], linestyle=m_style, color=col, linewidth=1.8, label=m_lbl)
    else:
        plt.plot(sub["horizon"], sub["fss_accumulated"], marker=m_style, color=col, linewidth=2.2, label=m_lbl)

plt.axhline(0.5, color="black", linestyle="--", alpha=0.7, label="Useful Skill (FSS = 0.50)")
plt.title("Architecture Comparison: Spatial Skill (FSS at 30 km Radius)\n(Heavy Convective Precipitation >= 5.0 mm/hr)", fontsize=12, fontweight="bold")
plt.xlabel("Forecast Horizon", fontsize=11)
plt.ylabel("FSS (30 km Radius)", fontsize=11)
plt.ylim(-0.02, 1.02)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True, fontsize=9.5, loc="lower left")
plt.tight_layout()
fig3_path = FIG_DIR / "fig_model_comparison_fss.png"
plt.savefig(fig3_path)
plt.close()

# ------------------------------------------------------------------------------
# Fig 4: Heavy Rain FSS at 5.0 mm/hr Across All Models & Radii (+60m and +120m)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)

for idx, target_lead in enumerate(["+60m", "+120m"]):
    ax = axes[idx]
    for m_key, m_lbl, col, m_style in [
        ("ConvLSTM_Seed42", "ConvLSTM", "#1f77b4", "o"),
        ("SimVP_Optimal", "SimVP", "#2ca02c", "s"),
        ("UNet_ConvLSTM", "U-Net-ConvLSTM", "#ff7f0e", "^"),
        ("Persistence", "Persistence", "#7f7f7f", "x"),
    ]:
        sub = df_fss[(df_fss["model_name"] == m_key) & (df_fss["threshold_mm_hr"] == 5.0) & (df_fss["horizon"] == target_lead)]
        ax.plot(sub["neighborhood_radius_km"], sub["fss_accumulated"], marker=m_style, color=col, linewidth=2.0, label=m_lbl)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.7, label="Useful Target (0.50)")
    ax.set_title(f"Heavy Rain FSS at {target_lead} Lead Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Neighborhood Radius (km)", fontsize=10)
    ax.set_ylabel("Accumulated FSS", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

plt.suptitle("Heavy Rain (>= 5.0 mm/hr) Spatial Skill Resolution Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig4_path = FIG_DIR / "fig_heavy_rain_fss_5mm.png"
plt.savefig(fig4_path)
plt.close()

# ------------------------------------------------------------------------------
# Fig 5: Pointwise CSI vs. Neighborhood FSS (Revealing Hidden Regional Skill)
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=180)
sub_comp = df_fss[(df_fss["model_name"] == "SimVP_Optimal") & (df_fss["threshold_mm_hr"] == 5.0) & (df_fss["neighborhood_radius_km"] == 30.0)]

x = np.arange(len(LEADS))
width = 0.35

rects1 = plt.bar(x - width/2, sub_comp["pointwise_csi"], width, label="Pointwise CSI (Strict 10x10 km Grid)", color="#d62728", alpha=0.85)
rects2 = plt.bar(x + width/2, sub_comp["fss_accumulated"], width, label="Spatial FSS (30 km Neighborhood)", color="#2ca02c", alpha=0.85)

plt.title("Pointwise Grid-Cell CSI vs. Spatial Neighborhood FSS\n(Revealing Hidden Regional Forecast Skill in SimVP)", fontsize=12, fontweight="bold")
plt.xlabel("Forecast Horizon", fontsize=11)
plt.ylabel("Score", fontsize=11)
plt.xticks(x, LEADS)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=":", alpha=0.6, axis="y")
plt.legend(frameon=True, fontsize=10, loc="upper right")

for rect in rects1:
    h = rect.get_height()
    plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
for rect in rects2:
    h = rect.get_height()
    plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
fig5_path = FIG_DIR / "fig_pointwise_csi_vs_neighborhood_fss.png"
plt.savefig(fig5_path)
plt.close()

# ------------------------------------------------------------------------------
# Fig 6: Spatial Displacement Case Study Map
# ------------------------------------------------------------------------------
step_vis = 35
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)

o_map = gt_cont[step_vis, 1]  # +60m
c_map = models_qpf["ConvLSTM_Seed42"][step_vis, 1]
s_map = models_qpf["SimVP_Optimal"][step_vis, 1]
vmax = max(o_map.max(), 15.0)

im0 = axes[0].imshow(o_map, cmap="YlGnBu", vmin=0, vmax=vmax, origin="lower")
axes[0].contour(o_map >= 5.0, levels=[0.5], colors=["red"], linewidths=[2])
axes[0].set_title(f"Ground Truth Observed (+60m)\n{ref_times[step_vis]}\n(Red contour: >= 5 mm/hr)", fontsize=10, fontweight="bold")

im1 = axes[1].imshow(c_map, cmap="YlGnBu", vmin=0, vmax=vmax, origin="lower")
axes[1].contour(o_map >= 5.0, levels=[0.5], colors=["red"], linewidths=[1.5], linestyles="--")
axes[1].contour(c_map >= 5.0, levels=[0.5], colors=["black"], linewidths=[2])
axes[1].set_title("ConvLSTM Forecast (+60m)\n(Black: Pred >= 5 mm/hr | Red dashed: Obs)", fontsize=10, fontweight="bold")

im2 = axes[2].imshow(s_map, cmap="YlGnBu", vmin=0, vmax=vmax, origin="lower")
axes[2].contour(o_map >= 5.0, levels=[0.5], colors=["red"], linewidths=[1.5], linestyles="--")
axes[2].contour(s_map >= 5.0, levels=[0.5], colors=["black"], linewidths=[2])
axes[2].set_title("SimVP Forecast (+60m)\n(Black: Pred >= 5 mm/hr | Red dashed: Obs)", fontsize=10, fontweight="bold")

for ax in axes:
    ax.set_xlabel("Longitude Index (0.1 deg)", fontsize=9)
    ax.set_ylabel("Latitude Index (0.1 deg)", fontsize=9)

cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.15, label="Rain Rate (mm/hr)")
plt.suptitle("Spatial Displacement Case Study: Observed Storm Core vs. Model Predictions over Meghalaya", fontsize=13, fontweight="bold")
fig6_path = FIG_DIR / "fig_spatial_displacement_example.png"
plt.savefig(fig6_path, bbox_inches="tight")
plt.close()

print(f"Generated all 6 publication figures in: {FIG_DIR}")
print("=" * 80)
print("SPATIAL VERIFICATION EXPERIMENT COMPLETE.")
print("=" * 80)
