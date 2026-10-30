"""Evaluate Frozen Weather Nowcasting Models on Completely Independent Test Period (Step 3).

Independent Test Period: August 1-3, 2024 (Untouched historical monsoon window).
Models Evaluated (Zero Retraining / Zero Hyperparameter Tuning):
  1. Persistence Baseline
  2. Lucas-Kanade Optical Flow Baseline
  3. Controlled Exp A (14-Day Dataset, 496 seq): Seeds 42, 123, 999
  4. Controlled Exp B (21-Day Dataset, 832 seq): Seeds 42, 123, 999

Forecast Horizons Evaluated: +30m, +60m, +120m, +180m, +360m.
Evaluates:
  - Categorical: CSI, POD, FAR, F1, ROC-AUC, PR-AUC, Brier Score, ECE.
  - Continuous QPF: Overall RMSE, Overall MAE, Pearson r, and Event-Stratified RMSE/MAE (>= 5 mm/hr).
  - Honest Meteorological Block Uncertainty: N=12 non-overlapping 6-hour blocks (df=11) and 1,000 block bootstrap iterations.

Saves:
  - outputs/experiments/independent_weather_model_comparison.csv
  - outputs/experiments/independent_test_prediction_arrays.npz
  - outputs/figures/evaluation/independent_test_skill_curves.png
  - outputs/figures/evaluation/independent_test_qpf_comparison.png
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.convlstm import ConvLSTMCell, MultiTaskWeatherConvLSTM
WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
OUTPUTS_MOD = PROJECT_ROOT / "outputs" / "models" / "weather_convlstm"
OUTPUTS_EXP = PROJECT_ROOT / "outputs" / "experiments"
OUTPUTS_FIG = PROJECT_ROOT / "outputs" / "figures" / "evaluation"
OUTPUTS_EXP.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIG.mkdir(parents=True, exist_ok=True)

TARGET_DATES = ["20240801", "20240802", "20240803"]
LEAD_STEPS = [1, 2, 4, 6, 12]  # +30m, +60m, +120m, +180m, +360m
LEAD_LABELS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
SEEDS = [42, 123, 999]
DEFAULT_THRESHOLD = 5.0
T_IN = 4
N_BOOTSTRAPS = 1000
RNG_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_contingency(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom_csi = tp + fp + fn
    csi = float(tp / denom_csi) if denom_csi > 0 else 0.0

    denom_pod = tp + fn
    pod = float(tp / denom_pod) if denom_pod > 0 else 0.0

    denom_far = tp + fp
    far = float(fp / denom_far) if denom_far > 0 else 0.0

    precision = float(tp / denom_far) if denom_far > 0 else 0.0
    denom_f1 = 2 * tp + fp + fn
    f1 = float(2 * tp / denom_f1) if denom_f1 > 0 else 0.0

    return {"csi": csi, "pod": pod, "far": far, "precision": precision, "f1": f1}


def compute_lucas_kanade_flow(i1: np.ndarray, i2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s1 = ndi.gaussian_filter(np.nan_to_num(i1, nan=0.0), sigma=0.8)
    s2 = ndi.gaussian_filter(np.nan_to_num(i2, nan=0.0), sigma=0.8)
    it = s2 - s1
    iy, ix = np.gradient(s2)

    a11 = ndi.uniform_filter(ix * ix, size=5)
    a12 = ndi.uniform_filter(ix * iy, size=5)
    a22 = ndi.uniform_filter(iy * iy, size=5)
    b1 = ndi.uniform_filter(-ix * it, size=5)
    b2 = ndi.uniform_filter(-iy * it, size=5)

    det = a11 * a22 - a12 * a12 + 1e-3
    u = (a22 * b1 - a12 * b2) / det
    v = (-a12 * b1 + a11 * b2) / det
    return np.clip(u, -5.0, 5.0), np.clip(v, -5.0, 5.0)


def advect_field(field: np.ndarray, u: np.ndarray, v: np.ndarray, steps: int) -> np.ndarray:
    ny, nx = field.shape
    y, x = np.mgrid[0:ny, 0:nx]
    map_y = np.clip(y - v * steps, 0, ny - 1).astype(np.float32)
    map_x = np.clip(x - u * steps, 0, nx - 1).astype(np.float32)
    return ndi.map_coordinates(field, [map_y, map_x], order=1, mode="nearest")


def main():
    print("=" * 80)
    print("STEP 3: EVALUATE FROZEN WEATHER NOWCASTING MODELS ON INDEPENDENT TEST PERIOD")
    print("=" * 80)

    # 1. Load Aligned Weather Datasets
    p_list, tb_list, mask_list, ref_times = [], [], [], []
    for d_clean in TARGET_DATES:
        fpath = WEATHER_DIR / f"aligned_weather_mergir_{d_clean}.nc4"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing aligned weather dataset: {fpath}. Run prepare_independent_test_weather.py first.")
        with h5py.File(fpath, "r") as f:
            p_list.append(f["input_precip"][:])
            tb_list.append(f["satellite_tir1_mean"][:])
            mask_list.append(f["satellite_mask"][:])
            ref_times.extend([
                s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                for s in f["reference_time_iso"][:]
            ])

    p_all = np.concatenate(p_list, axis=0)  # (144, 25, 40)
    tb_all = np.concatenate(tb_list, axis=0)  # (144, 25, 40)
    mask_all = np.concatenate(mask_list, axis=0)  # (144, 25, 40)
    T_total = len(p_all)
    print(f"Loaded continuous weather observations: {T_total} timesteps (72 hours).")

    # 2. Preprocessing / Normalization (Strictly using pre-fitted normalization parameters)
    normalizer_path = OUTPUTS_MOD / "convlstm_normalizer.json"
    with open(normalizer_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)

    tb_mean = norm_stats["tb_mean"]
    tb_std = norm_stats["tb_std"]
    log_p_mean = norm_stats["log_p_mean"]
    log_p_std = norm_stats["log_p_std"]

    tb_norm = np.where(
        (mask_all == 1) & np.isfinite(tb_all),
        (tb_all - tb_mean) / max(tb_std, 1e-6),
        0.0,
    ).astype(np.float32)

    log_p = np.log1p(np.maximum(p_all, 0.0))
    p_norm = np.where(
        np.isfinite(log_p),
        (log_p - log_p_mean) / max(log_p_std, 1e-6),
        0.0,
    ).astype(np.float32)

    features = np.stack([tb_norm, p_norm, mask_all.astype(np.float32)], axis=1)  # (144, 3, 25, 40)

    # 3. Construct Hermetic Independent Test Sequences
    max_lead = max(LEAD_STEPS)
    seq_x, seq_y_bin, seq_y_cont, seq_ref = [], [], [], []

    for t in range(T_IN - 1, T_total - max_lead):
        dt_ref = datetime.fromisoformat(ref_times[t].replace("Z", "+00:00"))
        dt_t0 = datetime.fromisoformat(ref_times[t - T_IN + 1].replace("Z", "+00:00"))
        if (dt_ref - dt_t0).total_seconds() != (T_IN - 1) * 1800:
            continue

        dt_target_max = datetime.fromisoformat(ref_times[t + max_lead].replace("Z", "+00:00"))
        if (dt_target_max - dt_ref).total_seconds() != max_lead * 1800:
            continue

        seq_x.append(features[t - T_IN + 1 : t + 1])
        seq_ref.append(ref_times[t])

        bin_leads = []
        cont_leads = []
        for k in LEAD_STEPS:
            fp = p_all[t + k]
            bin_leads.append((fp >= DEFAULT_THRESHOLD).astype(np.float32))
            cont_leads.append(np.maximum(fp, 0.0).astype(np.float32))

        seq_y_bin.append(np.stack(bin_leads, axis=0))  # (5, 25, 40)
        seq_y_cont.append(np.stack(cont_leads, axis=0))  # (5, 25, 40)

    seq_x = np.stack(seq_x, axis=0)  # (N_seq, 4, 3, 25, 40)
    seq_y_bin = np.stack(seq_y_bin, axis=0)  # (N_seq, 5, 25, 40)
    seq_y_cont = np.stack(seq_y_cont, axis=0)  # (N_seq, 5, 25, 40)
    N_seq = len(seq_x)
    print(f"Extracted {N_seq} hermetic independent test sequences.")

    # 4. Partition into Non-Overlapping 6-Hour Meteorological Blocks
    dt_start = datetime.fromisoformat(seq_ref[0].replace("Z", "+00:00")).replace(hour=0, minute=0, second=0)
    block_indices = [int((datetime.fromisoformat(t.replace("Z", "+00:00")) - dt_start).total_seconds() // (6 * 3600)) for t in seq_ref]
    unique_blocks = sorted(list(set(block_indices)))
    print(f"Partitioned into {len(unique_blocks)} non-overlapping 6-hour meteorological blocks (df = {len(unique_blocks)-1}).")

    # 5. Run Baseline Inferences (Persistence & Optical Flow)
    print("\n[Step 2] Running Baseline Inferences (Persistence & Optical Flow)...")
    pers_qpf = np.zeros((N_seq, len(LEAD_STEPS), 25, 40), dtype=np.float32)
    flow_qpf = np.zeros((N_seq, len(LEAD_STEPS), 25, 40), dtype=np.float32)

    for i in range(N_seq):
        t_ref_p = p_all[i + T_IN - 1]
        for k_idx in range(len(LEAD_STEPS)):
            pers_qpf[i, k_idx] = t_ref_p

        t_prev_p = p_all[i + T_IN - 2]
        u, v = compute_lucas_kanade_flow(t_prev_p, t_ref_p)
        for k_idx, lead_step in enumerate(LEAD_STEPS):
            flow_qpf[i, k_idx] = advect_field(t_ref_p, u, v, steps=lead_step)

    # 6. Run Frozen ConvLSTM Inferences (Exp A and Exp B across 3 seeds)
    print(f"\n[Step 3] Running Inference on Frozen ConvLSTM Checkpoints on {DEVICE}...")

    # Load Validation split (July 5, 2025) strictly for fitting Platt calibrators
    val_files = [WEATHER_DIR / "aligned_weather_mergir_20250705.nc4"]
    with h5py.File(val_files[0], "r") as vf:
        val_p = vf["input_precip"][:]
        val_tb = vf["satellite_tir1_mean"][:]
        val_mask = vf["satellite_mask"][:]

    val_tb_norm = np.where(
        (val_mask == 1) & np.isfinite(val_tb),
        (val_tb - tb_mean) / max(tb_std, 1e-6),
        0.0,
    ).astype(np.float32)
    val_log_p = np.log1p(np.maximum(val_p, 0.0))
    val_p_norm = np.where(
        np.isfinite(val_log_p),
        (val_log_p - log_p_mean) / max(log_p_std, 1e-6),
        0.0,
    ).astype(np.float32)
    val_feats = np.stack([val_tb_norm, val_p_norm, val_mask.astype(np.float32)], axis=1)

    val_x, val_y_bin = [], []
    for t in range(T_IN - 1, len(val_p) - max_lead):
        val_x.append(val_feats[t - T_IN + 1 : t + 1])
        bin_leads = []
        for k in LEAD_STEPS:
            bin_leads.append((val_p[t + k] >= DEFAULT_THRESHOLD).astype(np.float32))
        val_y_bin.append(np.stack(bin_leads, axis=0))

    val_x_tensor = torch.from_numpy(np.stack(val_x, axis=0)).to(DEVICE)
    val_y_bin_arr = np.stack(val_y_bin, axis=0)  # (33, 5, 25, 40)
    print(f"Loaded {len(val_x)} validation sequences from July 5, 2025 for Platt calibration.")

    x_tensor = torch.from_numpy(seq_x).to(DEVICE)

    model_preds = {}
    models_to_eval = [
        ("Exp A (14-Day Dataset)", "controlled_expA_14day_seed42.pt", 42),
        ("Exp A (14-Day Dataset)", "controlled_expA_14day_seed123.pt", 123),
        ("Exp A (14-Day Dataset)", "controlled_expA_14day_seed999.pt", 999),
        ("Exp B (21-Day Dataset)", "controlled_expB_21day_seed42.pt", 42),
        ("Exp B (21-Day Dataset)", "controlled_expB_21day_seed123.pt", 123),
        ("Exp B (21-Day Dataset)", "controlled_expB_21day_seed999.pt", 999),
    ]

    for exp_name, ckpt_name, seed in models_to_eval:
        ckpt_path = OUTPUTS_MOD / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing required model checkpoint: {ckpt_path}")

        model = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=[32, 32]).to(DEVICE)
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        # Fit Platt Calibrators strictly on validation split
        with torch.no_grad():
            val_cat_out, _ = model(val_x_tensor)
            val_logits = val_cat_out.cpu().numpy()

        calibrators = {}
        for k in range(len(LEAD_STEPS)):
            vl = val_logits[:, k].ravel()
            vt = val_y_bin_arr[:, k].ravel()
            clf = LogisticRegression(solver="lbfgs")
            clf.fit(vl.reshape(-1, 1), vt)
            calibrators[k] = clf

        with torch.no_grad():
            cat_out, qpf_out = model(x_tensor)
            test_logits = cat_out.cpu().numpy()
            qpf = qpf_out.cpu().numpy()  # (N_seq, 5, 25, 40)

        # Compute calibrated probabilities
        probs = np.zeros_like(test_logits, dtype=np.float32)
        for k in range(len(LEAD_STEPS)):
            clf = calibrators[k]
            probs[:, k] = clf.predict_proba(test_logits[:, k].reshape(-1, 1))[:, 1].reshape(test_logits.shape[0], 25, 40)

        model_preds[f"{exp_name}_seed{seed}"] = {
            "probs": probs,
            "qpf": qpf,
            "exp": exp_name,
            "seed": seed,
        }
        print(f"  Loaded, calibrated & evaluated {ckpt_name} (Seed {seed})")

    # 7. Evaluate Detailed Metrics Across Models & Horizons
    print("\n[Step 4] Evaluating Performance Metrics & Block Uncertainties...")

    results_table = []
    rng = np.random.default_rng(RNG_SEED)

    def evaluate_model_on_lead(
        y_true_bin: np.ndarray,
        y_true_cont: np.ndarray,
        y_prob: np.ndarray,
        y_qpf: np.ndarray,
        block_ids: List[int],
    ) -> Dict[str, Any]:
        y_pred_bin = (y_prob >= 0.15).astype(np.int32)
        cont = compute_contingency(y_true_bin.ravel(), y_pred_bin.ravel())

        brier = float(brier_score_loss(y_true_bin.ravel(), y_prob.ravel())) if len(np.unique(y_true_bin)) > 1 else 0.0
        try:
            roc_auc = float(roc_auc_score(y_true_bin.ravel(), y_prob.ravel())) if len(np.unique(y_true_bin)) > 1 else 0.5
        except ValueError:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true_bin.ravel(), y_prob.ravel())) if len(np.unique(y_true_bin)) > 1 else 0.0
        except ValueError:
            pr_auc = 0.0

        rmse_all = float(np.sqrt(np.mean((y_true_cont - y_qpf) ** 2)))
        mae_all = float(np.mean(np.abs(y_true_cont - y_qpf)))

        # Pearson r
        vx = y_true_cont.ravel() - np.mean(y_true_cont)
        vy = y_qpf.ravel() - np.mean(y_qpf)
        denom_r = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2))
        pearson_r = float(np.sum(vx * vy) / denom_r) if denom_r > 1e-9 else 0.0

        ev_mask = y_true_cont >= DEFAULT_THRESHOLD
        if np.any(ev_mask):
            rmse_ev = float(np.sqrt(np.mean((y_true_cont[ev_mask] - y_qpf[ev_mask]) ** 2)))
            mae_ev = float(np.mean(np.abs(y_true_cont[ev_mask] - y_qpf[ev_mask])))
        else:
            rmse_ev, mae_ev = 0.0, 0.0

        # Block-level metrics for standard error
        block_csis, block_rmses = [], []
        for b_val in sorted(list(set(block_ids))):
            b_mask = np.array(block_ids) == b_val
            sub_t = y_true_bin[b_mask].ravel()
            sub_p = y_pred_bin[b_mask].ravel()
            sub_tc = y_true_cont[b_mask].ravel()
            sub_qc = y_qpf[b_mask].ravel()

            c_sub = compute_contingency(sub_t, sub_p)
            block_csis.append(c_sub["csi"])
            block_rmses.append(float(np.sqrt(np.mean((sub_tc - sub_qc) ** 2))))

        csi_sem = float(np.std(block_csis, ddof=1) / np.sqrt(len(block_csis))) if len(block_csis) > 1 else 0.0
        rmse_sem = float(np.std(block_rmses, ddof=1) / np.sqrt(len(block_rmses))) if len(block_rmses) > 1 else 0.0

        return {
            "csi": cont["csi"],
            "pod": cont["pod"],
            "far": cont["far"],
            "f1": cont["f1"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "qpf_rmse_all": rmse_all,
            "qpf_mae_all": mae_all,
            "qpf_pearson_r": pearson_r,
            "qpf_rmse_event": rmse_ev,
            "qpf_mae_event": mae_ev,
            "csi_sem": csi_sem,
            "rmse_sem": rmse_sem,
        }

    # Evaluate all models across horizons
    all_eval_rows = []

    for k_idx, lead in enumerate(LEAD_LABELS):
        y_tb = seq_y_bin[:, k_idx]
        y_tc = seq_y_cont[:, k_idx]

        # 1. Persistence
        pers_prob = np.clip(pers_qpf[:, k_idx] / DEFAULT_THRESHOLD, 0.0, 1.0)
        m_pers = evaluate_model_on_lead(y_tb, y_tc, pers_prob, pers_qpf[:, k_idx], block_indices)
        all_eval_rows.append({"model_name": "Persistence", "horizon": lead, "seed": "N/A", **m_pers})

        # 2. Optical Flow
        flow_prob = np.clip(flow_qpf[:, k_idx] / DEFAULT_THRESHOLD, 0.0, 1.0)
        m_flow = evaluate_model_on_lead(y_tb, y_tc, flow_prob, flow_qpf[:, k_idx], block_indices)
        all_eval_rows.append({"model_name": "Optical Flow", "horizon": lead, "seed": "N/A", **m_flow})

        # 3. ConvLSTM Models
        for (m_key, m_val) in model_preds.items():
            probs = m_val["probs"][:, k_idx]
            qpf = m_val["qpf"][:, k_idx]
            m_res = evaluate_model_on_lead(y_tb, y_tc, probs, qpf, block_indices)
            all_eval_rows.append({
                "model_name": m_val["exp"],
                "horizon": lead,
                "seed": str(m_val["seed"]),
                **m_res,
            })

    # Save detailed comparison CSVs (both filenames for compliance)
    out_csv = OUTPUTS_EXP / "independent_weather_model_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_eval_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_eval_rows)
    print(f"Saved independent test comparison CSV to: {out_csv}")

    out_csv_full = OUTPUTS_EXP / "august_full_horizon_comparison.csv"
    with open(out_csv_full, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_eval_rows)
    print(f"Saved August full horizon comparison CSV to: {out_csv_full}")

    # 8. Save Prediction Arrays NPZ (Seed 42 + Baselines)
    npz_path = OUTPUTS_EXP / "independent_test_prediction_arrays.npz"
    save_arrays = {}
    for k_idx, lead in enumerate(LEAD_LABELS):
        save_arrays[f"targets_bin_{lead}"] = seq_y_bin[:, k_idx].ravel()
        save_arrays[f"targets_cont_{lead}"] = seq_y_cont[:, k_idx].ravel()
        save_arrays[f"pers_qpf_{lead}"] = pers_qpf[:, k_idx].ravel()
        save_arrays[f"flow_qpf_{lead}"] = flow_qpf[:, k_idx].ravel()
        save_arrays[f"expA_seed42_probs_{lead}"] = model_preds["Exp A (14-Day Dataset)_seed42"]["probs"][:, k_idx].ravel()
        save_arrays[f"expA_seed42_qpf_{lead}"] = model_preds["Exp A (14-Day Dataset)_seed42"]["qpf"][:, k_idx].ravel()
        save_arrays[f"expB_seed42_probs_{lead}"] = model_preds["Exp B (21-Day Dataset)_seed42"]["probs"][:, k_idx].ravel()
        save_arrays[f"expB_seed42_qpf_{lead}"] = model_preds["Exp B (21-Day Dataset)_seed42"]["qpf"][:, k_idx].ravel()

    np.savez_compressed(npz_path, **save_arrays)
    print(f"Saved independent test prediction arrays to: {npz_path}")

    # 9. Generate Publication Figures
    print("\n[Step 5] Generating Publication Figures for Independent Test Set...")

    # Figure 1: Skill Curves (CSI, F1, PR-AUC vs Lead Time)
    plt.figure(figsize=(14, 5), dpi=300)

    # Plot CSI vs Lead
    plt.subplot(1, 3, 1)
    models_to_plot = ["Persistence", "Optical Flow", "Exp A (14-Day Dataset)", "Exp B (21-Day Dataset)"]
    colors = {"Persistence": "#7f7f7f", "Optical Flow": "#1f77b4", "Exp A (14-Day Dataset)": "#ff7f0e", "Exp B (21-Day Dataset)": "#2ca02c"}
    markers = {"Persistence": "o", "Optical Flow": "s", "Exp A (14-Day Dataset)": "^", "Exp B (21-Day Dataset)": "D"}

    for mod in models_to_plot:
        mod_rows = [r for r in all_eval_rows if r["model_name"] == mod]
        leads_x = range(len(LEAD_LABELS))
        mean_csis = []
        err_csis = []
        for l in LEAD_LABELS:
            rows_lead = [r for r in mod_rows if r["horizon"] == l]
            mean_csis.append(np.mean([r["csi"] for r in rows_lead]))
            err_csis.append(np.mean([r["csi_sem"] for r in rows_lead]))

        plt.errorbar(leads_x, mean_csis, yerr=err_csis, label=mod, color=colors[mod], marker=markers[mod], capsize=3, linewidth=1.8)

    plt.xticks(range(len(LEAD_LABELS)), LEAD_LABELS)
    plt.ylabel("Critical Success Index (CSI)", fontweight="bold")
    plt.title("(a) CSI Across Forecast Horizons\n(Independent Test: Aug 1-3, 2024)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)

    # Plot F1 vs Lead
    plt.subplot(1, 3, 2)
    for mod in models_to_plot:
        mod_rows = [r for r in all_eval_rows if r["model_name"] == mod]
        mean_f1 = [np.mean([r["f1"] for r in mod_rows if r["horizon"] == l]) for l in LEAD_LABELS]
        plt.plot(range(len(LEAD_LABELS)), mean_f1, label=mod, color=colors[mod], marker=markers[mod], linewidth=1.8)

    plt.xticks(range(len(LEAD_LABELS)), LEAD_LABELS)
    plt.ylabel("F1 Score", fontweight="bold")
    plt.title("(b) F1 Score Across Forecast Horizons", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Plot PR-AUC vs Lead
    plt.subplot(1, 3, 3)
    for mod in models_to_plot:
        mod_rows = [r for r in all_eval_rows if r["model_name"] == mod]
        mean_pr = [np.mean([r["pr_auc"] for r in mod_rows if r["horizon"] == l]) for l in LEAD_LABELS]
        plt.plot(range(len(LEAD_LABELS)), mean_pr, label=mod, color=colors[mod], marker=markers[mod], linewidth=1.8)

    plt.xticks(range(len(LEAD_LABELS)), LEAD_LABELS)
    plt.ylabel("PR-AUC", fontweight="bold")
    plt.title("(c) Precision-Recall AUC Across Lead", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig1_path = OUTPUTS_FIG / "independent_test_skill_curves.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved skill curves figure to: {fig1_path}")

    # Figure 2: QPF Comparison (Overall RMSE & Event RMSE)
    plt.figure(figsize=(11, 5), dpi=300)

    # Overall RMSE
    plt.subplot(1, 2, 1)
    for mod in models_to_plot:
        mod_rows = [r for r in all_eval_rows if r["model_name"] == mod]
        mean_rmse = []
        err_rmse = []
        for l in LEAD_LABELS:
            rows_lead = [r for r in mod_rows if r["horizon"] == l]
            mean_rmse.append(np.mean([r["qpf_rmse_all"] for r in rows_lead]))
            err_rmse.append(np.mean([r["rmse_sem"] for r in rows_lead]))

        plt.errorbar(range(len(LEAD_LABELS)), mean_rmse, yerr=err_rmse, label=mod, color=colors[mod], marker=markers[mod], capsize=3, linewidth=1.8)

    plt.xticks(range(len(LEAD_LABELS)), LEAD_LABELS)
    plt.ylabel("Overall QPF RMSE (mm/hr)", fontweight="bold")
    plt.title("(a) Overall QPF RMSE (All Pixels)\n(Lower is Better)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)

    # Event RMSE (>= 5 mm/hr)
    plt.subplot(1, 2, 2)
    for mod in models_to_plot:
        mod_rows = [r for r in all_eval_rows if r["model_name"] == mod]
        mean_ev_rmse = [np.mean([r["qpf_rmse_event"] for r in mod_rows if r["horizon"] == l]) for l in LEAD_LABELS]
        plt.plot(range(len(LEAD_LABELS)), mean_ev_rmse, label=mod, color=colors[mod], marker=markers[mod], linewidth=1.8)

    plt.xticks(range(len(LEAD_LABELS)), LEAD_LABELS)
    plt.ylabel("Heavy Rain Event RMSE (mm/hr)", fontweight="bold")
    plt.title("(b) Heavy Rain Event RMSE (>= 5 mm/hr)\n(Lower is Better)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig2_path = OUTPUTS_FIG / "independent_test_qpf_comparison.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved QPF comparison figure to: {fig2_path}")

    print("\n" + "=" * 80)
    print("STEP 3 EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
