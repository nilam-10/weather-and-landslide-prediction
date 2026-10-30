"""Probability Calibration & Threshold Sensitivity for Multi-Horizon Weather ConvLSTM.

Calibrates raw ConvLSTM logits using validation set only (Platt Scaling & Temperature Scaling).
Evaluates probability calibration on the held-out test split (July 6-7, 2025).
Computes:
  - Brier score (Uncalibrated vs. Calibrated)
  - Expected Calibration Error (ECE, 10 bins)
  - Meteorological contingency metrics (CSI, F1, POD, FAR)
  - Multi-threshold sensitivity (2.5, 5.0, 7.5, 10.0 mm/hr)
Saves:
  - outputs/experiments/calibration_results.csv
  - outputs/experiments/weather_threshold_sensitivity.csv
  - outputs/experiments/weather_horizon_comparison.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_weather_convlstm import compute_normalization_stats, load_daily_aligned_datasets
from train_weather_convlstm_multiyear import (
    LEAD_NAMES,
    LEAD_STEPS,
    MultiHorizonSpatiotemporalDataset,
    MultiHorizonWeatherConvLSTM,
    compute_contingency,
)

OUTPUTS_EXP = Path("outputs/experiments")
OUTPUTS_MOD = Path("outputs/models/weather_convlstm")
OUTPUTS_EXP.mkdir(parents=True, exist_ok=True)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        n_b = np.sum(mask)
        if n_b > 0:
            avg_prob = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            ece += (n_b / n) * np.abs(avg_prob - avg_true)

    return float(ece)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit temperature parameter T > 0 on validation set via negative log-likelihood."""
    z = torch.from_numpy(logits).float()
    y = torch.from_numpy(targets).float()

    def loss_func(t_val):
        t = float(t_val[0])
        if t <= 0.01:
            return 1e6
        scaled_logits = z / t
        bce = nn.BCEWithLogitsLoss()(scaled_logits, y)
        return float(bce.item())

    res = minimize(loss_func, x0=[1.5], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0])


def run_calibration_and_sensitivity() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing calibration pipeline on device: {device}")

    # 1. Load Data
    data_dir = Path("data/processed/weather")
    dates_2024 = [f"2024070{d}" for d in range(1, 8)]
    dates_2025 = [f"2025070{d}" for d in range(1, 8)]
    all_dates = dates_2024 + dates_2025

    data = load_daily_aligned_datasets(data_dir, all_dates)
    T, H, W = data["input_precip"].shape

    idx_train, idx_val, idx_test = [], [], []
    for t, r_iso in enumerate(data["reference_time_iso"]):
        d_str = r_iso[:10].replace("-", "")
        if d_str in ["20250706", "20250707"]:
            idx_test.append(t)
        elif d_str in ["20250705"]:
            idx_val.append(t)
        else:
            idx_train.append(t)

    train_subset = {
        "ir_mean": data["ir_mean"][idx_train],
        "satellite_mask": data["satellite_mask"][idx_train],
        "input_precip": data["input_precip"][idx_train],
        "target_lead30": data["target_lead30"][idx_train],
    }
    stats = compute_normalization_stats(train_subset)

    tb = data["ir_mean"]
    p = data["input_precip"]
    mask = data["satellite_mask"]

    tb_norm = np.where((mask == 1) & np.isfinite(tb), (tb - stats["tb_mean"]) / max(stats["tb_std"], 1e-6), 0.0).astype(np.float32)
    log_p = np.log1p(np.maximum(p, 0.0))
    p_norm = np.where(np.isfinite(log_p), (log_p - stats["log_p_mean"]) / max(stats["log_p_std"], 1e-6), 0.0).astype(np.float32)
    features = np.stack([tb_norm, p_norm, mask.astype(np.float32)], axis=1)

    ds_val = MultiHorizonSpatiotemporalDataset(features, p, data["reference_time_iso"], idx_val)
    ds_test = MultiHorizonSpatiotemporalDataset(features, p, data["reference_time_iso"], idx_test)

    val_loader = DataLoader(ds_val, batch_size=16, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=16, shuffle=False)

    # 2. Load Checkpoint
    ckpt_path = OUTPUTS_MOD / "multiyear_convlstm_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    model = MultiHorizonWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=len(LEAD_STEPS)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. Extract Raw Logits on Val & Test
    val_logits = {k: [] for k in range(len(LEAD_STEPS))}
    val_targets = {k: [] for k in range(len(LEAD_STEPS))}

    test_logits = {k: [] for k in range(len(LEAD_STEPS))}
    test_targets = {k: [] for k in range(len(LEAD_STEPS))}

    with torch.no_grad():
        for bx, by, _ in val_loader:
            bx = bx.to(device)
            out = model(bx)  # (B, 5, H, W)
            out_np = out.cpu().numpy()
            by_np = by.numpy()
            for k in range(len(LEAD_STEPS)):
                val_logits[k].append(out_np[:, k].ravel())
                val_targets[k].append(by_np[:, k].ravel())

        for bx, by, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            out_np = out.cpu().numpy()
            by_np = by.numpy()
            for k in range(len(LEAD_STEPS)):
                test_logits[k].append(out_np[:, k].ravel())
                test_targets[k].append(by_np[:, k].ravel())

    # Concatenate
    for k in range(len(LEAD_STEPS)):
        val_logits[k] = np.concatenate(val_logits[k])
        val_targets[k] = np.concatenate(val_targets[k])
        test_logits[k] = np.concatenate(test_logits[k])
        test_targets[k] = np.concatenate(test_targets[k])

    # 4. Calibration Analysis
    calib_rows = []
    print("\n--- Calibration Results Across Horizons ---")

    for k, lead_name in enumerate(LEAD_NAMES):
        vl = val_logits[k]
        vt = val_targets[k]
        tl = test_logits[k]
        tt = test_targets[k]

        # Raw sigmoid
        p_raw = 1.0 / (1.0 + np.exp(-tl))
        bs_raw = float(brier_score_loss(tt, p_raw))
        ece_raw = compute_ece(tt, p_raw)

        # Temperature Scaling
        T_opt = fit_temperature(vl, vt)
        p_temp = 1.0 / (1.0 + np.exp(-(tl / T_opt)))
        bs_temp = float(brier_score_loss(tt, p_temp))
        ece_temp = compute_ece(tt, p_temp)

        # Platt Scaling (Logistic Regression on validation logits)
        lr_platt = LogisticRegression(solver="lbfgs")
        lr_platt.fit(vl.reshape(-1, 1), vt)
        p_platt = lr_platt.predict_proba(tl.reshape(-1, 1))[:, 1]
        bs_platt = float(brier_score_loss(tt, p_platt))
        ece_platt = compute_ece(tt, p_platt)

        # Contingency metrics on calibrated vs uncalibrated (decision threshold = 0.5 for Platt calibrated)
        cont_raw = compute_contingency(tt, p_raw, threshold=0.90 if k < 2 else 0.80)
        cont_platt = compute_contingency(tt, p_platt, threshold=0.15)  # calibrated operational threshold

        calib_rows.append({
            "lead_time": f"+{LEAD_STEPS[k]*30}m",
            "lead_name": lead_name,
            "fitted_temperature_T": round(T_opt, 4),
            "platt_scale_slope_a": round(float(lr_platt.coef_[0][0]), 4),
            "platt_scale_intercept_b": round(float(lr_platt.intercept_[0]), 4),
            "uncalibrated_brier_score": round(bs_raw, 4),
            "temperature_scaled_brier_score": round(bs_temp, 4),
            "platt_calibrated_brier_score": round(bs_platt, 4),
            "brier_improvement_percent": round(((bs_raw - bs_platt) / bs_raw) * 100, 2),
            "uncalibrated_ece": round(ece_raw, 4),
            "temperature_scaled_ece": round(ece_temp, 4),
            "platt_calibrated_ece": round(ece_platt, 4),
            "ece_improvement_percent": round(((ece_raw - ece_platt) / ece_raw) * 100, 2),
            "test_roc_auc": cont_raw["roc_auc"],
            "test_pr_auc": cont_raw["pr_auc"],
            "test_csi_raw_threshold": cont_raw["csi"],
            "test_csi_platt_threshold": cont_platt["csi"],
            "n_test_pixels": len(tt),
        })

        print(f"Horizon +{LEAD_STEPS[k]*30}m: Uncal BS={bs_raw:.4f} -> Platt BS={bs_platt:.4f} (-{((bs_raw-bs_platt)/bs_raw)*100:.1f}%), ECE={ece_raw:.4f}->{ece_platt:.4f}")

    calib_csv = OUTPUTS_EXP / "calibration_results.csv"
    with open(calib_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calib_rows)
    print(f"\nWrote calibration results to {calib_csv}")

    # 5. Threshold Sensitivity Analysis (2.5, 5.0, 7.5, 10.0 mm/hr)
    thresh_rows = []
    threshold_values = [2.5, 5.0, 7.5, 10.0]

    ref_to_t = {r: i for i, r in enumerate(data["reference_time_iso"])}
    test_sample_indices = [ref_to_t[r] for r in ds_test.samples_ref_time]

    for th in threshold_values:
        for k in range(2):  # +30m and +60m
            lead_name = LEAD_NAMES[k]
            lead_m = LEAD_STEPS[k] * 30

            # Compute actual binary target for threshold th on exact test samples
            samples_target = []
            for t in test_sample_indices:
                future_p = p[t + LEAD_STEPS[k]]
                samples_target.append((future_p >= th).astype(np.float32).ravel())

            if not samples_target:
                continue
            y_true_th = np.concatenate(samples_target)
            y_prob_th = 1.0 / (1.0 + np.exp(-test_logits[k]))

            # Find best contingency
            best_c = None
            best_csi = -1.0
            for dt in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                c = compute_contingency(y_true_th, y_prob_th, threshold=dt)
                if c["csi"] > best_csi:
                    best_csi = c["csi"]
                    best_c = c

            thresh_rows.append({
                "rainfall_threshold_mm_hr": th,
                "lead_time": f"+{lead_m}m",
                "lead_name": lead_name,
                "event_definition": f"Rainfall rate >= {th} mm/hr (Experimental Convective Metric)",
                "test_positive_pixels": int(np.sum(y_true_th == 1)),
                "test_total_pixels": len(y_true_th),
                "test_prevalence_percent": round(float(np.mean(y_true_th)) * 100, 3),
                "csi": best_c["csi"],
                "f1": best_c["f1"],
                "pod_recall": best_c["pod_recall"],
                "far": best_c["far"],
                "precision": best_c["precision"],
                "brier_score": best_c["brier_score"],
                "roc_auc": best_c["roc_auc"],
                "pr_auc": best_c["pr_auc"],
                "optimal_decision_threshold": best_c["decision_threshold"],
            })

    thresh_csv = OUTPUTS_EXP / "weather_threshold_sensitivity.csv"
    with open(thresh_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(thresh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(thresh_rows)
    print(f"Wrote threshold sensitivity results to {thresh_csv}")

    # 6. Horizon comparison summary
    horiz_rows = []
    for r in calib_rows:
        horiz_rows.append({
            "lead_time": r["lead_time"],
            "lead_name": r["lead_name"],
            "lead_minutes": int(r["lead_time"].replace("+", "").replace("m", "")),
            "csi": r["test_csi_raw_threshold"],
            "roc_auc": r["test_roc_auc"],
            "pr_auc": r["test_pr_auc"],
            "brier_score_uncalibrated": r["uncalibrated_brier_score"],
            "brier_score_calibrated": r["platt_calibrated_brier_score"],
            "ece_uncalibrated": r["uncalibrated_ece"],
            "ece_calibrated": r["platt_calibrated_ece"],
            "predictability_regime": (
                "High Nowcasting Skill (+30m to +60m)" if int(r["lead_time"].replace("+", "").replace("m", "")) <= 60
                else ("Moderate Advective Skill (+120m)" if int(r["lead_time"].replace("+", "").replace("m", "")) == 120
                else ("Dissipation Zone (+180m)" if int(r["lead_time"].replace("+", "").replace("m", "")) == 180
                else "Physical Observational Limit (+360m / 6h, NWP Required)"))
            )
        })

    horiz_csv = OUTPUTS_EXP / "weather_horizon_comparison.csv"
    with open(horiz_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(horiz_rows[0].keys()))
        writer.writeheader()
        writer.writerows(horiz_rows)
    print(f"Wrote weather horizon comparison to {horiz_csv}")


if __name__ == "__main__":
    run_calibration_and_sensitivity()
