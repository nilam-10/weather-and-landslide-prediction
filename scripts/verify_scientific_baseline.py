"""Scientific Baseline Verification & Metric Reproduction Suite (Phase 1).

Reproduces exact confusion matrices (TP, TN, FP, FN), raw Accuracy, Balanced Accuracy,
Precision, Recall (POD), Specificity, FAR, F1, CSI, ROC-AUC, PR-AUC, Brier score,
and continuous QPF metrics directly from stored prediction arrays across all models,
horizons, decision thresholds, and evaluation periods.

Ensures:
- Accuracy is computed strictly from (TP + TN) / N, never estimated from F1 or CSI.
- Rainfall thresholds (5.0 mm/hr vs 0.1 mm/hr) are explicitly and correctly labeled.
- Produces outputs/reports/BASELINE_VERIFICATION.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs" / "experiments"
REP_DIR = PROJECT_ROOT / "outputs" / "reports"
MOD_DIR = PROJECT_ROOT / "outputs" / "models"
REP_DIR.mkdir(parents=True, exist_ok=True)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute exact confusion matrix and all derived classification metrics."""
    y_true_b = (y_true >= 0.5).astype(np.int32).ravel()
    y_pred_b = (y_prob >= threshold).astype(np.int32).ravel()

    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall_pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall_pod + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    brier = float(brier_score_loss(y_true_b, y_prob.ravel())) if len(np.unique(y_true_b)) > 1 else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true_b, y_prob.ravel())) if len(np.unique(y_true_b)) > 1 else 0.5
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true_b, y_prob.ravel())) if len(np.unique(y_true_b)) > 1 else 0.0
    except ValueError:
        pr_auc = 0.0

    prevalence = (tp + fn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_samples": total,
        "prevalence": prevalence,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_pod": recall_pod,
        "specificity": specificity,
        "far": far,
        "f1": f1,
        "csi": csi,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }


def compute_continuous_qpf_metrics(
    y_true_cont: np.ndarray,
    y_pred_qpf: np.ndarray,
    event_threshold: float = 5.0,
) -> Dict[str, float]:
    """Compute exact continuous regression metrics: RMSE, MAE, Pearson r, and event-stratified RMSE/MAE."""
    yt = y_true_cont.ravel().astype(np.float64)
    yp = y_pred_qpf.ravel().astype(np.float64)

    rmse_all = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae_all = float(np.mean(np.abs(yt - yp)))

    # Pearson correlation
    v_yt = yt - np.mean(yt)
    v_yp = yp - np.mean(yp)
    denom = np.sqrt(np.sum(v_yt ** 2)) * np.sqrt(np.sum(v_yp ** 2))
    r = float(np.sum(v_yt * v_yp) / denom) if denom > 1e-12 else 0.0

    # Event-stratified (>= event_threshold)
    ev_mask = yt >= event_threshold
    if np.any(ev_mask):
        rmse_ev = float(np.sqrt(np.mean((yt[ev_mask] - yp[ev_mask]) ** 2)))
        mae_ev = float(np.mean(np.abs(yt[ev_mask] - yp[ev_mask])))
    else:
        rmse_ev, mae_ev = 0.0, 0.0

    return {
        "qpf_rmse_all": rmse_all,
        "qpf_mae_all": mae_all,
        "qpf_pearson_r": r,
        "qpf_rmse_event": rmse_ev,
        "qpf_mae_event": mae_ev,
    }


def run_baseline_verification() -> None:
    print("=" * 80)
    print("STARTING SCIENTIFIC BASELINE VERIFICATION & METRIC REPRODUCTION (PHASE 1)")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. August 1–3, 2024 Independent Frozen Test Period (N=129,000 pixels)
    # -------------------------------------------------------------------------
    aug_npz_path = EXP_DIR / "independent_test_prediction_arrays.npz"
    if aug_npz_path.exists():
        print("\n[Dataset 1] Loading August 1-3, 2024 Independent Test Prediction Arrays...")
        aug_data = np.load(aug_npz_path)
        horizons = ["+30m", "+60m", "+120m", "+180m", "+360m"]

        # Models stored: pers, flow, expA_seed42, expB_seed42
        for h in horizons:
            y_true_bin = aug_data[f"targets_bin_{h}"]
            y_true_cont = aug_data[f"targets_cont_{h}"]

            # A. Persistence Baseline
            pers_qpf = aug_data[f"pers_qpf_{h}"]
            pers_prob = np.clip(pers_qpf / 10.0, 0.0, 1.0)
            for d_thresh in [0.15, 0.50]:
                pers_cls = compute_binary_metrics(y_true_bin, pers_prob, threshold=d_thresh)
                pers_reg = compute_continuous_qpf_metrics(y_true_cont, pers_qpf, event_threshold=5.0)
                rows.append({
                    "evaluation_period": "August 1-3, 2024 (Frozen Independent Test)",
                    "model_name": "Persistence Baseline",
                    "model_family": "Zero-Skill Lagged Hold",
                    "forecast_horizon": h,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                    **pers_cls,
                    **pers_reg,
                    "checkpoint_name": "N/A (Deterministic Observation Lag Hold)",
                    "checkpoint_sha256": "N/A",
                    "net_trainable_parameters": 0,
                    "verification_source": "independent_test_prediction_arrays.npz",
                })

            # B. Optical Flow Baseline
            flow_qpf = aug_data[f"flow_qpf_{h}"]
            flow_prob = np.clip(flow_qpf / 10.0, 0.0, 1.0)
            for d_thresh in [0.15, 0.50]:
                flow_cls = compute_binary_metrics(y_true_bin, flow_prob, threshold=d_thresh)
                flow_reg = compute_continuous_qpf_metrics(y_true_cont, flow_qpf, event_threshold=5.0)
                rows.append({
                    "evaluation_period": "August 1-3, 2024 (Frozen Independent Test)",
                    "model_name": "Lucas-Kanade Optical Flow",
                    "model_family": "Kinematic Semi-Lagrangian Advection",
                    "forecast_horizon": h,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                    **flow_cls,
                    **flow_reg,
                    "checkpoint_name": "N/A (SciPy/rainymotion Advection)",
                    "checkpoint_sha256": "N/A",
                    "net_trainable_parameters": 0,
                    "verification_source": "independent_test_prediction_arrays.npz",
                })

            # C. Controlled Exp A (14-Day Dataset, Seed 42)
            expA_prob = aug_data[f"expA_seed42_probs_{h}"]
            expA_qpf = aug_data[f"expA_seed42_qpf_{h}"]
            for d_thresh in [0.15, 0.50]:
                expA_cls = compute_binary_metrics(y_true_bin, expA_prob, threshold=d_thresh)
                expA_reg = compute_continuous_qpf_metrics(y_true_cont, expA_qpf, event_threshold=5.0)
                rows.append({
                    "evaluation_period": "August 1-3, 2024 (Frozen Independent Test)",
                    "model_name": "ConvLSTM Exp A (14-Day Dataset, Seed 42)",
                    "model_family": "MultiTask Spatiotemporal ConvLSTM",
                    "forecast_horizon": h,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                    **expA_cls,
                    **expA_reg,
                    "checkpoint_name": "controlled_expA_14day_seed42.pt",
                    "checkpoint_sha256": "75921b742e2cb890026f10c243e218fde8857810038149352aabd09f1f160288",
                    "net_trainable_parameters": 161034,
                    "verification_source": "independent_test_prediction_arrays.npz",
                })

            # D. Controlled Exp B (21-Day Dataset, Seed 42, Production Dashboard Model)
            expB_prob = aug_data[f"expB_seed42_probs_{h}"]
            expB_qpf = aug_data[f"expB_seed42_qpf_{h}"]
            for d_thresh in [0.15, 0.50]:
                expB_cls = compute_binary_metrics(y_true_bin, expB_prob, threshold=d_thresh)
                expB_reg = compute_continuous_qpf_metrics(y_true_cont, expB_qpf, event_threshold=5.0)
                rows.append({
                    "evaluation_period": "August 1-3, 2024 (Frozen Independent Test)",
                    "model_name": "ConvLSTM Exp B (21-Day Dataset, Seed 42 - Canonical Dashboard)",
                    "model_family": "MultiTask Spatiotemporal ConvLSTM",
                    "forecast_horizon": h,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                    **expB_cls,
                    **expB_reg,
                    "checkpoint_name": "controlled_expB_21day_seed42.pt",
                    "checkpoint_sha256": "c9768f4638e45bfd2e49b8b210aeaeb6f99316698edb2f9fb5468878db40baf6",
                    "net_trainable_parameters": 161034,
                    "verification_source": "independent_test_prediction_arrays.npz",
                })

    # -------------------------------------------------------------------------
    # 2. July 6–7, 2025 Multiyear Pilot Held-Out Test Period (N=85,000 pixels)
    # -------------------------------------------------------------------------
    multi_npz_path = EXP_DIR / "stored_test_predictions_multihorizon.npz"
    if multi_npz_path.exists():
        print("\n[Dataset 2] Loading July 6-7, 2025 Multihorizon Benchmark Predictions...")
        m_data = np.load(multi_npz_path)
        horizons_m = [("30m", "+30m"), ("60m", "+60m"), ("120m", "+120m"), ("180m", "+180m"), ("360m", "+360m")]

        for key_suffix, h_label in horizons_m:
            y_true_bin = m_data[f"test_targets_{key_suffix}"]
            raw_prob = m_data[f"raw_probs_{key_suffix}"]
            cal_prob = m_data[f"cal_probs_{key_suffix}"]

            # Uncalibrated ConvLSTM
            for d_thresh in [0.50, 0.90]:
                uncal_cls = compute_binary_metrics(y_true_bin, raw_prob, threshold=d_thresh)
                rows.append({
                    "evaluation_period": "July 6-7, 2025 (Pilot Held-Out Test)",
                    "model_name": "MultiHorizon Weather ConvLSTM (Uncalibrated)",
                    "model_family": "5-Head Recurrent Spatiotemporal ConvLSTM",
                    "forecast_horizon": h_label,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Standard Default (0.50) / Tuned Raw (0.90)",
                    **uncal_cls,
                    "qpf_rmse_all": np.nan,
                    "qpf_mae_all": np.nan,
                    "qpf_pearson_r": np.nan,
                    "qpf_rmse_event": np.nan,
                    "qpf_mae_event": np.nan,
                    "checkpoint_name": "multiyear_convlstm_model.pt",
                    "checkpoint_sha256": "b48705522e468b88370e147af2ef99a123b0cdf90160bca35d0080f7e307efa7",
                    "net_trainable_parameters": 137669,
                    "verification_source": "stored_test_predictions_multihorizon.npz",
                })

            # Platt Calibrated ConvLSTM
            for d_thresh in [0.15, 0.50]:
                cal_cls = compute_binary_metrics(y_true_bin, cal_prob, threshold=d_thresh)
                rows.append({
                    "evaluation_period": "July 6-7, 2025 (Pilot Held-Out Test)",
                    "model_name": "MultiHorizon Weather ConvLSTM (Platt Calibrated)",
                    "model_family": "5-Head Recurrent Spatiotemporal ConvLSTM",
                    "forecast_horizon": h_label,
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": d_thresh,
                    "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                    **cal_cls,
                    "qpf_rmse_all": np.nan,
                    "qpf_mae_all": np.nan,
                    "qpf_pearson_r": np.nan,
                    "qpf_rmse_event": np.nan,
                    "qpf_mae_event": np.nan,
                    "checkpoint_name": "multiyear_convlstm_model.pt",
                    "checkpoint_sha256": "b48705522e468b88370e147af2ef99a123b0cdf90160bca35d0080f7e307efa7",
                    "net_trainable_parameters": 137669,
                    "verification_source": "stored_test_predictions_multihorizon.npz",
                })

    # -------------------------------------------------------------------------
    # 3. July 6–7, 2025 Controlled 14-Day vs 21-Day Comparison (N=81,000 pixels)
    # -------------------------------------------------------------------------
    ctrl_npz_path = EXP_DIR / "stored_test_predictions_controlled_seeds.npz"
    if ctrl_npz_path.exists():
        print("\n[Dataset 3] Loading Controlled 14-Day vs 21-Day Test Predictions...")
        c_data = np.load(ctrl_npz_path)
        horizons_c = ["+30m", "+60m", "+120m", "+180m", "+360m"]

        for h in horizons_c:
            for exp_prefix, exp_label, ckpt, pcount in [
                ("expA", "Controlled Exp A (14-Day Dataset)", "controlled_expA_14day_seed42.pt", 161034),
                ("expB", "Controlled Exp B (21-Day Dataset)", "controlled_expB_21day_seed42.pt", 161034),
            ]:
                y_bin = c_data[f"{exp_prefix}_targets_bin_{h}"]
                y_cont = c_data[f"{exp_prefix}_targets_cont_{h}"]
                prob = c_data[f"{exp_prefix}_probs_cal_{h}"]
                qpf = c_data[f"{exp_prefix}_qpf_rates_{h}"]

                for d_thresh in [0.15, 0.50]:
                    cls_m = compute_binary_metrics(y_bin, prob, threshold=d_thresh)
                    reg_m = compute_continuous_qpf_metrics(y_cont, qpf, event_threshold=5.0)
                    rows.append({
                        "evaluation_period": "July 6-7, 2025 (Controlled Size Experiment)",
                        "model_name": f"{exp_label} (Seed 42)",
                        "model_family": "MultiTask Spatiotemporal ConvLSTM",
                        "forecast_horizon": h,
                        "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                        "decision_threshold": d_thresh,
                        "threshold_tuning_status": "Validation-Tuned (0.15) / Standard Default (0.50)",
                        **cls_m,
                        **reg_m,
                        "checkpoint_name": ckpt,
                        "checkpoint_sha256": "c9768f4638e45bfd" if "expB" in exp_prefix else "75921b742e2cb890",
                        "net_trainable_parameters": pcount,
                        "verification_source": "stored_test_predictions_controlled_seeds.npz",
                    })

    # -------------------------------------------------------------------------
    # 4. Tabular Pixel-Wise Baselines (+30m, July 6–7, 2025, N=94,697 samples)
    # -------------------------------------------------------------------------
    base_json_path = MOD_DIR / "weather_baseline" / "baseline_metrics.json"
    if base_json_path.exists():
        print("\n[Dataset 4] Loading Tabular Pixel-Wise Baselines from baseline_metrics.json...")
        with open(base_json_path, "r") as f:
            b_json = json.load(f)

        for b_name, b_label, ckpt, params in [
            ("persistence", "Eulerian Lagged Persistence Baseline", "N/A", 0),
            ("logistic_regression", "Tabular Logistic Regression Baseline", "N/A", 4),
            ("pytorch_mlp", "PyTorch 3-Layer Tabular MLP (64-32-1)", "pytorch_baseline_model.pt", 2369),
        ]:
            if b_name in b_json and "test" in b_json[b_name]:
                tdata = b_json[b_name]["test"]
                cm = tdata["confusion_matrix"]
                tp = int(cm["tp"])
                tn = int(cm["tn"])
                fp = int(cm["fp"])
                fn = int(cm["fn"])
                tot = tp + tn + fp + fn
                acc = (tp + tn) / tot if tot > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bal_acc = (rec + spec) / 2.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                rows.append({
                    "evaluation_period": "July 6-7, 2025 (Pilot Held-Out Test)",
                    "model_name": b_label,
                    "model_family": "Tabular Pixel-Wise Baseline",
                    "forecast_horizon": "+30m",
                    "target_event_definition": "Precipitation Rate >= 5.0 mm/hr (IMD Convective Heavy Rain)",
                    "decision_threshold": b_json[b_name].get("tuned_threshold", 0.50),
                    "threshold_tuning_status": "Validation-Tuned Threshold",
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "total_samples": tot,
                    "prevalence": (tp + fn) / tot if tot > 0 else 0.0,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "precision": prec,
                    "recall_pod": rec,
                    "specificity": spec,
                    "far": far,
                    "f1": f1,
                    "csi": csi,
                    "roc_auc": tdata.get("roc_auc", np.nan),
                    "pr_auc": np.nan,
                    "brier_score": np.nan,
                    "qpf_rmse_all": np.nan,
                    "qpf_mae_all": np.nan,
                    "qpf_pearson_r": np.nan,
                    "qpf_rmse_event": np.nan,
                    "qpf_mae_event": np.nan,
                    "checkpoint_name": ckpt,
                    "checkpoint_sha256": "N/A",
                    "net_trainable_parameters": params,
                    "verification_source": "outputs/models/weather_baseline/baseline_metrics.json",
                })

    # -------------------------------------------------------------------------
    # 5. Geomorphic Landslide Susceptibility Models (3-Fold Spatial Block CV)
    # -------------------------------------------------------------------------
    lnd_csv_path = EXP_DIR / "landslide_spatial_cv_final.csv"
    if lnd_csv_path.exists():
        print("\n[Dataset 5] Loading Geomorphic Landslide Susceptibility Models...")
        with open(lnd_csv_path, "r", encoding="utf-8") as f:
            lnd_rows = list(csv.DictReader(f))

        # Model aggregations
        models_lnd = sorted(list(set(r["model"] for r in lnd_rows)))
        for m_name in models_lnd:
            sub_r = [r for r in lnd_rows if r["model"] == m_name]
            auc_vals = [float(r["roc_auc"]) for r in sub_r]
            pr_vals = [float(r["pr_auc"]) for r in sub_r]
            brier_vals = [float(r["brier_score"]) for r in sub_r]
            tot_test = sum(int(r["n_test"]) for r in sub_r)
            tot_pos = sum(int(r["n_pos"]) for r in sub_r)

            rows.append({
                "evaluation_period": "Historical Landslide Inventory (2007-2024 Spatial Block CV)",
                "model_name": f"{m_name} (Symmetric Point Extraction)",
                "model_family": "Static Geomorphic Terrain Susceptibility (90m LSI)",
                "forecast_horizon": "Static (Zero Lead Time)",
                "target_event_definition": "Rainfall-Triggered Landslide Occurrence (NASA GLC/COOLR)",
                "decision_threshold": 0.50,
                "threshold_tuning_status": "Standard Default (0.50)",
                "tp": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "total_samples": tot_test,
                "prevalence": tot_pos / tot_test if tot_test > 0 else 0.0,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "precision": np.nan,
                "recall_pod": np.nan,
                "specificity": np.nan,
                "far": np.nan,
                "f1": np.nan,
                "csi": np.nan,
                "roc_auc": float(np.mean(auc_vals)),
                "pr_auc": float(np.mean(pr_vals)),
                "brier_score": float(np.mean(brier_vals)),
                "qpf_rmse_all": np.nan,
                "qpf_mae_all": np.nan,
                "qpf_pearson_r": np.nan,
                "qpf_rmse_event": np.nan,
                "qpf_mae_event": np.nan,
                "checkpoint_name": "xgboost_lsm.json" if "XGBoost" in m_name else "N/A (Scikit-Learn)",
                "checkpoint_sha256": "N/A",
                "net_trainable_parameters": 50 if "XGBoost" in m_name else 9,
                "verification_source": "outputs/experiments/landslide_spatial_cv_final.csv",
            })

    # Save to CSV
    out_csv = REP_DIR / "BASELINE_VERIFICATION.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # Format floats cleanly
                clean_r = {}
                for k, v in r.items():
                    if isinstance(v, float):
                        clean_r[k] = f"{v:.4f}" if np.isfinite(v) else "N/A"
                    else:
                        clean_r[k] = v
                writer.writerow(clean_r)

        print(f"\nSuccessfully wrote {len(rows)} verified baseline evaluation rows to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_verification()
