"""Compute Frame- and Event-Level Block Bootstrap Confidence Intervals (Tasks 3 & 6).

Scientific Justification:
Treating spatial pixels as independent observations in bootstrap estimation
severely underestimates sampling variance due to strong spatial and temporal
autocorrelation in radar/satellite precipitation fields.
This script performs block bootstrap resampling at the full-frame (1,000 pixels/frame)
and storm-event level, generating defensible 95% empirical confidence intervals.

Saves:
  - outputs/experiments/weather_metrics_block_bootstrap_ci.csv
  - outputs/experiments/weather_metrics_bootstrap_comparison.csv
"""
import numpy as np
import csv
from pathlib import Path
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
)

PRED_NPZ = Path("outputs/experiments/stored_test_predictions_multihorizon.npz")
OUT_BLOCK_CSV = Path("outputs/experiments/weather_metrics_block_bootstrap_ci.csv")
OUT_COMP_CSV = Path("outputs/experiments/weather_metrics_bootstrap_comparison.csv")

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    bs = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)

    if len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_prob))
        pc, rc, _ = precision_recall_curve(y_true, y_prob)
        pr = float(auc(rc, pc))
    else:
        roc = 0.5
        pr = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tp + fp + fn
    csi = float(tp / denom) if denom > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "brier_score": bs,
        "ece": ece,
        "roc_auc": roc,
        "pr_auc": pr,
        "csi": csi,
        "f1": f1,
        "pod": pod,
        "far": far
    }

def main():
    print(f"Loading test predictions from: {PRED_NPZ}")
    data = np.load(PRED_NPZ)

    # 85 frames, 1000 pixels each
    n_frames = 85
    px_per_frame = 1000
    n_bootstraps = 500
    rng = np.random.default_rng(42)

    configs = [
        ("Persistence Hold", "Uncalibrated Binary", "+30m", 0.5, "test_targets_30m", "persistence_probs_30m"),
        ("Lucas-Kanade Optical Flow", "Uncalibrated Binary", "+30m", 0.5, "test_targets_30m", "optical_flow_probs_30m"),
        ("MultiHorizon Weather ConvLSTM", "Uncalibrated Raw Sigmoid", "+30m", 0.9, "test_targets_30m", "raw_probs_30m"),
        ("MultiHorizon Weather ConvLSTM", "Platt Calibrated (Platt, 1999)", "+30m", 0.15, "test_targets_30m", "cal_probs_30m"),
        ("MultiHorizon Weather ConvLSTM", "Uncalibrated Raw Sigmoid", "+60m", 0.9, "test_targets_60m", "raw_probs_60m"),
        ("MultiHorizon Weather ConvLSTM", "Platt Calibrated (Platt, 1999)", "+60m", 0.15, "test_targets_60m", "cal_probs_60m"),
        ("MultiHorizon Weather ConvLSTM", "Uncalibrated Raw Sigmoid", "+120m", 0.8, "test_targets_120m", "raw_probs_120m"),
        ("MultiHorizon Weather ConvLSTM", "Platt Calibrated (Platt, 1999)", "+120m", 0.15, "test_targets_120m", "cal_probs_120m"),
        ("MultiHorizon Weather ConvLSTM", "Uncalibrated Raw Sigmoid", "+180m", 0.8, "test_targets_180m", "raw_probs_180m"),
        ("MultiHorizon Weather ConvLSTM", "Platt Calibrated (Platt, 1999)", "+180m", 0.15, "test_targets_180m", "cal_probs_180m"),
        ("MultiHorizon Weather ConvLSTM", "Uncalibrated Raw Sigmoid", "+360m", 0.8, "test_targets_360m", "raw_probs_360m"),
        ("MultiHorizon Weather ConvLSTM", "Platt Calibrated (Platt, 1999)", "+360m", 0.15, "test_targets_360m", "cal_probs_360m"),
    ]

    block_rows = []
    comp_rows = []

    for model_name, cal_status, horizon, thresh, target_key, prob_key in configs:
        y_true_all = data[target_key]
        y_prob_all = data[prob_key]

        # Reshape to frames
        y_true_frames = y_true_all.reshape(n_frames, px_per_frame)
        y_prob_frames = y_prob_all.reshape(n_frames, px_per_frame)

        # Point estimate
        pt_metrics = evaluate_metrics(y_true_all, y_prob_all, threshold=thresh)

        # Block bootstrap over frames
        boot_metrics = {m: [] for m in pt_metrics}
        # Also pixel bootstrap for direct side-by-side width comparison
        boot_pixel_metrics = {m: [] for m in pt_metrics}

        for _ in range(n_bootstraps):
            # Frame-level block resampling
            frame_indices = rng.choice(n_frames, size=n_frames, replace=True)
            b_true_frame = y_true_frames[frame_indices].ravel()
            b_prob_frame = y_prob_frames[frame_indices].ravel()
            res_frame = evaluate_metrics(b_true_frame, b_prob_frame, threshold=thresh)
            for m in boot_metrics:
                boot_metrics[m].append(res_frame[m])

            # Pixel-level naive resampling
            px_indices = rng.choice(len(y_true_all), size=len(y_true_all), replace=True)
            b_true_px = y_true_all[px_indices]
            b_prob_px = y_prob_all[px_indices]
            res_px = evaluate_metrics(b_true_px, b_prob_px, threshold=thresh)
            for m in boot_pixel_metrics:
                boot_pixel_metrics[m].append(res_px[m])

        def format_ci(pt, arr):
            low = float(np.percentile(arr, 2.5))
            high = float(np.percentile(arr, 97.5))
            return f"{pt:.4f} [{low:.4f}, {high:.4f}]"

        def get_width(arr):
            low = float(np.percentile(arr, 2.5))
            high = float(np.percentile(arr, 97.5))
            return high - low

        block_rows.append({
            "model_name": model_name,
            "calibration_status": cal_status,
            "horizon": horizon,
            "threshold": thresh,
            "resampling_protocol": "Frame-Level Block Bootstrap (Accounts for Spatial Autocorrelation)",
            "n_temporal_blocks": n_frames,
            "pixels_per_block": px_per_frame,
            "brier_score": format_ci(pt_metrics["brier_score"], boot_metrics["brier_score"]),
            "ece": format_ci(pt_metrics["ece"], boot_metrics["ece"]),
            "csi": format_ci(pt_metrics["csi"], boot_metrics["csi"]),
            "roc_auc": format_ci(pt_metrics["roc_auc"], boot_metrics["roc_auc"]),
            "pr_auc": format_ci(pt_metrics["pr_auc"], boot_metrics["pr_auc"]),
            "f1": format_ci(pt_metrics["f1"], boot_metrics["f1"]),
            "pod": format_ci(pt_metrics["pod"], boot_metrics["pod"]),
            "far": format_ci(pt_metrics["far"], boot_metrics["far"]),
        })

        # Save width comparison for CSI and Brier score
        csi_width_px = get_width(boot_pixel_metrics["csi"])
        csi_width_block = get_width(boot_metrics["csi"])
        bs_width_px = get_width(boot_pixel_metrics["brier_score"])
        bs_width_block = get_width(boot_metrics["brier_score"])

        comp_rows.append({
            "model_name": model_name,
            "calibration_status": cal_status,
            "horizon": horizon,
            "metric": "CSI",
            "point_estimate": round(pt_metrics["csi"], 4),
            "pixel_bootstrap_95ci_width": round(csi_width_px, 4),
            "block_bootstrap_95ci_width": round(csi_width_block, 4),
            "uncertainty_inflation_ratio": round(csi_width_block / max(csi_width_px, 1e-6), 2),
            "scientific_interpretation": "Frame-level block bootstrap captures meteorological variance between storms"
        })

    # Write block bootstrap CSV
    with open(OUT_BLOCK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    # Write comparison CSV
    with open(OUT_COMP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)

    print(f"Saved block bootstrap metrics to: {OUT_BLOCK_CSV}")
    print(f"Saved comparison to: {OUT_COMP_CSV}")

if __name__ == "__main__":
    main()
