"""Optical Flow Kinematic Extrapolation Weather Nowcasting Baseline.

Adapted from rainymotion (Ayzel et al., GMD 2019) and Lucas-Kanade optical flow.
Extrapolates 0.1-degree IMERG precipitation fields across multi-horizon leads:
  +30m (1 step), +60m (2 steps), +120m (4 steps), +180m (6 steps).

Evaluates against Persistence and ground-truth IMERG threshold exceedance (>= 5.0 mm/hr).
Strict anti-leakage guarantee: velocity fields derived strictly from past observations (t-1, t).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

DEFAULT_DATA_DIR = Path("data/processed/weather")
DEFAULT_OUTPUT_DIR = Path("outputs/models/weather_optical_flow")
DEFAULT_THRESHOLD = 5.0  # mm/hr
LEAD_HORIZONS = {
    "lead_30min": 1,
    "lead_60min": 2,
    "lead_120min": 4,
    "lead_180min": 6,
    "lead_360min": 12,
}
DATE_SPLITS = {
    "train": ["20250701", "20250702", "20250703", "20250704"],
    "val": ["20250705"],
    "test": ["20250706", "20250707"],
}


def load_all_aligned_precipitation(
    data_dir: Path,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """Load continuous IMERG precipitation sequences across all 7 pilot days."""
    p_list = []
    time_iso_list = []
    ref_iso_list = []
    lats = None
    lons = None
    split_indices: Dict[str, Tuple[int, int]] = {}

    current_idx = 0
    for split_name, dates in DATE_SPLITS.items():
        start_idx = current_idx
        for d in dates:
            fpath = data_dir / f"aligned_weather_mergir_{d}.nc4"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required aligned dataset: {fpath}")

            with h5py.File(fpath, "r") as f:
                if lats is None:
                    lats = f["lat"][:]
                    lons = f["lon"][:]

                p_data = f["input_precip"][:]
                p_list.append(p_data)

                nom = [
                    s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                    for s in f["nominal_slot_iso"][:]
                ]
                ref = [
                    s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                    for s in f["reference_time_iso"][:]
                ]
                time_iso_list.extend(nom)
                ref_iso_list.extend(ref)
                current_idx += len(p_data)
        split_indices[split_name] = (start_idx, current_idx - 1)

    p_all = np.concatenate(p_list, axis=0)  # shape: (336, 25, 40)
    return p_all, time_iso_list, ref_iso_list, lats, lons, split_indices


def compute_lucas_kanade_flow(
    i1: np.ndarray,
    i2: np.ndarray,
    sigma: float = 0.8,
    win_size: int = 5,
    reg: float = 1e-3,
    max_vel: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dense optical flow displacement field between consecutive frames.
    
    Args:
        i1: Precipitation frame at t-1
        i2: Precipitation frame at t
        sigma: Gaussian smoothing scale
        win_size: Integration window size for structure tensor
        reg: Regularization parameter for matrix inversion
        max_vel: Maximum physically allowed advection speed (grid units per step)
        
    Returns:
        (u, v): Horizontal and vertical velocity fields in grid units per time step.
    """
    s1 = ndi.gaussian_filter(np.nan_to_num(i1, nan=0.0), sigma=sigma)
    s2 = ndi.gaussian_filter(np.nan_to_num(i2, nan=0.0), sigma=sigma)
    it = s2 - s1
    iy, ix = np.gradient(s2)

    # Window integration for structure tensor components
    a11 = ndi.uniform_filter(ix * ix, size=win_size)
    a12 = ndi.uniform_filter(ix * iy, size=win_size)
    a22 = ndi.uniform_filter(iy * iy, size=win_size)
    b1 = ndi.uniform_filter(-ix * it, size=win_size)
    b2 = ndi.uniform_filter(-iy * it, size=win_size)

    det = a11 * a22 - a12 * a12 + reg
    u = (a22 * b1 - a12 * b2) / det
    v = (-a12 * b1 + a11 * b2) / det

    # Spatial smoothing and clipping of velocity field
    u = np.clip(ndi.gaussian_filter(u, sigma=1.0), -max_vel, max_vel)
    v = np.clip(ndi.gaussian_filter(v, sigma=1.0), -max_vel, max_vel)
    return u.astype(np.float32), v.astype(np.float32)


def semi_lagrangian_extrapolate(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lead_steps: int,
) -> np.ndarray:
    """Extrapolate 2D precipitation field forward in time via semi-Lagrangian backward warping.
    
    Args:
        img: Precipitation frame at t
        u, v: Velocity fields at t
        lead_steps: Number of forward time steps (k)
        
    Returns:
        Extrapolated precipitation field at t + k.
    """
    h, w = img.shape
    y, x = np.indices((h, w), dtype=np.float32)
    src_y = y - lead_steps * v
    src_x = x - lead_steps * u
    warped = ndi.map_coordinates(
        np.nan_to_num(img, nan=0.0),
        [src_y, src_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.maximum(warped, 0.0).astype(np.float32)


def calculate_contingency_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_continuous: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive contingency verification metrics."""
    tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
    fp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
    fn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
    tn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))
    total = tp + fp + fn + tn
    pos_samples = tp + fn

    csi = tp / max(tp + fp + fn, 1)
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = (tp + tn) / max(total, 1)
    prevalence = pos_samples / max(total, 1)

    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / max(total, 1)
    num = (tp + tn) - expected_correct
    den = total - expected_correct
    hss = float(num / den) if den > 0 else 0.0

    brier = float(np.mean((y_pred_binary.astype(float) - y_true_binary.astype(float)) ** 2))

    roc_auc = None
    pr_auc = None
    if y_pred_continuous is not None and pos_samples > 0 and pos_samples < total:
        try:
            roc_auc = float(roc_auc_score(y_true_binary, y_pred_continuous))
            precisions, recalls, _ = precision_recall_curve(y_true_binary, y_pred_continuous)
            pr_auc = float(auc(recalls, precisions))
        except Exception:
            pass

    return {
        "csi": round(csi, 4),
        "f1": round(f1, 4),
        "pod_recall": round(pod, 4),
        "far": round(far, 4),
        "precision": round(prec, 4),
        "accuracy": round(acc, 4),
        "hss": round(hss, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "total_pixels": total,
        "positive_pixels": pos_samples,
        "prevalence": round(prevalence, 5),
    }


def evaluate_optical_flow_and_persistence(
    precip_array: np.ndarray,
    split_indices: Dict[str, Tuple[int, int]],
    threshold: float = DEFAULT_THRESHOLD,
    sigma: float = 0.8,
    win_size: int = 5,
    reg: float = 1e-3,
) -> Dict[str, Any]:
    """Evaluate Optical Flow and Persistence across all splits and lead horizons."""
    results: Dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "threshold_mm_hr": threshold,
            "sigma": sigma,
            "win_size": win_size,
            "regularization": reg,
            "leads_minutes": [k * 30 for k in LEAD_HORIZONS.values()],
        },
        "optical_flow": {},
        "persistence": {},
    }

    for split_name, (start_idx, end_idx) in split_indices.items():
        results["optical_flow"][split_name] = {}
        results["persistence"][split_name] = {}

        for lead_name, lead_k in LEAD_HORIZONS.items():
            # Adjust range so t-1 is valid and t + lead_k <= end_idx
            eval_start = max(start_idx, 1)
            eval_end = end_idx - lead_k

            if eval_start > eval_end:
                continue

            y_trues = []
            y_preds_of_bin = []
            y_preds_of_cont = []
            y_preds_p_bin = []
            y_preds_p_cont = []

            for t in range(eval_start, eval_end + 1):
                i_prev = precip_array[t - 1]
                i_curr = precip_array[t]
                i_future = precip_array[t + lead_k]

                gt_bin = (i_future >= threshold).astype(np.int32)

                # Optical Flow extrapolation
                u, v = compute_lucas_kanade_flow(i_prev, i_curr, sigma=sigma, win_size=win_size, reg=reg)
                of_pred_rate = semi_lagrangian_extrapolate(i_curr, u, v, lead_k)
                of_bin = (of_pred_rate >= threshold).astype(np.int32)

                # Persistence extrapolation
                p_bin = (i_curr >= threshold).astype(np.int32)

                y_trues.append(gt_bin.ravel())
                y_preds_of_bin.append(of_bin.ravel())
                y_preds_of_cont.append(of_pred_rate.ravel())
                y_preds_p_bin.append(p_bin.ravel())
                y_preds_p_cont.append(i_curr.ravel())

            all_y_true = np.concatenate(y_trues)
            all_of_bin = np.concatenate(y_preds_of_bin)
            all_of_cont = np.concatenate(y_preds_of_cont)
            all_p_bin = np.concatenate(y_preds_p_bin)
            all_p_cont = np.concatenate(y_preds_p_cont)

            results["optical_flow"][split_name][lead_name] = calculate_contingency_metrics(
                all_y_true, all_of_bin, all_of_cont
            )
            results["persistence"][split_name][lead_name] = calculate_contingency_metrics(
                all_y_true, all_p_bin, all_p_cont
            )

    return results


def plot_optical_flow_evaluation(
    results: Dict[str, Any],
    precip_array: np.ndarray,
    output_dir: Path,
    test_slice_idx: int = 280,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Generate high-resolution evaluation figures for optical flow baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSI vs Lead Time comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    leads = [30, 60, 120, 180]
    lead_keys = ["lead_30min", "lead_60min", "lead_120min", "lead_180min"]

    # Test split performance
    test_csi_of = [results["optical_flow"]["test"][k]["csi"] for k in lead_keys]
    test_csi_p = [results["persistence"]["test"][k]["csi"] for k in lead_keys]
    test_pod_of = [results["optical_flow"]["test"][k]["pod_recall"] for k in lead_keys]
    test_pod_p = [results["persistence"]["test"][k]["pod_recall"] for k in lead_keys]

    ax1.plot(leads, test_csi_of, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Optical Flow (rainymotion adapt)")
    ax1.plot(leads, test_csi_p, "s--", color="#7f7f7f", linewidth=2.0, markersize=7, label="Lagged Persistence")
    ax1.set_xlabel("Lead Time (minutes)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Critical Success Index (CSI)", fontsize=12, fontweight="bold")
    ax1.set_title("Test Set CSI vs. Forecast Horizon (Threshold >= 5 mm/hr)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0.0, 0.5)

    ax2.plot(leads, test_pod_of, "o-", color="#2ca02c", linewidth=2.5, markersize=8, label="Optical Flow POD")
    ax2.plot(leads, test_pod_p, "s--", color="#d62728", linewidth=2.0, markersize=7, label="Persistence POD")
    ax2.set_xlabel("Lead Time (minutes)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability of Detection (POD / Recall)", fontsize=12, fontweight="bold")
    ax2.set_title("Test Set POD vs. Forecast Horizon", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0.0, 0.75)

    plt.tight_layout()
    fig_path = output_dir / "optical_flow_lead_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 2. Spatial Extrapolation Case Study
    t = test_slice_idx
    i_prev = precip_array[t - 1]
    i_curr = precip_array[t]
    u, v = compute_lucas_kanade_flow(i_prev, i_curr)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=300)
    cmap = plt.cm.YlGnBu

    # Row 1: Ground truth evolution
    vmax = max(np.max(i_curr), 15.0)
    im0 = axes[0, 0].imshow(i_curr, cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"Input Rain Rate P(t)\n[t={t}]", fontsize=10, fontweight="bold")
    axes[0, 1].imshow(precip_array[t + 1], cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Ground Truth (+30m)", fontsize=10, fontweight="bold")
    axes[0, 2].imshow(precip_array[t + 2], cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 2].set_title("Ground Truth (+60m)", fontsize=10, fontweight="bold")
    axes[0, 3].imshow(precip_array[t + 4], cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 3].set_title("Ground Truth (+120m)", fontsize=10, fontweight="bold")

    # Row 2: Optical flow predictions
    # Quiver plot of velocity field on input
    step = 3
    y_sub, x_sub = np.mgrid[0:25:step, 0:40:step]
    axes[1, 0].imshow(i_curr, cmap=cmap, vmin=0, vmax=vmax, alpha=0.8)
    axes[1, 0].quiver(x_sub, y_sub, u[::step, ::step], -v[::step, ::step], color="red", scale=25)
    axes[1, 0].set_title("Advection Vectors (u, v)", fontsize=10, fontweight="bold")

    pred_30 = semi_lagrangian_extrapolate(i_curr, u, v, 1)
    axes[1, 1].imshow(pred_30, cmap=cmap, vmin=0, vmax=vmax)
    axes[1, 1].set_title("Optical Flow (+30m)", fontsize=10, fontweight="bold")

    pred_60 = semi_lagrangian_extrapolate(i_curr, u, v, 2)
    axes[1, 2].imshow(pred_60, cmap=cmap, vmin=0, vmax=vmax)
    axes[1, 2].set_title("Optical Flow (+60m)", fontsize=10, fontweight="bold")

    pred_120 = semi_lagrangian_extrapolate(i_curr, u, v, 4)
    axes[1, 3].imshow(pred_120, cmap=cmap, vmin=0, vmax=vmax)
    axes[1, 3].set_title("Optical Flow (+120m)", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label("Precipitation Rate (mm/hr)", fontsize=11)

    spatial_path = output_dir / "optical_flow_spatial_extrapolation.png"
    plt.savefig(spatial_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_pipeline() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Optical Flow Weather Nowcasting Baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to aligned weather NetCDF files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for metrics and plots")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Rainfall threshold in mm/hr")
    parser.add_argument("--sigma", type=float, default=0.8, help="Gaussian smoothing scale")
    parser.add_argument("--win-size", type=int, default=5, help="Lucas-Kanade window size")
    parser.add_argument("--reg", type=float, default=1e-3, help="Ridge regularization parameter")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTICAL FLOW PRECIPITATION NOWCASTING PIPELINE (rainymotion adapt)")
    print("=" * 70)
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold:        {args.threshold} mm/hr")
    print(f"Lead horizons:    +30m, +60m, +120m, +180m")

    # Load data
    precip_all, times, refs, lats, lons, splits = load_all_aligned_precipitation(args.data_dir)
    print(f"Loaded precipitation array: {precip_all.shape} across {len(splits)} splits.")
    for sname, (s_idx, e_idx) in splits.items():
        print(f"  - Split '{sname}': steps {s_idx}..{e_idx} ({e_idx - s_idx + 1} steps)")

    # Evaluate
    print("\nExecuting optical flow evaluation across splits...")
    results = evaluate_optical_flow_and_persistence(
        precip_all, splits, threshold=args.threshold, sigma=args.sigma, win_size=args.win_size, reg=args.reg
    )

    # Print summary table for Test split
    print("\n" + "=" * 70)
    print("TEST SPLIT BENCHMARK: OPTICAL FLOW vs. PERSISTENCE (July 6-7, 2025)")
    print("=" * 70)
    print(f"{'Lead Horizon':<15} {'Model':<18} {'CSI':<8} {'POD':<8} {'FAR':<8} {'F1':<8} {'HSS':<8}")
    print("-" * 70)
    for lead_key in ["lead_30min", "lead_60min", "lead_120min", "lead_180min"]:
        of_res = results["optical_flow"]["test"][lead_key]
        p_res = results["persistence"]["test"][lead_key]
        lead_disp = lead_key.replace("lead_", "+")
        print(f"{lead_disp:<15} {'Optical Flow':<18} {of_res['csi']:<8.4f} {of_res['pod_recall']:<8.4f} {of_res['far']:<8.4f} {of_res['f1']:<8.4f} {of_res['hss']:<8.4f}")
        print(f"{lead_disp:<15} {'Persistence':<18} {p_res['csi']:<8.4f} {p_res['pod_recall']:<8.4f} {p_res['far']:<8.4f} {p_res['f1']:<8.4f} {p_res['hss']:<8.4f}")
        print("-" * 70)

    # Save metrics JSON
    metrics_path = args.output_dir / "optical_flow_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optical flow metrics to: {metrics_path}")

    # Generate figures
    plot_optical_flow_evaluation(results, precip_all, args.output_dir, threshold=args.threshold)
    print(f"Saved evaluation figures to: {args.output_dir}")
    print("Optical flow baseline pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
