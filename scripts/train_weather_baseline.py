"""Weather Nowcasting ML Baseline Pipeline.

Predicts future 30-minute IMERG heavy-rainfall occurrence from available historical
satellite IR brightness temperature (MERGIR) and prior IMERG precipitation observations.

Chronological Split:
  - Training:   July 1 - July 4, 2025 (4 days, 192 slots)
  - Validation: July 5, 2025 (1 day, 48 slots)
  - Testing:    July 6 - July 7, 2025 (2 days, 96 slots)

Strict Anti-Leakage & Featurization Rules:
  - Preprocessing scaler is fit ONLY on Training data.
  - Zero future rainfall in input features (t_end <= T_ref).
  - Evaluates Persistence baseline, Logistic Regression baseline, and PyTorch MLP baseline.
  - Documents IMERG proxy label caveat and hardware capabilities.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DEFAULT_HEAVY_RAIN_THRESHOLD = 5.0  # mm/hr (IMD standard for heavy convective rain)
PROXY_CAVEAT = (
    "IMERG rainfall is a satellite-retrieval product (proxy label) rather than independent "
    "ground-truth rain-gauge validation. MERGIR IR is one of the observational inputs to IMERG, "
    "so this baseline serves as a pipeline feasibility demonstration, not independent proof of skill."
)


def inspect_hardware() -> Dict[str, Any]:
    """Inspect local hardware and detect GPU/VRAM/CUDA support."""
    info: Dict[str, Any] = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_available": HAS_TORCH,
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_mb": None,
        "device_selected": "cpu",
    }

    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_total_mb"] = round(total_bytes / (1024 * 1024), 2)
            info["device_selected"] = "cuda"
            info["cuda_version"] = torch.version.cuda
    return info


def load_daily_aligned_datasets(
    data_dir: Path,
    dates: List[str],
) -> Dict[str, np.ndarray]:
    """Load and concatenate daily aligned NetCDF datasets for given dates."""
    ir_mean_list = []
    ir_min_list = []
    input_p_list = []
    target_30_list = []
    target_60_list = []
    mask_list = []
    time_iso_list = []
    ref_iso_list = []
    lats = None
    lons = None

    for d in dates:
        fpath = data_dir / f"aligned_weather_mergir_{d}.nc4"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required aligned dataset: {fpath}")

        with h5py.File(fpath, "r") as f:
            if lats is None:
                lats = f["lat"][:]
                lons = f["lon"][:]

            ir_mean_list.append(f["satellite_tir1_mean"][:])
            ir_min_list.append(f["satellite_tir1_min"][:])
            input_p_list.append(f["input_precip"][:])
            target_30_list.append(f["target_precip_lead30"][:])
            target_60_list.append(f["target_precip_lead60"][:])
            mask_list.append(f["satellite_mask"][:])

            # Read timestamps
            nom = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["nominal_slot_iso"][:]]
            ref = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["reference_time_iso"][:]]
            time_iso_list.extend(nom)
            ref_iso_list.extend(ref)

    return {
        "ir_mean": np.concatenate(ir_mean_list, axis=0),        # (T, 25, 40)
        "ir_min": np.concatenate(ir_min_list, axis=0),          # (T, 25, 40)
        "input_precip": np.concatenate(input_p_list, axis=0),  # (T, 25, 40)
        "target_lead30": np.concatenate(target_30_list, axis=0),# (T, 25, 40)
        "target_lead60": np.concatenate(target_60_list, axis=0),# (T, 25, 40)
        "satellite_mask": np.concatenate(mask_list, axis=0),    # (T, 25, 40)
        "nominal_slot_iso": time_iso_list,
        "reference_time_iso": ref_iso_list,
        "lat": lats,
        "lon": lons,
    }


def compute_local_spatial_mean(grid_3d: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compute local 3x3 neighborhood mean for each timestep without NaN propagation."""
    pad = kernel_size // 2
    T, H, W = grid_3d.shape
    is_valid = np.isfinite(grid_3d).astype(np.float32)
    filled = np.nan_to_num(grid_3d, nan=0.0)

    padded_data = np.pad(filled, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    padded_weights = np.pad(is_valid, ((0, 0), (pad, pad), (pad, pad)), mode="edge")

    sum_data = np.zeros_like(grid_3d, dtype=np.float32)
    sum_weights = np.zeros_like(grid_3d, dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            sum_data += padded_data[:, i:i+H, j:j+W]
            sum_weights += padded_weights[:, i:i+H, j:j+W]

    out = np.divide(sum_data, sum_weights, out=np.zeros_like(sum_data), where=(sum_weights > 0))
    return out


def extract_pixel_features_and_labels(
    data: Dict[str, np.ndarray],
    threshold_mm_hr: float = DEFAULT_HEAVY_RAIN_THRESHOLD,
    scaler: Optional[StandardScaler] = None,
    imputer: Optional[SimpleImputer] = None,
    fit_preprocessors: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, SimpleImputer, Dict[str, Any]]:
    """Extract tabular pixel-level features strictly from prior inputs and build binary labels.

    Features strictly available at or before T_ref:
      1. input_precip (mm/hr)
      2. ir_mean (K)
      3. ir_min (K)
      4. input_precip_spatial_mean (3x3 context, mm/hr)
      5. ir_mean_spatial_mean (3x3 context, K)
      6. normalized_lat
      7. normalized_lon
      8. sin_hour (diurnal cycle)
      9. cos_hour (diurnal cycle)
      10. ir_is_nan (binary indicator of missing satellite sensor pixel)

    Target:
      1 if target_lead30 >= threshold_mm_hr else 0
    """
    ir_mean = data["ir_mean"]
    ir_min = data["ir_min"]
    input_p = data["input_precip"]
    target_30 = data["target_lead30"]
    mask = data["satellite_mask"]
    lats = data["lat"]
    lons = data["lon"]
    ref_times = data["reference_time_iso"]

    T, H, W = ir_mean.shape

    # 3x3 spatial context
    input_p_context = compute_local_spatial_mean(input_p)
    ir_mean_context = compute_local_spatial_mean(ir_mean)

    # Missing indicator
    ir_is_nan_grid = np.isnan(ir_mean).astype(np.float32)

    # 2D coordinates broadcast
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_norm = (lat_grid - lat_grid.min()) / (lat_grid.max() - lat_grid.min() + 1e-6)
    lon_norm = (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min() + 1e-6)

    # Diurnal cycle
    sin_hours = np.zeros(T, dtype=np.float32)
    cos_hours = np.zeros(T, dtype=np.float32)
    for t_idx, iso_str in enumerate(ref_times):
        dt = datetime.fromisoformat(iso_str)
        hour_fraction = dt.hour + dt.minute / 60.0
        sin_hours[t_idx] = np.sin(2.0 * np.pi * hour_fraction / 24.0)
        cos_hours[t_idx] = np.cos(2.0 * np.pi * hour_fraction / 24.0)

    # Valid mask: target must be finite and >= 0, input >= 0, satellite valid
    valid = (target_30 >= 0.0) & np.isfinite(target_30) & (input_p >= 0.0) & (mask == 1)

    # Flatten arrays
    lat_grid_3d = np.repeat(lat_norm[None, :, :], T, axis=0)
    lon_grid_3d = np.repeat(lon_norm[None, :, :], T, axis=0)
    sin_grid_3d = np.repeat(sin_hours[:, None, None], H * W).reshape(T, H, W)
    cos_grid_3d = np.repeat(cos_hours[:, None, None], H * W).reshape(T, H, W)

    f_input_p = input_p[valid]
    f_ir_mean = ir_mean[valid]
    f_ir_min = ir_min[valid]
    f_input_p_ctx = input_p_context[valid]
    f_ir_ctx = ir_mean_context[valid]
    f_lat = lat_grid_3d[valid]
    f_lon = lon_grid_3d[valid]
    f_sin = sin_grid_3d[valid]
    f_cos = cos_grid_3d[valid]
    f_nan = ir_is_nan_grid[valid]

    X_raw = np.column_stack([
        f_input_p,
        f_ir_mean,
        f_ir_min,
        f_input_p_ctx,
        f_ir_ctx,
        f_lat,
        f_lon,
        f_sin,
        f_cos,
        f_nan,
    ])

    raw_targets = target_30[valid]
    y = (raw_targets >= threshold_mm_hr).astype(np.int32)
    prior_p = f_input_p  # for persistence baseline

    feature_names = [
        "input_precip_mm_hr",
        "ir_tb_mean_k",
        "ir_tb_min_k",
        "input_precip_3x3_ctx",
        "ir_tb_3x3_ctx",
        "lat_norm",
        "lon_norm",
        "sin_hour",
        "cos_hour",
        "ir_is_missing",
    ]

    # Preprocessing Imputer and Scaler: fit ONLY on training data
    if fit_preprocessors:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_raw)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        if imputer is None or scaler is None:
            raise ValueError("Must provide fitted imputer and scaler if fit_preprocessors is False.")
        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)

    metadata = {
        "total_valid_pixels": int(len(y)),
        "positive_events": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "feature_names": feature_names,
    }

    return X_scaled, y, prior_p, scaler, imputer, metadata


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate meteorological and ML classification metrics with zero-division guards."""
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_total - n_pos

    if n_total == 0:
        return {
            "total_samples": 0,
            "positive_samples": 0,
            "prevalence": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "csi": 0.0,
            "far": 0.0,
            "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = float(n_pos / n_total)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0  # Critical Success Index / Threat Score
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0          # False Alarm Ratio

    metrics = {
        "total_samples": int(n_total),
        "positive_samples": int(n_pos),
        "prevalence": round(prevalence, 6),
        "precision": round(precision, 4),
        "recall_pod": round(recall, 4),
        "f1_score": round(f1, 4),
        "csi_threat_score": round(csi, 4),
        "false_alarm_ratio": round(far, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }

    if y_prob is not None and n_pos > 0 and n_neg > 0:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


if HAS_TORCH:
    class WeatherMLP(nn.Module):
        """Lightweight neural network for checking GPU execution and non-linear baseline."""

        def __init__(self, in_features: int, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


def train_pytorch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 8,
    batch_size: int = 1024,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a lightweight PyTorch MLP on the specified device (GPU if available)."""
    if not HAS_TORCH:
        return None, {"error": "PyTorch not installed"}

    dev = torch.device(device)
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight_val = float(neg_count / (pos_count + 1e-5))

    model = WeatherMLP(in_features=X_train.shape[1], hidden_dim=64).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=dev))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = []
    start_time = time.perf_counter()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        history.append({"epoch": ep + 1, "train_loss": round(avg_loss, 4)})

    total_training_time_sec = round(time.perf_counter() - start_time, 2)

    train_info = {
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "total_training_time_seconds": total_training_time_sec,
        "history": history,
    }

    if device == "cuda":
        train_info["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        train_info["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)

    return model, train_info


def predict_pytorch(model: Any, X: np.ndarray, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Run inference with PyTorch model and return probabilities."""
    dev = torch.device(device)
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs_list = []

    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(dev)
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)

    return np.concatenate(probs_list)


def plot_evaluation(
    eval_data: Dict[str, Any],
    out_png: Path,
) -> None:
    """Render a comprehensive evaluation figure comparing Persistence, Logistic Regression, and PyTorch."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # 1. Confusion Matrix Comparison (Test set)
    ax_cm = axes[0, 0]
    models = ["Persistence", "Logistic Reg", "PyTorch MLP"]
    metrics_list = [
        eval_data["test"]["persistence"],
        eval_data["test"]["logistic_regression"],
        eval_data["test"]["pytorch_mlp"] if "pytorch_mlp" in eval_data["test"] else None,
    ]

    csi_vals = [m["csi_threat_score"] if m else 0.0 for m in metrics_list]
    f1_vals = [m["f1_score"] if m else 0.0 for m in metrics_list]
    pod_vals = [m["recall_pod"] if m else 0.0 for m in metrics_list]
    far_vals = [m["false_alarm_ratio"] if m else 0.0 for m in metrics_list]

    x = np.arange(len(models))
    width = 0.2
    ax_cm.bar(x - 1.5*width, csi_vals, width, label="CSI (Threat Score)", color="#2b5c8f")
    ax_cm.bar(x - 0.5*width, f1_vals, width, label="F1-Score", color="#3b8b6e")
    ax_cm.bar(x + 0.5*width, pod_vals, width, label="Recall (POD)", color="#d95f02")
    ax_cm.bar(x + 1.5*width, far_vals, width, label="False Alarm Ratio", color="#7570b3")
    ax_cm.set_xticks(x)
    ax_cm.set_xticklabels(models, fontweight="bold")
    ax_cm.set_ylabel("Score (0.0 to 1.0)")
    ax_cm.set_title("Test Set Performance Comparison (July 6–7, 2025)", fontweight="bold")
    ax_cm.legend(loc="upper right")
    ax_cm.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Precision-Recall Curves
    ax_pr = axes[0, 1]
    y_test = eval_data["raw_test"]["y_true"]
    if "y_prob_lr" in eval_data["raw_test"]:
        p_lr, r_lr, _ = precision_recall_curve(y_test, eval_data["raw_test"]["y_prob_lr"])
        ax_pr.plot(r_lr, p_lr, label=f"Logistic Regression (AUC={eval_data['test']['logistic_regression'].get('roc_auc', 'N/A')})", color="#3b8b6e", lw=2)
    if "y_prob_pt" in eval_data["raw_test"]:
        p_pt, r_pt, _ = precision_recall_curve(y_test, eval_data["raw_test"]["y_prob_pt"])
        ax_pr.plot(r_pt, p_pt, label=f"PyTorch MLP (AUC={eval_data['test']['pytorch_mlp'].get('roc_auc', 'N/A')})", color="#d95f02", lw=2)

    # Persistence point
    p_pers = eval_data["test"]["persistence"]["precision"]
    r_pers = eval_data["test"]["persistence"]["recall_pod"]
    ax_pr.scatter([r_pers], [p_pers], color="#2b5c8f", s=80, zorder=5, label=f"Persistence (P={p_pers:.2f}, R={r_pers:.2f})")
    ax_pr.axhline(eval_data["test"]["persistence"]["prevalence"], color="gray", linestyle=":", label=f"No-Skill Climatology ({eval_data['test']['persistence']['prevalence']*100:.1f}%)")
    ax_pr.set_xlabel("Recall (Probability of Detection)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve (Severe Rain >= 5.0 mm/hr)", fontweight="bold")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    # 3. Feature Importance (Logistic Regression Coefficients)
    ax_coef = axes[1, 0]
    coefs = eval_data["lr_coefficients"]
    feat_names = eval_data["feature_names"]
    y_pos = np.arange(len(feat_names))
    ax_coef.barh(y_pos, coefs, align="center", color=np.where(np.array(coefs) > 0, "#2b5c8f", "#d95f02"))
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(feat_names, fontsize=9)
    ax_coef.invert_yaxis()
    ax_coef.set_xlabel("Logistic Regression Coefficient Weight")
    ax_coef.set_title("Feature Importance (Standardized Coefficients)", fontweight="bold")
    ax_coef.axvline(0, color="black", lw=1)
    ax_coef.grid(axis="x", linestyle="--", alpha=0.5)

    # 4. Calibration & Diagnostic Note
    ax_note = axes[1, 1]
    ax_note.axis("off")
    hw = eval_data["hardware"]
    gpu_txt = f"{hw['gpu_name']} ({hw['vram_total_mb']} MB VRAM)" if hw["cuda_available"] else "None (CPU Execution)"
    note_txt = (
        "PRELIMINARY PIPELINE TEST SUMMARY\n"
        "--------------------------------------------------\n"
        f"Heavy-Rain Threshold: >= {eval_data['threshold_mm_hr']} mm/hr (IMERG 30-min)\n"
        f"Chronological Split: Train (Jul 1-4) | Val (Jul 5) | Test (Jul 6-7)\n"
        f"Hardware Used: {eval_data['training_device'].upper()} | GPU: {gpu_txt}\n\n"
        "TEST SET METRICS:\n"
        f"  * Persistence:    CSI={eval_data['test']['persistence']['csi_threat_score']:.3f} | F1={eval_data['test']['persistence']['f1_score']:.3f} | FAR={eval_data['test']['persistence']['false_alarm_ratio']:.3f}\n"
        f"  * Logistic Reg:   CSI={eval_data['test']['logistic_regression']['csi_threat_score']:.3f} | F1={eval_data['test']['logistic_regression']['f1_score']:.3f} | FAR={eval_data['test']['logistic_regression']['false_alarm_ratio']:.3f}\n"
    )
    if "pytorch_mlp" in eval_data["test"]:
        note_txt += f"  * PyTorch MLP:    CSI={eval_data['test']['pytorch_mlp']['csi_threat_score']:.3f} | F1={eval_data['test']['pytorch_mlp']['f1_score']:.3f} | FAR={eval_data['test']['pytorch_mlp']['false_alarm_ratio']:.3f}\n"

    note_txt += (
        "\nIMPORTANT METHODOLOGICAL NOTES:\n"
        "1. IMERG is a retrieval product (proxy label), not ground-truth gauge data.\n"
        "2. MERGIR IR is used inside the IMERG retrieval algorithm.\n"
        "3. Successful GPU execution verifies data loading and pipeline throughput,\n"
        "   NOT operational forecasting skill.\n"
        "4. This 7-day pilot validates reproducible end-to-end tensor flow."
    )
    ax_note.text(0.05, 0.95, note_txt, transform=ax_note.transAxes, fontsize=9.5,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"))

    plt.suptitle("Multimodal AI Platform: Weather Nowcasting Baseline Benchmark", fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Wrote evaluation figure to {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weather forecasting baseline pipeline.")
    parser.add_argument("--data-dir", default="data/processed/weather", help="Directory of aligned NetCDF files")
    parser.add_argument("--out-dir", default="outputs/models/weather_baseline", help="Directory to save baseline outputs")
    parser.add_argument("--threshold", type=float, default=DEFAULT_HEAVY_RAIN_THRESHOLD, help="Heavy rain threshold (mm/hr)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device for PyTorch model")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STEP 6: WEATHER NOWCASTING ML BASELINE PIPELINE")
    print(f"Target Variable: Future 30-min IMERG Heavy Rain (>= {args.threshold} mm/hr)")
    print("=" * 65)

    # 1. Hardware Inspection
    hw = inspect_hardware()
    print("\n[Hardware & CUDA Inspection]")
    print(f"  OS:              {hw['os']}")
    print(f"  Python:          {hw['python_version']}")
    print(f"  PyTorch:         {hw.get('torch_version', 'Not installed')}")
    print(f"  CUDA Available:  {hw['cuda_available']}")
    if hw['cuda_available']:
        print(f"  GPU Name:        {hw['gpu_name']}")
        print(f"  Dedicated VRAM:  {hw['vram_total_mb']} MB")
        print(f"  CUDA Version:    {hw.get('cuda_version', 'N/A')}")
    else:
        print("  CUDA Status:     No CUDA GPU available or PyTorch CPU-only.")

    # Select compute device
    if args.device == "auto":
        device = "cuda" if hw["cuda_available"] else "cpu"
    else:
        device = args.device if (args.device == "cpu" or hw["cuda_available"]) else "cpu"
    print(f"  Selected Device: {device.upper()}")

    # 2. Chronological Splits
    train_dates = ["20250701", "20250702", "20250703", "20250704"]
    val_dates = ["20250705"]
    test_dates = ["20250706", "20250707"]

    print("\n[Loading Aligned Datasets]")
    print(f"  Train:      July 1-4, 2025 ({len(train_dates)} days)")
    print(f"  Validation: July 5, 2025   ({len(val_dates)} days)")
    print(f"  Test:       July 6-7, 2025 ({len(test_dates)} days)")

    train_raw = load_daily_aligned_datasets(data_dir, train_dates)
    val_raw = load_daily_aligned_datasets(data_dir, val_dates)
    test_raw = load_daily_aligned_datasets(data_dir, test_dates)

    # 3. Extract Features & Labels with Strict Ordering
    print(f"\n[Extracting Pixel Features & Labels (Threshold >= {args.threshold} mm/hr)]")
    # Fit imputer and scaler ONLY on training data
    X_train, y_train, p_train, scaler, imputer, train_meta = extract_pixel_features_and_labels(
        train_raw, threshold_mm_hr=args.threshold, fit_preprocessors=True
    )
    # Transform val and test using fitted imputer and scaler
    X_val, y_val, p_val, _, _, val_meta = extract_pixel_features_and_labels(
        val_raw, threshold_mm_hr=args.threshold, scaler=scaler, imputer=imputer, fit_preprocessors=False
    )
    X_test, y_test, p_test, _, _, test_meta = extract_pixel_features_and_labels(
        test_raw, threshold_mm_hr=args.threshold, scaler=scaler, imputer=imputer, fit_preprocessors=False
    )

    print(f"  Train: {X_train.shape[0]:,} pixels | Positive: {train_meta['positive_events']:,} ({train_meta['prevalence']*100:.2f}%)")
    print(f"  Val:   {X_val.shape[0]:,} pixels | Positive: {val_meta['positive_events']:,} ({val_meta['prevalence']*100:.2f}%)")
    print(f"  Test:  {X_test.shape[0]:,} pixels | Positive: {test_meta['positive_events']:,} ({test_meta['prevalence']*100:.2f}%)")

    # 4. Persistence Baseline
    print("\n[Evaluating Persistence Baseline]")
    pred_persist_train = (p_train >= args.threshold).astype(int)
    pred_persist_val = (p_val >= args.threshold).astype(int)
    pred_persist_test = (p_test >= args.threshold).astype(int)

    metrics_persist = {
        "train": compute_binary_metrics(y_train, pred_persist_train),
        "val": compute_binary_metrics(y_val, pred_persist_val),
        "test": compute_binary_metrics(y_test, pred_persist_test),
    }
    print(f"  Persistence Test -> CSI: {metrics_persist['test']['csi_threat_score']:.4f} | F1: {metrics_persist['test']['f1_score']:.4f} | Recall: {metrics_persist['test']['recall_pod']:.4f} | Precision: {metrics_persist['test']['precision']:.4f}")

    # 5. Regularized Logistic Regression Baseline
    print("\n[Training Regularized Logistic Regression Baseline]")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    prob_lr_train = lr_model.predict_proba(X_train)[:, 1]
    prob_lr_val = lr_model.predict_proba(X_val)[:, 1]
    prob_lr_test = lr_model.predict_proba(X_test)[:, 1]

    # Tune decision threshold on Validation set to optimize CSI / F1
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = 0.5
    best_val_csi = -1.0
    for th in thresholds:
        m = compute_binary_metrics(y_val, (prob_lr_val >= th).astype(int))
        if m["csi_threat_score"] > best_val_csi:
            best_val_csi = m["csi_threat_score"]
            best_thresh = float(th)

    print(f"  Optimal Val Decision Threshold: {best_thresh:.2f} (Val CSI: {best_val_csi:.4f})")

    metrics_lr = {
        "train": compute_binary_metrics(y_train, (prob_lr_train >= best_thresh).astype(int), prob_lr_train),
        "val": compute_binary_metrics(y_val, (prob_lr_val >= best_thresh).astype(int), prob_lr_val),
        "test": compute_binary_metrics(y_test, (prob_lr_test >= best_thresh).astype(int), prob_lr_test),
        "tuned_threshold": best_thresh,
    }
    print(f"  Logistic Reg Test -> CSI: {metrics_lr['test']['csi_threat_score']:.4f} | F1: {metrics_lr['test']['f1_score']:.4f} | Recall: {metrics_lr['test']['recall_pod']:.4f} | Precision: {metrics_lr['test']['precision']:.4f} | ROC-AUC: {metrics_lr['test'].get('roc_auc')}")

    # 6. PyTorch MLP Baseline (checks GPU execution)
    pt_model = None
    pt_train_info = None
    prob_pt_test = None
    metrics_pt = None

    if HAS_TORCH:
        print(f"\n[Training Lightweight PyTorch MLP on {device.upper()}]")
        pt_model, pt_train_info = train_pytorch_model(
            X_train, y_train, X_val, y_val, device=device, epochs=8, batch_size=2048
        )
        prob_pt_train = predict_pytorch(pt_model, X_train, device=device)
        prob_pt_val = predict_pytorch(pt_model, X_val, device=device)
        prob_pt_test = predict_pytorch(pt_model, X_test, device=device)

        # Tune threshold on val
        best_pt_thresh = 0.5
        best_pt_csi = -1.0
        for th in thresholds:
            m = compute_binary_metrics(y_val, (prob_pt_val >= th).astype(int))
            if m["csi_threat_score"] > best_pt_csi:
                best_pt_csi = m["csi_threat_score"]
                best_pt_thresh = float(th)

        metrics_pt = {
            "train": compute_binary_metrics(y_train, (prob_pt_train >= best_pt_thresh).astype(int), prob_pt_train),
            "val": compute_binary_metrics(y_val, (prob_pt_val >= best_pt_thresh).astype(int), prob_pt_val),
            "test": compute_binary_metrics(y_test, (prob_pt_test >= best_pt_thresh).astype(int), prob_pt_test),
            "tuned_threshold": best_pt_thresh,
            "training_info": pt_train_info,
        }
        print(f"  PyTorch MLP Test -> CSI: {metrics_pt['test']['csi_threat_score']:.4f} | F1: {metrics_pt['test']['f1_score']:.4f} | Recall: {metrics_pt['test']['recall_pod']:.4f} | Precision: {metrics_pt['test']['precision']:.4f} | ROC-AUC: {metrics_pt['test'].get('roc_auc')}")
        print(f"  PyTorch Training Time: {pt_train_info['total_training_time_seconds']}s on {device.upper()}")

    # 7. Save Models, Scaler, and Imputer
    print("\n[Saving Model Artifacts]")
    imputer_path = out_dir / "imputer.joblib"
    joblib.dump(imputer, imputer_path)
    print(f"  Saved imputer to {imputer_path}")

    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")

    lr_model_path = out_dir / "logistic_regression_model.joblib"
    joblib.dump(lr_model, lr_model_path)
    print(f"  Saved Logistic Regression to {lr_model_path}")

    if pt_model is not None:
        pt_model_path = out_dir / "pytorch_baseline_model.pt"
        torch.save(pt_model.state_dict(), pt_model_path)
        print(f"  Saved PyTorch model weights to {pt_model_path}")

    # 8. Save Metrics JSON
    all_metrics = {
        "persistence": metrics_persist,
        "logistic_regression": metrics_lr,
    }
    if metrics_pt:
        all_metrics["pytorch_mlp"] = metrics_pt

    metrics_json_path = out_dir / "baseline_metrics.json"
    metrics_json_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"  Saved metrics to {metrics_json_path}")

    # 9. Save Configuration JSON
    config = {
        "pipeline": "Step 6 Weather Nowcasting ML Baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "heavy_rain_threshold_mm_hr": args.threshold,
        "study_region": {
            "name": "Meghalaya, India",
            "grid_dimensions": [25, 40],
            "bounding_box_wsen": [89.5, 24.5, 93.5, 27.0],
        },
        "chronological_splits": {
            "train": {"dates": train_dates, "days": len(train_dates), "samples": int(len(y_train))},
            "validation": {"dates": val_dates, "days": len(val_dates), "samples": int(len(y_val))},
            "test": {"dates": test_dates, "days": len(test_dates), "samples": int(len(y_test))},
        },
        "features": train_meta["feature_names"],
        "hardware": hw,
        "training_device": device,
        "anti_leakage_audit": {
            "strict_chronological_splits": True,
            "preprocessing_fit_on_train_only": True,
            "zero_future_rainfall_in_features": True,
        },
        "proxy_label_caveat": PROXY_CAVEAT,
        "disclaimer": "Preliminary pipeline test only. Successful GPU execution verifies data loading and pipeline throughput, NOT operational forecasting skill.",
    }
    config_json_path = out_dir / "model_config.json"
    config_json_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  Saved config to {config_json_path}")

    # 10. Generate Evaluation Figures
    eval_fig_data = {
        "test": {
            "persistence": metrics_persist["test"],
            "logistic_regression": metrics_lr["test"],
        },
        "raw_test": {
            "y_true": y_test,
            "y_prob_lr": prob_lr_test,
        },
        "lr_coefficients": list(lr_model.coef_[0]),
        "feature_names": train_meta["feature_names"],
        "hardware": hw,
        "training_device": device,
        "threshold_mm_hr": args.threshold,
    }
    if prob_pt_test is not None:
        eval_fig_data["test"]["pytorch_mlp"] = metrics_pt["test"]
        eval_fig_data["raw_test"]["y_prob_pt"] = prob_pt_test

    plot_png_path = out_dir / "weather_baseline_evaluation.png"
    plot_evaluation(eval_fig_data, plot_png_path)

    print("\n" + "=" * 65)
    print("STEP 6 BASELINE EXECUTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
