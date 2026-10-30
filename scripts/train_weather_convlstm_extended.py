"""Extended Multi-Horizon Spatiotemporal ConvLSTM Weather Nowcaster.

Extends the 2D ConvLSTM architecture to 4 prediction heads:
  - Lead 1: +30 min (t+1)
  - Lead 2: +60 min (t+2)
  - Lead 4: +120 min (t+4)
  - Lead 6: +180 min (t+6)

Trained on NVIDIA GeForce RTX 4050 Laptop GPU (cuda:0).
Evaluated on held-out test split (July 6-7, 2025).
Outputs:
  - outputs/models/weather_convlstm/extended_convlstm_model.pt
  - outputs/models/weather_convlstm/extended_convlstm_metrics.json
  - outputs/experiments/training_history_extended.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_weather_convlstm import (
    ConvLSTMCell,
    compute_normalization_stats,
    inspect_hardware,
    load_daily_aligned_datasets,
)

DEFAULT_HEAVY_RAIN_THRESHOLD = 5.0
LEAD_STEPS = [1, 2, 4, 6]  # +30m, +60m, +120m, +180m
LEAD_NAMES = ["lead_30min", "lead_60min", "lead_120min", "lead_180min"]


class ExtendedSpatiotemporalDataset(Dataset):
    """Dataset producing 4-lead target tensors."""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        stats: Dict[str, float],
        t_start: int,
        t_end: int,
        t_in: int = 4,
        threshold: float = DEFAULT_HEAVY_RAIN_THRESHOLD,
    ):
        super().__init__()
        self.t_in = t_in
        self.threshold = threshold

        tb = data["ir_mean"]
        p = data["input_precip"]
        mask = data["satellite_mask"]

        # Normalize features strictly with train statistics
        tb_norm = np.where(
            (mask == 1) & np.isfinite(tb),
            (tb - stats["tb_mean"]) / max(stats["tb_std"], 1e-6),
            0.0,
        ).astype(np.float32)

        log_p = np.log1p(np.maximum(p, 0.0))
        p_norm = np.where(
            np.isfinite(log_p),
            (log_p - stats["log_p_mean"]) / max(stats["log_p_std"], 1e-6),
            0.0,
        ).astype(np.float32)

        mask_ch = mask.astype(np.float32)
        self.features = np.stack([tb_norm, p_norm, mask_ch], axis=1)  # (T, 3, H, W)

        self.samples_x = []
        self.samples_y = []
        self.samples_valid = []
        self.samples_ref_time = []
        self.samples_last_p = []

        max_lead = max(LEAD_STEPS)
        actual_end = min(t_end, len(p) - max_lead - 1)

        for t in range(t_start, actual_end + 1):
            target_list = []
            val_list = []
            all_valid = True

            for k in LEAD_STEPS:
                future_p = p[t + k]
                is_val = np.isfinite(future_p)
                if not np.any(is_val):
                    all_valid = False
                    break
                target_list.append((future_p >= threshold).astype(np.float32))
                val_list.append(is_val.astype(np.float32))

            if not all_valid:
                continue

            x_seq = self.features[t - t_in + 1 : t + 1]
            self.samples_x.append(x_seq)
            self.samples_y.append(np.stack(target_list, axis=0))
            self.samples_valid.append(np.stack(val_list, axis=0))
            self.samples_ref_time.append(data["reference_time_iso"][t])
            self.samples_last_p.append(p[t])

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.samples_x[idx]),
            torch.from_numpy(self.samples_y[idx]),
            torch.from_numpy(self.samples_valid[idx]),
        )


class ExtendedWeatherConvLSTM(nn.Module):
    """4-Head Spatiotemporal ConvLSTM (+30m, +60m, +120m, +180m)."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        num_leads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_channels, hidden_channels[0], kernel_size)
        self.cell2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], kernel_size)

        # 4 Prediction heads
        heads = []
        for _ in range(num_leads):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels[1], 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(16, 1, kernel_size=1),
                )
            )
        self.lead_heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _, h, w = x.shape
        state1, state2 = None, None
        for t in range(seq_len):
            x_t = x[:, t]
            h1, c1 = self.cell1(x_t, state1)
            state1 = (h1, c1)
            h2, c2 = self.cell2(h1, state2)
            state2 = (h2, c2)

        last_hidden = state2[0]
        outputs = [head(last_hidden).squeeze(1) for head in self.lead_heads]
        return torch.stack(outputs, dim=1)  # (B, num_leads, H, W)


def calculate_metrics_for_lead(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    """Compute meteorological metrics for one lead."""
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    csi = tp / max(tp + fp + fn, 1)
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    brier = float(brier_score_loss(y_true, y_prob))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "csi": round(float(csi), 4),
        "f1": round(float(f1), 4),
        "pod_recall": round(float(pod), 4),
        "far": round(float(far), 4),
        "precision": round(float(prec), 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "decision_threshold": round(float(thresh), 3),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_pixels": int(len(y_true)),
        "positive_pixels": int(np.sum(y_true)),
    }


def train_extended_model():
    output_dir = PROJECT_ROOT / "outputs" / "models" / "weather_convlstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    hw = inspect_hardware()
    device = torch.device(hw["device_selected"])
    print(f"Using device: {device} ({hw.get('gpu_name', 'CPU')})")

    data_dir = PROJECT_ROOT / "data" / "processed" / "weather"
    all_dates = ["20250701", "20250702", "20250703", "20250704", "20250705", "20250706", "20250707"]
    all_data = load_daily_aligned_datasets(data_dir, all_dates)

    train_data = {k: v[:192] if isinstance(v, np.ndarray) and v.shape[0] == 336 else v for k, v in all_data.items()}
    stats = compute_normalization_stats(train_data)

    train_ds = ExtendedSpatiotemporalDataset(all_data, stats, t_start=3, t_end=191, t_in=4)
    val_ds = ExtendedSpatiotemporalDataset(all_data, stats, t_start=192, t_end=239, t_in=4)
    test_ds = ExtendedSpatiotemporalDataset(all_data, stats, t_start=240, t_end=335, t_in=4)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    model = ExtendedWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), kernel_size=3, num_leads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Lead loss weights
    weights = [0.35, 0.30, 0.20, 0.15]
    pos_weight = torch.tensor([stats["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_csi_30": [], "val_csi_60": [], "val_csi_120": [], "val_csi_180": []
    }

    print("\nTraining Extended 4-Head ConvLSTM for 15 epochs...")
    t0 = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, 16):
        model.train()
        total_train_loss = 0.0
        for x, y, val in train_loader:
            x, y, val = x.to(device), y.to(device), val.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = sum(weights[i] * criterion(logits[:, i], y[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_val_probs, all_val_targets = [], []
        with torch.no_grad():
            for x, y, val in val_loader:
                x, y, val = x.to(device), y.to(device), val.to(device)
                logits = model(x)
                loss = sum(weights[i] * criterion(logits[:, i], y[:, i]) for i in range(4))
                total_val_loss += loss.item()
                all_val_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_val_targets.append(y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_probs_arr = np.concatenate(all_val_probs, axis=0)
        val_targets_arr = np.concatenate(all_val_targets, axis=0)

        val_csis = []
        for i in range(4):
            vp = val_probs_arr[:, i].ravel()
            vt = val_targets_arr[:, i].ravel()
            cm = confusion_matrix(vt, (vp >= 0.5).astype(int), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            val_csis.append(round(tp / max(tp + fp + fn, 1), 4))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_csi_30"].append(val_csis[0])
        history["val_csi_60"].append(val_csis[1])
        history["val_csi_120"].append(val_csis[2])
        history["val_csi_180"].append(val_csis[3])

        print(f"Epoch {epoch:2d}/15 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val CSI: +30m={val_csis[0]:.4f}, +60m={val_csis[1]:.4f}, +120m={val_csis[2]:.4f}, +180m={val_csis[3]:.4f}")

    total_time = round(time.time() - t0, 2)
    print(f"Training completed in {total_time}s")

    # Save model weights
    model_save_path = output_dir / "extended_convlstm_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved weights: {model_save_path}")

    # Save history CSV
    hist_csv_path = exp_dir / "training_history_extended.csv"
    with hist_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_csi_30min", "val_csi_60min", "val_csi_120min", "val_csi_180min"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_csi_30"][i],
                history["val_csi_60"][i],
                history["val_csi_120"][i],
                history["val_csi_180"][i],
            ])
    print(f"Saved extended history CSV: {hist_csv_path}")

    # Evaluate on Held-out Test Set
    print("\nEvaluating on Held-Out Test Set (July 6-7, 2025)...")
    test_loader = DataLoader(test_ds, batch_size=12, shuffle=False)
    all_test_probs, all_test_targets = [], []
    with torch.no_grad():
        for x, y, val in test_loader:
            x = x.to(device)
            logits = model(x)
            all_test_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_test_targets.append(y.numpy())

    test_probs = np.concatenate(all_test_probs, axis=0)
    test_targets = np.concatenate(all_test_targets, axis=0)

    test_metrics = {}
    for i, lead_name in enumerate(LEAD_NAMES):
        tp_lead = test_probs[:, i].ravel()
        tt_lead = test_targets[:, i].ravel()
        # Tune threshold on validation for optimal CSI
        best_th, best_csi = 0.5, -1
        for th in np.linspace(0.1, 0.9, 17):
            pred_th = (val_probs_arr[:, i].ravel() >= th).astype(int)
            cm = confusion_matrix(val_targets_arr[:, i].ravel(), pred_th, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            csi = tp / max(tp + fp + fn, 1)
            if csi > best_csi:
                best_csi = csi
                best_th = th

        test_metrics[lead_name] = calculate_metrics_for_lead(tt_lead, tp_lead, thresh=best_th)
        print(f"Test {lead_name} (+{LEAD_STEPS[i]*30}m): CSI={test_metrics[lead_name]['csi']:.4f}, "
              f"POD={test_metrics[lead_name]['pod_recall']:.4f}, FAR={test_metrics[lead_name]['far']:.4f}, "
              f"ROC-AUC={test_metrics[lead_name]['roc_auc']:.4f}, PR-AUC={test_metrics[lead_name]['pr_auc']:.4f}")

    # Save metrics JSON
    metrics_path = output_dir / "extended_convlstm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": "ExtendedWeatherConvLSTM",
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device),
            "training_time_seconds": total_time,
            "test_metrics": test_metrics,
            "history": history,
        }, f, indent=2)
    print(f"Saved extended metrics to: {metrics_path}")


if __name__ == "__main__":
    train_extended_model()
