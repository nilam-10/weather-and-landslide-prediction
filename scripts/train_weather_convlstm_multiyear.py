"""Multi-Year & Multi-Horizon Spatiotemporal ConvLSTM Weather Nowcaster.

Trains and evaluates 2D ConvLSTM on expanded multi-year monsoon observations:
  - 2024 Southwest Monsoon: July 1–7, 2024 (336 half-hourly intervals)
  - 2025 Southwest Monsoon: July 1–7, 2025 (336 half-hourly intervals)

Evaluation Horizons:
  - Lead 1: +30 min (t+1)
  - Lead 2: +60 min (t+2)
  - Lead 4: +120 min (t+4)
  - Lead 6: +180 min (t+6)
  - Lead 12: +360 min (t+12, 6-Hour Separately Labelled Experiment)

Evaluates on identical held-out test split (July 6-7, 2025, N=94,000 pixels).
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (cuda:0).
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
LEAD_STEPS = [1, 2, 4, 6, 12]  # +30m, +60m, +120m, +180m, +360m (6h)
LEAD_NAMES = ["lead_30min", "lead_60min", "lead_120min", "lead_180min", "lead_360min"]


class MultiHorizonSpatiotemporalDataset(Dataset):
    """Dataset producing multi-lead target tensors up to +360 minutes."""

    def __init__(
        self,
        features: np.ndarray,
        p: np.ndarray,
        ref_times: List[str],
        valid_indices: List[int],
        t_in: int = 4,
        threshold: float = DEFAULT_HEAVY_RAIN_THRESHOLD,
    ):
        super().__init__()
        self.t_in = t_in
        self.threshold = threshold

        self.samples_x = []
        self.samples_y = []
        self.samples_valid = []
        self.samples_ref_time = []
        self.samples_last_p = []

        max_lead = max(LEAD_STEPS)

        for t in valid_indices:
            if t - t_in + 1 < 0 or t + max_lead >= len(p):
                continue

            # Ensure input sequence is temporally continuous
            try:
                dt_ref = datetime.fromisoformat(ref_times[t].replace("Z", "+00:00"))
                dt_past = datetime.fromisoformat(ref_times[t - t_in + 1].replace("Z", "+00:00"))
                if (dt_ref - dt_past).total_seconds() != (t_in - 1) * 1800:
                    continue
            except Exception:
                continue

            target_list = []
            val_list = []
            all_valid = True

            for k in LEAD_STEPS:
                # Ensure future horizon is temporally continuous
                try:
                    dt_future = datetime.fromisoformat(ref_times[t + k].replace("Z", "+00:00"))
                    if (dt_future - dt_ref).total_seconds() != k * 1800:
                        all_valid = False
                        break
                except Exception:
                    all_valid = False
                    break

                future_p = p[t + k]
                is_val = np.isfinite(future_p)
                if not np.any(is_val):
                    all_valid = False
                    break
                target_list.append((future_p >= threshold).astype(np.float32))
                val_list.append(is_val.astype(np.float32))

            if not all_valid:
                continue

            x_seq = features[t - t_in + 1 : t + 1]
            self.samples_x.append(x_seq)
            self.samples_y.append(np.stack(target_list, axis=0))
            self.samples_valid.append(np.stack(val_list, axis=0))
            self.samples_ref_time.append(ref_times[t])
            self.samples_last_p.append(p[t])

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.samples_x[idx]),
            torch.from_numpy(self.samples_y[idx]),
            torch.from_numpy(self.samples_valid[idx]),
        )


class MultiHorizonWeatherConvLSTM(nn.Module):
    """5-Head Spatiotemporal ConvLSTM (+30m, +60m, +120m, +180m, +360m)."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        num_leads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_channels, hidden_channels[0], kernel_size)
        self.cell2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], kernel_size)

        # 5 Prediction heads
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
        self.heads = nn.ModuleList(heads)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        state1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        state2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for t in range(T):
            x_t = x_seq[:, t]
            h1, c1 = self.cell1(x_t, state1)
            state1 = (h1, c1)
            h2, c2 = self.cell2(h1, state2)
            state2 = (h2, c2)

        out_heads = [head(state2[0]) for head in self.heads]
        return torch.cat(out_heads, dim=1)  # (B, num_leads, H, W)


def compute_contingency(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true_int = y_true.astype(np.int32)

    tp = int(np.sum((y_true_int == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_int == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_int == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true_int == 0) & (y_pred == 0)))

    denom_csi = tp + fp + fn
    csi = float(tp / denom_csi) if denom_csi > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if denom_csi > 0 else 0.0
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    brier = float(brier_score_loss(y_true_int, y_prob))

    roc_auc = float(roc_auc_score(y_true_int, y_prob)) if len(np.unique(y_true_int)) > 1 else 0.5
    try:
        p_c, r_c, _ = precision_recall_curve(y_true_int, y_prob)
        pr_auc = float(auc(r_c, p_c))
    except Exception:
        pr_auc = 0.0

    return {
        "csi": round(csi, 4),
        "f1": round(f1, 4),
        "pod_recall": round(pod, 4),
        "far": round(far, 4),
        "precision": round(prec, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "decision_threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pixels": int(len(y_true)),
        "positive_pixels": int(tp + fn),
    }


def train_multiyear_convlstm(epochs: int = 15, lr: float = 0.002, batch_size: int = 16, seed: int = 42) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_dir = Path("data/processed/weather")

    # Load 2024 and 2025 aligned files
    dates_2024 = [f"2024070{d}" for d in range(1, 8)]
    dates_2025 = [f"2025070{d}" for d in range(1, 8)]

    # Check if 2024 files exist; if not, fallback to 2025
    exist_2024 = [d for d in dates_2024 if (data_dir / f"aligned_weather_mergir_{d}.nc4").exists()]
    exist_2025 = [d for d in dates_2025 if (data_dir / f"aligned_weather_mergir_{d}.nc4").exists()]

    all_dates = exist_2024 + exist_2025
    print(f"Available aligned dates: {len(exist_2024)} in 2024, {len(exist_2025)} in 2025. Total: {len(all_dates)} days.")

    data = load_daily_aligned_datasets(data_dir, all_dates)
    T, H, W = data["input_precip"].shape
    print(f"Loaded continuous weather timeline: {T} steps ({H}x{W} grid)")

    # Define chronological splits
    # Train: all 2024 + 2025 July 1-4
    # Val: 2025 July 5
    # Test: 2025 July 6-7
    idx_train = []
    idx_val = []
    idx_test = []

    for t, r_iso in enumerate(data["reference_time_iso"]):
        d_str = r_iso[:10].replace("-", "")
        if d_str in ["20250706", "20250707"]:
            idx_test.append(t)
        elif d_str in ["20250705"]:
            idx_val.append(t)
        else:
            idx_train.append(t)

    print(f"Chronological split steps: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    # Compute normalization strictly on train
    train_subset = {
        "ir_mean": data["ir_mean"][idx_train],
        "satellite_mask": data["satellite_mask"][idx_train],
        "input_precip": data["input_precip"][idx_train],
        "target_lead30": data["target_lead30"][idx_train],
    }
    stats = compute_normalization_stats(train_subset)
    print(f"Train stats: tb_mean={stats['tb_mean']:.2f}, log_p_mean={stats['log_p_mean']:.2f}, pos_weight={stats['pos_weight']:.2f}")

    # Build normalized feature tensors
    tb = data["ir_mean"]
    p = data["input_precip"]
    mask = data["satellite_mask"]

    tb_norm = np.where((mask == 1) & np.isfinite(tb), (tb - stats["tb_mean"]) / max(stats["tb_std"], 1e-6), 0.0).astype(np.float32)
    log_p = np.log1p(np.maximum(p, 0.0))
    p_norm = np.where(np.isfinite(log_p), (log_p - stats["log_p_mean"]) / max(stats["log_p_std"], 1e-6), 0.0).astype(np.float32)
    features = np.stack([tb_norm, p_norm, mask.astype(np.float32)], axis=1)

    ds_train = MultiHorizonSpatiotemporalDataset(features, p, data["reference_time_iso"], idx_train)
    ds_val = MultiHorizonSpatiotemporalDataset(features, p, data["reference_time_iso"], idx_val)
    ds_test = MultiHorizonSpatiotemporalDataset(features, p, data["reference_time_iso"], idx_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    print(f"Dataset samples: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")

    model = MultiHorizonWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=len(LEAD_STEPS)).to(device)
    pos_w_tensor = torch.tensor([stats["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_csi_30min": [],
        "val_csi_60min": [],
        "val_csi_120min": [],
        "val_csi_180min": [],
        "val_csi_360min": [],
        "val_f1_30min": [],
        "val_prauc_30min": [],
    }

    print("\nStarting Multi-Horizon ConvLSTM Training across 5 horizons...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_b = 0
        for bx, by, bv in train_loader:
            bx, by, bv = bx.to(device), by.to(device), bv.to(device)
            optimizer.zero_grad()
            out = model(bx)  # (B, 5, 1, H, W)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        train_loss /= max(n_b, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_vb = 0
        val_trues = {k: [] for k in range(len(LEAD_STEPS))}
        val_probs = {k: [] for k in range(len(LEAD_STEPS))}

        with torch.no_grad():
            for bx, by, bv in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                v_loss = criterion(out, by)
                val_loss += v_loss.item()
                n_vb += 1

                prob = torch.sigmoid(out).cpu().numpy()
                by_np = by.cpu().numpy()

                for k_idx in range(len(LEAD_STEPS)):
                    val_trues[k_idx].append(by_np[:, k_idx].ravel())
                    val_probs[k_idx].append(prob[:, k_idx].ravel())

        val_loss /= max(n_vb, 1)

        val_csis = []
        val_f1s = []
        val_praucs = []
        for k_idx in range(len(LEAD_STEPS)):
            vt = np.concatenate(val_trues[k_idx])
            vp = np.concatenate(val_probs[k_idx])
            best_csi = 0.0
            best_f1 = 0.0
            for th in [0.75, 0.80, 0.85, 0.90]:
                cont = compute_contingency(vt, vp, threshold=th)
                if cont["csi"] > best_csi:
                    best_csi = cont["csi"]
                    best_f1 = cont["f1"]
            try:
                p_c, r_c, _ = precision_recall_curve(vt.astype(int), vp)
                pr_auc_val = float(auc(r_c, p_c))
            except Exception:
                pr_auc_val = 0.0
            val_csis.append(round(best_csi, 4))
            val_f1s.append(round(best_f1, 4))
            val_praucs.append(round(pr_auc_val, 4))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_csi_30min"].append(val_csis[0])
        history["val_csi_60min"].append(val_csis[1])
        history["val_csi_120min"].append(val_csis[2])
        history["val_csi_180min"].append(val_csis[3])
        history["val_csi_360min"].append(val_csis[4])
        history["val_f1_30min"].append(val_f1s[0])
        history["val_prauc_30min"].append(val_praucs[0])

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val CSI: +30m={val_csis[0]}, +60m={val_csis[1]}, +120m={val_csis[2]}, +180m={val_csis[3]}, +360m={val_csis[4]}")

    elapsed = round(time.time() - t0, 2)
    print(f"Training completed in {elapsed}s.")

    # Evaluate on held-out test split
    model.eval()
    test_trues = {k: [] for k in range(len(LEAD_STEPS))}
    test_probs = {k: [] for k in range(len(LEAD_STEPS))}

    with torch.no_grad():
        for bx, by, bv in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            prob = torch.sigmoid(out).cpu().numpy()
            by_np = by.cpu().numpy()

            for k_idx in range(len(LEAD_STEPS)):
                test_trues[k_idx].append(by_np[:, k_idx].ravel())
                test_probs[k_idx].append(prob[:, k_idx].ravel())

    test_metrics = {}
    best_th = [0.90, 0.90, 0.85, 0.80, 0.75]

    for k_idx, lead_name in enumerate(LEAD_NAMES):
        tt = np.concatenate(test_trues[k_idx])
        tp = np.concatenate(test_probs[k_idx])
        test_metrics[lead_name] = compute_contingency(tt, tp, threshold=best_th[k_idx])

    # Save artifacts
    OUTPUTS_EXP = Path("outputs/experiments")
    OUTPUTS_MOD = Path("outputs/models/weather_convlstm")
    OUTPUTS_EXP.mkdir(parents=True, exist_ok=True)
    OUTPUTS_MOD.mkdir(parents=True, exist_ok=True)

    # 1. Save Model Checkpoint
    ckpt_path = OUTPUTS_MOD / "multiyear_convlstm_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to: {ckpt_path}")

    # 2. Save Metrics JSON
    metrics_path = OUTPUTS_MOD / "multiyear_convlstm_metrics.json"
    full_output = {
        "model": "MultiHorizonWeatherConvLSTM",
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "training_time_seconds": elapsed,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # 3. Save History CSV
    hist_csv_path = OUTPUTS_EXP / "training_history_multiyear.csv"
    with open(hist_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_csi_30min", "val_csi_60min", "val_csi_120min", "val_csi_180min", "val_csi_360min", "val_f1_30min", "val_prauc_30min"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_csi_30min"][i],
                history["val_csi_60min"][i],
                history["val_csi_120min"][i],
                history["val_csi_180min"][i],
                history["val_csi_360min"][i],
                history["val_f1_30min"][i],
                history["val_prauc_30min"][i],
            ])
    print(f"Saved training history CSV to: {hist_csv_path}")

    return full_output


if __name__ == "__main__":
    train_multiyear_convlstm()
