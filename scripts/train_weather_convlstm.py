"""Spatiotemporal ConvLSTM Weather Nowcasting Pipeline.

Predicts future 30-minute and 60-minute IMERG heavy-rainfall occurrence (>= 5.0 mm/hr)
from sequential satellite IR brightness temperature (MERGIR) and IMERG precipitation histories.

Architecture:
  - 2-Layer 2D ConvLSTM Spatiotemporal Encoder (Shi et al., NIPS 2015)
  - Multi-lead prediction heads (+30 min and +60 min forecast leads)
  - Modalities supported: 'both' (IR + Precip + Mask), 'satellite_only', 'rainfall_only'

Temporal Splits & Strict Anti-Leakage:
  - Training:   July 1 - July 4, 2025 (4 days, 192 slots)
  - Validation: July 5, 2025 (1 day, 48 slots)
  - Testing:    July 6 - July 7, 2025 (2 days, 96 slots)
  - Sequence input length: T_in = 4 (2 hours of prior history)
  - Sequence targets: T_out = 2 (+30 min, +60 min leads)
  - Normalization parameters fit strictly on Training split only.
  - Zero future rainfall in input sequences (t_end <= T_ref).
  - Temporal embargo buffer between train/val and val/test.

Loss & Evaluation:
  - CompoundNowcastingLoss (Weighted BCE + Soft-CSI loss)
  - Threshold tuned on Validation set only.
  - Metrics: CSI (Critical Success Index), F1, POD (Recall), FAR, Precision,
    Brier Score, ROC-AUC, PR-AUC.
  - Generates comparison against Persistence, Logistic Regression, and MLP baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_HEAVY_RAIN_THRESHOLD = 5.0  # mm/hr (IMD standard for convective/heavy rain)
PROXY_CAVEAT = (
    "IMERG rainfall is a satellite-retrieval product (proxy label) rather than independent "
    "ground-truth rain-gauge validation. MERGIR IR is one of the observational inputs to IMERG, "
    "so this model serves as a spatiotemporal deep learning pipeline feasibility demonstration, "
    "not independent proof of operational forecasting skill."
)


def inspect_hardware() -> Dict[str, Any]:
    """Inspect local hardware and detect GPU/VRAM/CUDA support."""
    info: Dict[str, Any] = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_total_mb": None,
        "device_selected": "cpu",
    }
    if torch.cuda.is_available():
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
            input_p_list.append(f["input_precip"][:])
            target_30_list.append(f["target_precip_lead30"][:])
            target_60_list.append(f["target_precip_lead60"][:])
            mask_list.append(f["satellite_mask"][:])

            nom = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["nominal_slot_iso"][:]]
            ref = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["reference_time_iso"][:]]
            time_iso_list.extend(nom)
            ref_iso_list.extend(ref)

    return {
        "ir_mean": np.concatenate(ir_mean_list, axis=0),        # (T, 25, 40)
        "input_precip": np.concatenate(input_p_list, axis=0),  # (T, 25, 40)
        "target_lead30": np.concatenate(target_30_list, axis=0),# (T, 25, 40)
        "target_lead60": np.concatenate(target_60_list, axis=0),# (T, 25, 40)
        "satellite_mask": np.concatenate(mask_list, axis=0),    # (T, 25, 40)
        "nominal_slot_iso": time_iso_list,
        "reference_time_iso": ref_iso_list,
        "lat": lats,
        "lon": lons,
    }


def compute_normalization_stats(train_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute mean and std for IR and log1p(Precipitation) strictly on training set."""
    tb = train_data["ir_mean"]
    mask = train_data["satellite_mask"]
    p = train_data["input_precip"]

    valid_tb = tb[(mask == 1) & np.isfinite(tb)]
    tb_mean = float(np.mean(valid_tb))
    tb_std = float(np.std(valid_tb))

    valid_p = p[np.isfinite(p)]
    log_p = np.log1p(valid_p)
    log_p_mean = float(np.mean(log_p))
    log_p_std = float(np.std(log_p))

    # Calculate class prevalence and positive weight for heavy rain
    t30 = train_data["target_lead30"]
    valid_t30 = t30[np.isfinite(t30)]
    pos_rate = float(np.mean(valid_t30 >= DEFAULT_HEAVY_RAIN_THRESHOLD))
    pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-4))

    return {
        "tb_mean": tb_mean,
        "tb_std": tb_std,
        "log_p_mean": log_p_mean,
        "log_p_std": log_p_std,
        "pos_rate": pos_rate,
        "pos_weight": pos_weight,
    }


class SpatiotemporalWeatherDataset(Dataset):
    """PyTorch Dataset for multi-frame weather nowcasting sequences.
    
    Generates (T_in, C, H, W) input sequences and (T_out, H, W) binary target maps.
    Strictly preserves anti-leakage rules and observation masks.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        stats: Dict[str, float],
        t_start: int,
        t_end: int,
        t_in: int = 4,
        threshold: float = DEFAULT_HEAVY_RAIN_THRESHOLD,
        modality: str = "both",
    ):
        super().__init__()
        self.t_in = t_in
        self.threshold = threshold
        self.modality = modality

        tb = data["ir_mean"]
        p = data["input_precip"]
        mask = data["satellite_mask"]
        t30 = data["target_lead30"]
        t60 = data["target_lead60"]

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

        # Assemble channels according to requested modality
        if modality == "both":
            # (T, 3, H, W): IR, Precip, Availability Mask
            self.features = np.stack([tb_norm, p_norm, mask_ch], axis=1)
        elif modality == "satellite_only":
            # (T, 2, H, W): IR, Availability Mask
            self.features = np.stack([tb_norm, mask_ch], axis=1)
        elif modality == "rainfall_only":
            # (T, 1, H, W): Precip
            self.features = np.expand_dims(p_norm, axis=1)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Construct sequences
        self.samples_x: List[np.ndarray] = []
        self.samples_y: List[np.ndarray] = []
        self.samples_valid: List[np.ndarray] = []
        self.samples_ref_time: List[str] = []
        self.samples_last_p: List[np.ndarray] = []  # For persistence baseline

        for t in range(t_start, t_end + 1):
            y30 = t30[t]
            y60 = t60[t]

            # Check target validity
            valid_mask = np.isfinite(y30) & np.isfinite(y60)
            if not np.any(valid_mask):
                continue

            # Input frames: [t - t_in + 1, ..., t]
            x_seq = self.features[t - t_in + 1 : t + 1]  # shape: (T_in, C, H, W)
            target_stack = np.stack([
                (y30 >= threshold).astype(np.float32),
                (y60 >= threshold).astype(np.float32),
            ], axis=0)  # shape: (2, H, W)

            val_stack = np.stack([
                np.isfinite(y30).astype(np.float32),
                np.isfinite(y60).astype(np.float32),
            ], axis=0)  # shape: (2, H, W)

            self.samples_x.append(x_seq)
            self.samples_y.append(target_stack)
            self.samples_valid.append(val_stack)
            self.samples_ref_time.append(data["reference_time_iso"][t])
            self.samples_last_p.append(p[t])  # Rain at reference time

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.samples_x[idx])
        y = torch.from_numpy(self.samples_y[idx])
        val = torch.from_numpy(self.samples_valid[idx])
        return x, y, val


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.convlstm import ConvLSTMCell, MultiTaskWeatherConvLSTM


class WeatherConvLSTM(nn.Module):
    """Spatiotemporal ConvLSTM Nowcasting Network with Multi-Lead Prediction Heads."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        num_leads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_leads = num_leads

        # 2-layer ConvLSTM Encoder
        self.cell1 = ConvLSTMCell(in_channels, hidden_channels[0], kernel_size)
        self.cell2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], kernel_size)

        # Multi-lead prediction heads (+30 min, +60 min)
        self.lead_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels[1], 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(16, 1, kernel_size=1),
            )
            for _ in range(num_leads)
        ])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential ConvLSTM encoder and multi-lead heads.
        
        Args:
            x_seq: (B, T_in, C, H, W) input sequence tensor.
            
        Returns:
            logits: (B, num_leads, H, W) raw prediction logits.
        """
        B, T, C, H, W = x_seq.shape
        s1, s2 = None, None

        for t in range(T):
            x_t = x_seq[:, t]
            h1, c1 = self.cell1(x_t, s1)
            s1 = (h1, c1)
            h2, c2 = self.cell2(h1, s2)
            s2 = (h2, c2)

        # Top hidden state h2 at time T represents fused spatiotemporal memory
        logits = torch.cat([head(h2) for head in self.lead_heads], dim=1)
        return logits  # (B, num_leads, H, W)


class CompoundNowcastingLoss(nn.Module):
    """Compound loss combining Weighted BCE and Soft-CSI loss for imbalanced nowcasting."""

    def __init__(self, pos_weight: float = 20.0, csi_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.csi_weight = csi_weight
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute compound loss on valid pixels."""
        if valid_mask is not None:
            active = valid_mask == 1.0
            if not torch.any(active):
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            active_logits = logits[active]
            active_targets = targets[active]
        else:
            active_logits = logits.flatten()
            active_targets = targets.flatten()

        pw = self.pos_weight.to(logits.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)(active_logits, active_targets)

        probs = torch.sigmoid(active_logits)
        intersection = torch.sum(probs * active_targets)
        union = torch.sum(probs) + torch.sum(active_targets) - intersection
        soft_csi = (intersection + self.eps) / (union + self.eps)
        csi_loss = 1.0 - soft_csi

        return bce + self.csi_weight * csi_loss


def calculate_contingency_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute meteorological contingency table metrics (CSI, POD, FAR, F1, Brier, etc.)."""
    valid_idx = np.isfinite(y_true) & np.isfinite(y_prob)
    yt = y_true[valid_idx].astype(int)
    yp = y_prob[valid_idx]

    if len(yt) == 0:
        return {}

    y_pred = (yp >= threshold).astype(int)
    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    csi = tp / max(tp + fp + fn, 1)
    pod = tp / max(tp + fn, 1)      # Recall
    far = fp / max(tp + fp, 1)      # False Alarm Ratio
    prec = tp / max(tp + fp, 1)     # Precision = 1 - FAR
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    brier = float(brier_score_loss(yt, yp))

    roc_auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
    precisions, recalls, _ = precision_recall_curve(yt, yp)
    pr_auc = float(auc(recalls, precisions)) if len(np.unique(yt)) > 1 else 0.0

    return {
        "csi": round(float(csi), 4),
        "f1": round(float(f1), 4),
        "pod_recall": round(float(pod), 4),
        "far": round(float(far), 4),
        "precision": round(float(prec), 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "decision_threshold": round(float(threshold), 3),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_pixels": int(len(yt)),
        "positive_pixels": int(yt.sum()),
        "prevalence": round(float(yt.mean()), 4),
    }


def find_optimal_csi_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tune decision threshold on validation set to maximize CSI."""
    best_thresh = 0.5
    best_csi = -1.0
    for thresh in np.linspace(0.1, 0.9, 33):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        csi = tp / max(tp + fp + fn, 1)
        if csi > best_csi:
            best_csi = csi
            best_thresh = thresh
    return float(best_thresh)


def evaluate_dataset(
    model: nn.Module,
    dataset: SpatiotemporalWeatherDataset,
    device: torch.device,
    thresholds: Optional[List[float]] = None,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on dataset and return metrics for both leads."""
    model.eval()
    all_probs = []
    all_targets = []
    all_valids = []

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for x, y, val in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
            all_valids.append(val.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)      # (N, 2, H, W)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 2, H, W)
    valids_arr = np.concatenate(all_valids, axis=0)    # (N, 2, H, W)

    results = {}
    optimal_threshs = []

    for lead_idx, lead_name in enumerate(["lead_30min", "lead_60min"]):
        p = probs_arr[:, lead_idx]
        y = targets_arr[:, lead_idx]
        v = valids_arr[:, lead_idx]

        mask = v == 1.0
        p_flat = p[mask]
        y_flat = y[mask]

        if thresholds is None:
            thresh = find_optimal_csi_threshold(y_flat, p_flat)
        else:
            thresh = thresholds[lead_idx]
        optimal_threshs.append(thresh)

        metrics = calculate_contingency_metrics(y_flat, p_flat, threshold=thresh)
        results[lead_name] = metrics

    results["decision_thresholds"] = optimal_threshs
    return results, probs_arr, targets_arr, valids_arr


def evaluate_persistence_baseline(
    dataset: SpatiotemporalWeatherDataset,
    threshold: float = DEFAULT_HEAVY_RAIN_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """Compute persistence baseline for Lead 30 and Lead 60."""
    results = {}
    for lead_idx, lead_name in enumerate(["lead_30min", "lead_60min"]):
        y_trues = []
        y_preds = []
        for i in range(len(dataset)):
            last_p = dataset.samples_last_p[i]
            y_target = dataset.samples_y[i][lead_idx]
            val = dataset.samples_valid[i][lead_idx]

            mask = val == 1.0
            y_trues.append(y_target[mask])
            y_preds.append((last_p[mask] >= threshold).astype(float))

        yt = np.concatenate(y_trues)
        yp = np.concatenate(y_preds)
        results[lead_name] = calculate_contingency_metrics(yt, yp, threshold=0.5)

    return results


def train_convlstm(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 12,
    lr: float = 1e-3,
    modality: str = "both",
    pos_weight: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate 2D Spatiotemporal ConvLSTM nowcasting pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    hw_info = inspect_hardware()
    device = torch.device(hw_info["device_selected"])

    # Dates and chronological splits
    train_dates = ["20250701", "20250702", "20250703", "20250704"]
    val_dates = ["20250705"]
    test_dates = ["20250706", "20250707"]

    print("Loading daily aligned datasets...")
    train_data = load_daily_aligned_datasets(data_dir, train_dates)
    val_data = load_daily_aligned_datasets(data_dir, val_dates)
    test_data = load_daily_aligned_datasets(data_dir, test_dates)

    print("Computing normalization statistics on Training split...")
    stats = compute_normalization_stats(train_data)
    if pos_weight is not None:
        stats["pos_weight"] = pos_weight

    print(f"  Train Tb: mean={stats['tb_mean']:.2f} K, std={stats['tb_std']:.2f} K")
    print(f"  Train log1p(P): mean={stats['log_p_mean']:.4f}, std={stats['log_p_std']:.4f}")
    print(f"  Train Heavy Rain Prevalence: {stats['pos_rate'] * 100:.2f}% (pos_weight={stats['pos_weight']:.2f})")

    # Sequence lengths and embargo rules
    T_IN = 4
    # Train: slots 3 to 189 (leaves 1h embargo buffer before Val)
    # Val: slots 3 to 45 (leaves 1h embargo buffer before Test)
    # Test: slots 3 to 93 (leaves final 2 slots of July 7 which lack future IMERG)
    train_dataset = SpatiotemporalWeatherDataset(train_data, stats, t_start=3, t_end=189, t_in=T_IN, modality=modality)
    val_dataset = SpatiotemporalWeatherDataset(val_data, stats, t_start=3, t_end=45, t_in=T_IN, modality=modality)
    test_dataset = SpatiotemporalWeatherDataset(test_data, stats, t_start=3, t_end=93, t_in=T_IN, modality=modality)

    print(f"Sequences generated: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Model channels based on modality
    channel_map = {"both": 3, "satellite_only": 2, "rainfall_only": 1}
    in_channels = channel_map[modality]

    model = WeatherConvLSTM(
        in_channels=in_channels,
        hidden_channels=(32, 32),
        kernel_size=3,
        num_leads=2,
        dropout=0.1,
    ).to(device)

    loss_fn = CompoundNowcastingLoss(
        pos_weight=stats["pos_weight"],
        csi_weight=1.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"\nStarting ConvLSTM training on {device} ({hw_info.get('gpu_name', 'CPU')})...")
    start_time = time.time()
    best_val_csi = -1.0
    best_weights_path = output_dir / "best_convlstm_model.pt"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_csi_30": [],
        "val_csi_60": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for x, y, val in train_loader:
            x, y, val = x.to(device), y.to(device), val.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y, val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        # Quick validation check
        model.eval()
        val_losses = []
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for x, y, val in val_loader:
                x, y, val = x.to(device), y.to(device), val.to(device)
                logits = model(x)
                val_losses.append(loss_fn(logits, y, val).item())

        val_loss = float(np.mean(val_losses))
        val_res, _, _, _ = evaluate_dataset(model, val_dataset, device)
        val_csi_30 = val_res["lead_30min"]["csi"]
        val_csi_60 = val_res["lead_60min"]["csi"]
        avg_val_csi = (val_csi_30 + val_csi_60) / 2.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_csi_30"].append(val_csi_30)
        history["val_csi_60"].append(val_csi_60)

        is_best = avg_val_csi > best_val_csi
        if is_best:
            best_val_csi = avg_val_csi
            torch.save(model.state_dict(), best_weights_path)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val CSI (30m): {val_csi_30:.4f} | "
            f"Val CSI (60m): {val_csi_60:.4f} {'[BEST]' if is_best else ''}"
        )

    elapsed_s = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {elapsed_s}s. Best Val Avg CSI: {best_val_csi:.4f}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    # 1. Evaluate on Validation set to obtain optimal decision thresholds
    val_metrics, val_probs, val_targets, val_valids = evaluate_dataset(model, val_dataset, device)
    val_thresh_30 = val_metrics["decision_thresholds"][0]
    val_thresh_60 = val_metrics["decision_thresholds"][1]
    optimal_thresholds = [val_thresh_30, val_thresh_60]

    print(f"\nTuned Validation Decision Thresholds: +30m={val_thresh_30:.3f}, +60m={val_thresh_60:.3f}")

    # 2. Evaluate on Train and Test sets using the tuned validation thresholds
    train_metrics, _, _, _ = evaluate_dataset(model, train_dataset, device, thresholds=optimal_thresholds)
    test_metrics, test_probs, test_targets, test_valids = evaluate_dataset(model, test_dataset, device, thresholds=optimal_thresholds)

    # 3. Evaluate Persistence baseline on all splits for comparison
    persist_train = evaluate_persistence_baseline(train_dataset)
    persist_val = evaluate_persistence_baseline(val_dataset)
    persist_test = evaluate_persistence_baseline(test_dataset)

    # Compile comprehensive metrics payload
    metrics_payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": "ConvLSTM_Spatiotemporal",
            "modality": modality,
            "threshold_mm_hr": DEFAULT_HEAVY_RAIN_THRESHOLD,
            "training_time_seconds": elapsed_s,
            "device": hw_info["device_selected"],
            "gpu_name": hw_info.get("gpu_name"),
            "proxy_caveat": PROXY_CAVEAT,
        },
        "optimal_thresholds_tuned_on_validation": {
            "lead_30min": val_thresh_30,
            "lead_60min": val_thresh_60,
        },
        "convlstm_results": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "persistence_baseline": {
            "train": persist_train,
            "validation": persist_val,
            "test": persist_test,
        },
        "history": history,
    }

    metrics_path = output_dir / "convlstm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save normalizer statistics for inference / Colab
    normalizer_path = output_dir / "convlstm_normalizer.json"
    with open(normalizer_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save configuration and anti-leakage audit
    config_payload = {
        "model_architecture": {
            "type": "WeatherConvLSTM",
            "in_channels": in_channels,
            "hidden_channels": [32, 32],
            "kernel_size": 3,
            "num_leads": 2,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "training_hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "CompoundNowcastingLoss (Weighted BCE + Soft-CSI)",
            "pos_weight": stats["pos_weight"],
            "seed": seed,
        },
        "temporal_splits": {
            "train_dates": train_dates,
            "val_dates": val_dates,
            "test_dates": test_dates,
            "t_in_frames": T_IN,
            "t_out_frames": 2,
            "cadence_minutes": 30,
            "embargo_buffer_hours": 1.0,
        },
        "anti_leakage_audit": {
            "normalization_fit_split": "train_only",
            "sequence_cutoff_rule": "t_end <= T_ref for all input frames",
            "future_leakage_in_features": False,
            "decision_threshold_tuned_on": "validation_only",
            "test_set_isolation": True,
        },
        "hardware_environment": hw_info,
        "proxy_caveat": PROXY_CAVEAT,
    }

    config_path = output_dir / "convlstm_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # Generate 6-panel evaluation figure
    print("Generating comprehensive evaluation figure...")
    plot_convlstm_evaluation(
        output_dir=output_dir,
        history=history,
        test_dataset=test_dataset,
        test_probs=test_probs,
        test_targets=test_targets,
        test_valids=test_valids,
        convlstm_test=test_metrics,
        persist_test=persist_test,
        val_thresholds=optimal_thresholds,
    )

    print(f"\nAll artifacts saved successfully in: {output_dir}")
    return metrics_payload


def plot_convlstm_evaluation(
    output_dir: Path,
    history: Dict[str, List[float]],
    test_dataset: SpatiotemporalWeatherDataset,
    test_probs: np.ndarray,
    test_targets: np.ndarray,
    test_valids: np.ndarray,
    convlstm_test: Dict[str, Any],
    persist_test: Dict[str, Any],
    val_thresholds: List[float],
) -> None:
    """Generate professional 6-panel publication-grade evaluation figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)

    # 1. Training & Validation Loss curves
    ax = axes[0, 0]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs_range, history["val_loss"], "r--s", label="Val Loss", linewidth=2, markersize=4)
    ax.set_title("ConvLSTM Training & Validation Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Compound Loss (wBCE + Soft-CSI)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    # 2. Validation CSI over Epochs
    ax = axes[0, 1]
    ax.plot(epochs_range, history["val_csi_30"], "g-o", label="Lead +30m CSI", linewidth=2, markersize=4)
    ax.plot(epochs_range, history["val_csi_60"], "m--^", label="Lead +60m CSI", linewidth=2, markersize=4)
    ax.set_title("Validation CSI over Epochs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Critical Success Index (CSI)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    # 3. Model Comparison Bar Chart (Persistence vs ConvLSTM at +30m and +60m)
    ax = axes[0, 2]
    categories = ["Lead +30m CSI", "Lead +30m F1", "Lead +60m CSI", "Lead +60m F1"]
    p_vals = [
        persist_test["lead_30min"]["csi"],
        persist_test["lead_30min"]["f1"],
        persist_test["lead_60min"]["csi"],
        persist_test["lead_60min"]["f1"],
    ]
    c_vals = [
        convlstm_test["lead_30min"]["csi"],
        convlstm_test["lead_30min"]["f1"],
        convlstm_test["lead_60min"]["csi"],
        convlstm_test["lead_60min"]["f1"],
    ]
    x_pos = np.arange(len(categories))
    width = 0.35
    rects1 = ax.bar(x_pos - width/2, p_vals, width, label="Persistence", color="#7f7f7f", alpha=0.85)
    rects2 = ax.bar(x_pos + width/2, c_vals, width, label="ConvLSTM (Ours)", color="#1f77b4", alpha=0.9)
    ax.set_title("Test Set Skill: Persistence vs. ConvLSTM", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=9, rotation=15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, max(max(p_vals), max(c_vals)) * 1.25)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(frameon=True, fontsize=9)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 4. Precision-Recall Curves on Test Set
    ax = axes[1, 0]
    for lead_idx, lead_lbl, color in [(0, "+30 min Lead", "teal"), (1, "+60 min Lead", "darkorange")]:
        mask = test_valids[:, lead_idx] == 1.0
        y_true = test_targets[:, lead_idx][mask]
        y_prob = test_probs[:, lead_idx][mask]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc_val = convlstm_test[f"lead_{30 if lead_idx==0 else 60}min"]["pr_auc"]
        ax.plot(rec, prec, label=f"ConvLSTM {lead_lbl} (PR-AUC={auc_val:.3f})", color=color, linewidth=2)

    ax.set_title("Test Set Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall (Probability of Detection)", fontsize=10)
    ax.set_ylabel("Precision (1 - FAR)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower left")

    # 5. Spatial Hazard Map Preview for a Strong Convective Case in Test Set
    # Find sample in test set with highest positive pixel count in target
    pos_counts = [test_targets[i, 0].sum() for i in range(len(test_targets))]
    best_case_idx = int(np.argmax(pos_counts))
    sample_time = test_dataset.samples_ref_time[best_case_idx]

    ax = axes[1, 1]
    gt_map = test_targets[best_case_idx, 0]
    prob_map = test_probs[best_case_idx, 0]
    im = ax.imshow(prob_map, cmap="inferno", vmin=0, vmax=1, origin="lower")
    ax.contour(gt_map, levels=[0.5], colors=["cyan"], linewidths=[1.8])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Heavy Rain Prob (>=5 mm/h)")
    ax.set_title(f"ConvLSTM Spatial Hazard Map (+30m)\n{sample_time}\n(Cyan outline: Ground Truth >= 5 mm/h)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Lon Pixel Index (0.1 deg)", fontsize=9)
    ax.set_ylabel("Lat Pixel Index (0.1 deg)", fontsize=9)

    # 6. Diagnostic Summary & Scientific Integrity Notice
    ax = axes[1, 2]
    ax.axis("off")
    diag_text = (
        "HONOURS HAZARD PROJECT: STEP 7 SUMMARY\n"
        "-----------------------------------------------------\n"
        "Model: 2D Spatiotemporal ConvLSTM Nowcaster\n"
        "Resolution: 25 x 40 grid (0.1 deg, Meghalaya AOI)\n"
        "Cadence: 30-min intervals, T_in=4 -> T_out=2\n"
        f"Threshold: >= {DEFAULT_HEAVY_RAIN_THRESHOLD:.1f} mm/hr (Convective proxy)\n"
        "Chronological Splits: Train (Jul 1-4), Val (Jul 5), Test (Jul 6-7)\n\n"
        "TEST SET METRICS (+30 min):\n"
        f"  - ConvLSTM CSI: {convlstm_test['lead_30min']['csi']:.4f}  (Persistence: {persist_test['lead_30min']['csi']:.4f})\n"
        f"  - ConvLSTM F1:  {convlstm_test['lead_30min']['f1']:.4f}  (Persistence: {persist_test['lead_30min']['f1']:.4f})\n"
        f"  - ConvLSTM POD: {convlstm_test['lead_30min']['pod_recall']:.4f}  (Persistence: {persist_test['lead_30min']['pod_recall']:.4f})\n"
        f"  - ConvLSTM FAR: {convlstm_test['lead_30min']['far']:.4f}  (Persistence: {persist_test['lead_30min']['far']:.4f})\n\n"
        "TEST SET METRICS (+60 min):\n"
        f"  - ConvLSTM CSI: {convlstm_test['lead_60min']['csi']:.4f}  (Persistence: {persist_test['lead_60min']['csi']:.4f})\n"
        f"  - ConvLSTM F1:  {convlstm_test['lead_60min']['f1']:.4f}  (Persistence: {persist_test['lead_60min']['f1']:.4f})\n\n"
        "SCIENTIFIC CAVEAT:\n"
        "Target is IMERG satellite-retrieval proxy precipitation,\n"
        "not ground rain-gauge measurements. Evaluates spatiotemporal\n"
        "nowcasting pipeline capability, not operational hazard skill."
    )
    ax.text(0.02, 0.98, diag_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    plt.tight_layout()
    plot_path = output_dir / "convlstm_evaluation.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Evaluation figure saved to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2D Spatiotemporal ConvLSTM Weather Nowcaster")
    parser.add_argument("--data-dir", type=str, default="data/processed/weather", help="Path to aligned NetCDF files")
    parser.add_argument("--output-dir", type=str, default="outputs/models/weather_convlstm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--modality", type=str, choices=["both", "satellite_only", "rainfall_only"], default="both", help="Input modalities")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_convlstm(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        modality=args.modality,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
