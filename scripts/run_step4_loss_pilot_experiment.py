"""Step 4 Controlled Loss Pilot Experiment.

Trains and evaluates 4 controlled loss configurations on SimVP (Seed 42):
  - Exp A: Baseline (BCE + 0.5 * SmoothL1)
  - Exp B: Intensity-Weighted L1 (BCE + 0.5 * Weighted-L1)
  - Exp C: Multiscale Soft-FSS (BCE + 0.5 * SmoothL1 + 0.5 * Soft-FSS)
  - Exp D: Combined Objective (BCE + 0.5 * Weighted-L1 + 0.5 * Soft-FSS)

Selects checkpoints strictly on validation loss (July 5, 2025).
Evaluates comprehensive validation and locked independent test metrics.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.spatial_losses import CompoundNowcastingLoss
from src.evaluation.spatial_metrics import SpatialVerificationEngine as SVE
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = PROJECT_ROOT / "outputs/experiments/model_research/step4_loss_experiment"
CFG_DIR = EXP_DIR / "EXPERIMENT_CONFIGS"
LOG_DIR = EXP_DIR / "TRAINING_LOGS"
CKPT_DIR = EXP_DIR / "CHECKPOINTS"

for d in [EXP_DIR, CFG_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEADS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
RADII_KM = [10.0, 20.0, 30.0, 50.0]
THRESHOLDS = [0.1, 1.0, 5.0]
GRID_KM = 10.55
SEED = 42

print("=" * 80)
print("STARTING STEP 4 CONTROLLED LOSS PILOT EXPERIMENT")
print(f"Device: {DEVICE} | Seed: {SEED} | Experiment Dir: {EXP_DIR}")
print("=" * 80)


# ==============================================================================
# 1. LOAD DATASETS: TRAIN (38 DAYS), VAL (JULY 5, 2025), TEST (JULY 12-14, 2023)
# ==============================================================================
print("\n>>> Loading Datasets...")
train_dates = (
    [f"202307{d:02d}" for d in range(1, 11)] +
    [f"202407{d:02d}" for d in range(1, 15)] +
    [f"202408{d:02d}" for d in range(5, 15)] +
    [f"202507{d:02d}" for d in range(1, 5)]
)
val_dates = ["20250705"]
test_dates = ["20230712", "20230713", "20230714"]

tr_x, tr_yb, tr_yc, tr_m, tr_t = build_advanced_sequences(train_dates, t_in=4)
val_x, val_yb, val_yc, val_m, val_t = build_advanced_sequences(val_dates, t_in=4)
te_x, te_yb, te_yc, te_m, te_t = build_advanced_sequences(test_dates, t_in=4)

print(f"Dataset Split: Train={len(tr_x)} seqs | Val={len(val_x)} seqs | Test={len(te_x)} seqs")


class WeatherDataset(Dataset):
    def __init__(self, x_list, yb_list, yc_list, m_list):
        self.x = torch.from_numpy(np.stack(x_list, axis=0))
        self.yb = torch.from_numpy(np.stack(yb_list, axis=0))
        self.yc = torch.from_numpy(np.stack(yc_list, axis=0))
        self.m = torch.from_numpy(np.stack(m_list, axis=0))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "yb": self.yb[idx],
            "yc": self.yc[idx],
            "m": self.m[idx],
        }


train_ds = WeatherDataset(tr_x, tr_yb, tr_yc, tr_m)
val_ds = WeatherDataset(val_x, val_yb, val_yc, val_m)
test_ds = WeatherDataset(te_x, te_yb, te_yc, te_m)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


# ==============================================================================
# 2. DEFINE THE FOUR LOSS CONFIGURATIONS
# ==============================================================================
experiment_specs = [
    {
        "id": "ExpA_Baseline",
        "name": "SimVP_ExpA_Baseline",
        "description": "Canonical baseline: BCE occurrence loss + 0.5 * SmoothL1 continuous QPF loss",
        "loss_fn": CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.0, use_weighted_l1=False, use_soft_fss=False),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.0,
        "use_weighted_l1": False,
        "use_soft_fss": False,
    },
    {
        "id": "ExpB_WeightedL1",
        "name": "SimVP_ExpB_IntensityWeightedL1",
        "description": "Intensity-weighted L1: BCE + 0.5 * w(y) * SmoothL1 emphasizing convective rain >= 5 mm/hr",
        "loss_fn": CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.0, use_weighted_l1=True, use_soft_fss=False, threshold=5.0),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.0,
        "use_weighted_l1": True,
        "use_soft_fss": False,
    },
    {
        "id": "ExpC_SoftFSS",
        "name": "SimVP_ExpC_MultiscaleSoftFSS",
        "description": "Multiscale Soft-FSS: BCE + 0.5 * SmoothL1 + 0.5 * Multiscale Soft-FSS (10, 20, 30 km)",
        "loss_fn": CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.5, use_weighted_l1=False, use_soft_fss=True, radii_cells=[1, 2, 3], threshold=5.0),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.5,
        "use_weighted_l1": False,
        "use_soft_fss": True,
    },
    {
        "id": "ExpD_Combined",
        "name": "SimVP_ExpD_CombinedObjective",
        "description": "Combined objective: BCE + 0.5 * Weighted-L1 + 0.5 * Multiscale Soft-FSS",
        "loss_fn": CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.5, use_weighted_l1=True, use_soft_fss=True, radii_cells=[1, 2, 3], threshold=5.0),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.5,
        "use_weighted_l1": True,
        "use_soft_fss": True,
    },
]

# Save experiment configurations JSON
for spec in experiment_specs:
    cfg_path = CFG_DIR / f"{spec['id']}_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "id": spec["id"],
            "name": spec["name"],
            "description": spec["description"],
            "seed": SEED,
            "architecture": "SimVPWeather",
            "hidden_dim": 32,
            "t_in": 4,
            "num_leads": 5,
            "num_translator_blocks": 3,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 15,
            "batch_size": 16,
            "w_bce": spec["w_bce"],
            "w_qpf": spec["w_qpf"],
            "w_fss": spec["w_fss"],
            "use_weighted_l1": spec["use_weighted_l1"],
            "use_soft_fss": spec["use_soft_fss"],
        }, f, indent=2)


# ==============================================================================
# 3. TRAINING & VALIDATION LOOP (STRICT SELECTION ON VALIDATION LOSS)
# ==============================================================================
def train_model(spec: Dict[str, Any]) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5, num_translator_blocks=3).to(DEVICE)
    loss_fn = spec["loss_fn"].to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    epochs = 15
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    training_logs = []

    print(f"\n>>> Training {spec['name']} on {DEVICE}...")
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss_total, t_loss_bce, t_loss_qpf, t_loss_fss = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            bx = batch["x"].to(DEVICE)
            byb = batch["yb"].to(DEVICE)
            byc = batch["yc"].to(DEVICE)
            bm = batch["m"].to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                p_cat, p_qpf = model(bx)
                loss, diags = loss_fn(p_cat, p_qpf, byb, byc, mask=bm)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss_total += diags["loss_total"]
            t_loss_bce += diags["loss_bce"]
            t_loss_qpf += diags["loss_qpf"]
            t_loss_fss += diags["loss_fss"]
            n_batches += 1

        scheduler.step()
        train_summary = {
            "epoch": epoch,
            "train_loss_total": round(t_loss_total / n_batches, 4),
            "train_loss_bce": round(t_loss_bce / n_batches, 4),
            "train_loss_qpf": round(t_loss_qpf / n_batches, 4),
            "train_loss_fss": round(t_loss_fss / n_batches, 4),
        }

        # Validation loop
        model.eval()
        v_loss_total, v_loss_bce, v_loss_qpf, v_loss_fss = 0.0, 0.0, 0.0, 0.0
        v_batches = 0
        val_tp, val_fp, val_fn = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                bx = batch["x"].to(DEVICE)
                byb = batch["yb"].to(DEVICE)
                byc = batch["yc"].to(DEVICE)
                bm = batch["m"].to(DEVICE)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    p_cat, p_qpf = model(bx)
                    v_loss, v_diags = loss_fn(p_cat, p_qpf, byb, byc, mask=bm)

                v_loss_total += v_diags["loss_total"]
                v_loss_bce += v_diags["loss_bce"]
                v_loss_qpf += v_diags["loss_qpf"]
                v_loss_fss += v_diags["loss_fss"]
                v_batches += 1

                # Validation +30m occurrence contingency
                prob0 = torch.sigmoid(p_cat[:, 0]).cpu().numpy()
                true0 = byb[:, 0].cpu().numpy().astype(int)
                b0 = (prob0 >= 0.5).astype(int)
                val_tp += int(np.sum((b0 == 1) & (true0 == 1)))
                val_fp += int(np.sum((b0 == 1) & (true0 == 0)))
                val_fn += int(np.sum((b0 == 0) & (true0 == 1)))

        v_mean_loss = v_loss_total / v_batches
        v_csi_30m = val_tp / max(val_tp + val_fp + val_fn, 1)

        epoch_record = {
            **train_summary,
            "val_loss_total": round(v_mean_loss, 4),
            "val_loss_bce": round(v_loss_bce / v_batches, 4),
            "val_loss_qpf": round(v_loss_qpf / v_batches, 4),
            "val_loss_fss": round(v_loss_fss / v_batches, 4),
            "val_csi_30m": round(v_csi_30m, 4),
        }
        training_logs.append(epoch_record)

        if v_mean_loss < best_val_loss:
            best_val_loss = v_mean_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/15 | TrLoss: {train_summary['train_loss_total']:.4f} | ValLoss: {v_mean_loss:.4f} | Val +30m CSI: {v_csi_30m:.4f}")

    train_duration = time.time() - start_t
    ckpt_path = CKPT_DIR / f"{spec['id']}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}. Saved checkpoint to: {ckpt_path.name}")

    # Write training log CSV
    log_csv = LOG_DIR / f"{spec['id']}_training_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(training_logs[0].keys()))
        writer.writeheader()
        writer.writerows(training_logs)

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    meta = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "train_duration_sec": round(train_duration, 2),
        "ckpt_path": str(ckpt_path),
    }
    return model, training_logs, meta


# ==============================================================================
# 4. COMPREHENSIVE EVALUATION FUNCTION
# ==============================================================================
def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    eval_set_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate model for all horizons, thresholds, circular radii, and diagnostics."""
    all_preds_qpf = []
    all_obs_qpf = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            bx = batch["x"].to(DEVICE)
            byc = batch["yc"]
            bm = batch["m"]

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                _, p_qpf = model(bx)

            all_preds_qpf.append(p_qpf.cpu().float().numpy())
            all_obs_qpf.append(byc.float().numpy())
            all_masks.append(bm.float().numpy())

    preds_arr = np.concatenate(all_preds_qpf, axis=0)  # (N, Leads, H, W)
    obs_arr = np.concatenate(all_obs_qpf, axis=0)      # (N, Leads, H, W)
    mask_arr = np.concatenate(all_masks, axis=0)      # (N, Leads, H, W)

    metric_rows = []
    diag_rows = []

    for h_idx, lead in enumerate(LEADS):
        obs_h = obs_arr[:, h_idx]
        fcst_h = preds_arr[:, h_idx]
        mask_h = mask_arr[:, h_idx]

        # Diagnostics: intensity distributions
        obs_max = float(np.max(obs_h))
        fcst_max = float(np.max(fcst_h))
        fcst_p99 = float(np.percentile(fcst_h, 99.0))
        fcst_p95 = float(np.percentile(fcst_h, 95.0))
        fcst_mean = float(np.mean(fcst_h))
        obs_mean = float(np.mean(obs_h))

        # Continuous metrics
        q_flat = fcst_h.ravel()
        o_flat = obs_h.ravel()
        rmse_val = float(np.sqrt(np.mean((q_flat - o_flat) ** 2)))
        mae_val = float(np.mean(np.abs(q_flat - o_flat)))
        r_val, _ = pearsonr(q_flat, o_flat) if np.std(q_flat) > 1e-6 else (0.0, 1.0)

        diag_rows.append({
            "model_name": model_name,
            "eval_set": eval_set_name,
            "horizon": lead,
            "obs_max_mm_hr": round(obs_max, 2),
            "fcst_max_mm_hr": round(fcst_max, 2),
            "fcst_p99_mm_hr": round(fcst_p99, 2),
            "fcst_p95_mm_hr": round(fcst_p95, 2),
            "fcst_mean_mm_hr": round(fcst_mean, 3),
            "obs_mean_mm_hr": round(obs_mean, 3),
            "intensity_ratio_max": round(fcst_max / max(obs_max, 1e-3), 3),
            "qpf_rmse": round(rmse_val, 4),
            "qpf_mae": round(mae_val, 4),
        })

        for th in THRESHOLDS:
            o_bin = (obs_h >= th).astype(int).ravel()
            f_bin = (fcst_h >= th).astype(int).ravel()
            tp = int(np.sum((o_bin == 1) & (f_bin == 1)))
            fp = int(np.sum((o_bin == 0) & (f_bin == 1)))
            fn = int(np.sum((o_bin == 1) & (f_bin == 0)))
            tn = int(np.sum((o_bin == 0) & (f_bin == 0)))

            csi_pt = float(tp / max(tp + fp + fn, 1))
            pod_pt = float(tp / max(tp + fn, 1))
            far_pt = float(fp / max(tp + fp, 1))

            for r_km in RADII_KM:
                k_cells = SVE.km_to_grid_radius(r_km, GRID_KM)

                # Corrected Circular FSS
                res_cir = SVE.evaluate_sequence_dataset(
                    obs_sequences=obs_h,
                    fcst_sequences=fcst_h,
                    threshold=th,
                    radius_cells=k_cells,
                    mask=mask_h,
                    kernel_type="circular",
                )

                # Circular Neighborhood Contingency
                n_tp, n_fp, n_fn = 0, 0, 0
                for t in range(len(obs_h)):
                    nc = SVE.compute_neighborhood_contingency(
                        obs_field=obs_h[t],
                        fcst_field=fcst_h[t],
                        threshold=th,
                        radius_cells=k_cells,
                        mask=mask_h[t],
                        kernel_type="circular",
                    )
                    n_tp += nc["neighborhood_tp"]
                    n_fp += nc["neighborhood_fp"]
                    n_fn += nc["neighborhood_fn"]

                n_csi = float(n_tp / max(n_tp + n_fp + n_fn, 1))
                n_pod = float(n_tp / max(n_tp + n_fn, 1))
                n_far = float(n_fp / max(n_tp + n_fp, 1))

                metric_rows.append({
                    "model_name": model_name,
                    "eval_set": eval_set_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "grid_radius_cells": k_cells,
                    "circular_fss": res_cir["accumulated_fss"],
                    "circular_n_csi": round(n_csi, 4),
                    "circular_n_pod": round(n_pod, 4),
                    "circular_n_far": round(n_far, 4),
                    "pointwise_csi": round(csi_pt, 4),
                    "pointwise_pod": round(pod_pt, 4),
                    "pointwise_far": round(far_pt, 4),
                    "qpf_rmse": round(rmse_val, 4),
                    "qpf_mae": round(mae_val, 4),
                    "pearson_r": round(r_val, 4),
                })

    return metric_rows, diag_rows


# ==============================================================================
# 5. EXECUTE THE 4 EXPERIMENTAL RUNS & EVALUATE
# ==============================================================================
trained_models: Dict[str, Tuple[nn.Module, Dict[str, Any]]] = {}
all_val_metrics = []
all_test_metrics = []
all_intensity_diags = []
loss_comp_rows = []

for spec in experiment_specs:
    model, history, meta = train_model(spec)
    trained_models[spec["id"]] = (model, meta)

    # Log final epoch loss component breakdown
    last_log = history[-1]
    loss_comp_rows.append({
        "experiment_id": spec["id"],
        "model_name": spec["name"],
        "best_epoch": meta["best_epoch"],
        "best_val_loss": meta["best_val_loss"],
        "train_duration_sec": meta["train_duration_sec"],
        "train_loss_total": last_log["train_loss_total"],
        "train_loss_bce": last_log["train_loss_bce"],
        "train_loss_qpf": last_log["train_loss_qpf"],
        "train_loss_fss": last_log["train_loss_fss"],
        "val_loss_total": last_log["val_loss_total"],
        "val_loss_bce": last_log["val_loss_bce"],
        "val_loss_qpf": last_log["val_loss_qpf"],
        "val_loss_fss": last_log["val_loss_fss"],
        "val_csi_30m": last_log["val_csi_30m"],
    })

    # Evaluate on Validation Set (July 5, 2025)
    print(f"Evaluating {spec['id']} on Validation Set...")
    v_metrics, v_diags = evaluate_model_on_loader(model, val_loader, spec["name"], "Validation_20250705")
    all_val_metrics.extend(v_metrics)
    all_intensity_diags.extend(v_diags)

    # Evaluate on Locked Independent Test Set (July 12-14, 2023)
    print(f"Evaluating {spec['id']} on Locked Test Set...")
    t_metrics, t_diags = evaluate_model_on_loader(model, test_loader, spec["name"], "PrimaryTest_20230712_14")
    all_test_metrics.extend(t_metrics)
    all_intensity_diags.extend(t_diags)


# ==============================================================================
# 6. EXPORT ALL REQUIRED CSV ARTIFACTS
# ==============================================================================
# 1. VALIDATION_RESULTS.csv
p_val_csv = EXP_DIR / "VALIDATION_RESULTS.csv"
with open(p_val_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_val_metrics[0].keys()))
    writer.writeheader()
    writer.writerows(all_val_metrics)
print(f"\nSaved {len(all_val_metrics)} rows to: {p_val_csv.name}")

# 2. LOSS_ABLATION_COMPARISON.csv (Test Set comparison across all 4 configurations)
p_comp_csv = EXP_DIR / "LOSS_ABLATION_COMPARISON.csv"
with open(p_comp_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_test_metrics[0].keys()))
    writer.writeheader()
    writer.writerows(all_test_metrics)
print(f"Saved {len(all_test_metrics)} rows to: {p_comp_csv.name}")

# 3. PREDICTION_INTENSITY_DIAGNOSTICS.csv
p_diag_csv = EXP_DIR / "PREDICTION_INTENSITY_DIAGNOSTICS.csv"
with open(p_diag_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_intensity_diags[0].keys()))
    writer.writeheader()
    writer.writerows(all_intensity_diags)
print(f"Saved {len(all_intensity_diags)} rows to: {p_diag_csv.name}")

# 4. LOSS_COMPONENT_DIAGNOSTICS.csv
p_loss_diag = EXP_DIR / "LOSS_COMPONENT_DIAGNOSTICS.csv"
with open(p_loss_diag, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(loss_comp_rows[0].keys()))
    writer.writeheader()
    writer.writerows(loss_comp_rows)
print(f"Saved {len(loss_comp_rows)} rows to: {p_loss_diag.name}")

print("\n" + "=" * 80)
print("PILOT EXPERIMENT COMPLETE. STOPPING FOR REPORTING.")
print("=" * 80)
