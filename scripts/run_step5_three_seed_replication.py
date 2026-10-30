"""Step 5: Three-Seed Replication Experiment — SimVP Weather Forecasting.

Executes a balanced 2x3 factorial design comparing:
  - Configuration Exp A: Canonical Baseline (BCE + 0.5 * SmoothL1)
  - Configuration Exp D: Combined Objective (BCE + 0.5 * Intensity-Weighted L1 + 0.5 * Multiscale Soft-FSS)
Across Seeds: 42, 123, 999.

Evaluates:
  - Validation Split: July 5, 2025 (33 sequences) — Primary basis for model selection
  - Locked Independent Test Split: July 12-14, 2023 (129 sequences) — Confirmatory diagnostics
"""
from __future__ import annotations

import csv
import json
import os
import random
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
EXP_DIR = PROJECT_ROOT / "outputs/experiments/model_research/step5_three_seed_replication"
CFG_DIR = EXP_DIR / "EXPERIMENT_CONFIGS"
LOG_DIR = EXP_DIR / "TRAINING_LOGS"
CKPT_DIR = EXP_DIR / "CHECKPOINTS"

for d in [EXP_DIR, CFG_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEADS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
RADII_KM = [10.0, 20.0, 30.0, 50.0]
THRESHOLDS = [0.1, 1.0, 5.0]
GRID_KM = 10.55
SEEDS = [42, 123, 999]

print("=" * 80)
print("STARTING STEP 5 THREE-SEED REPLICATION EXPERIMENT")
print(f"Device: {DEVICE} | Seeds: {SEEDS} | Output Directory: {EXP_DIR}")
print("=" * 80)


# ==============================================================================
# 1. LOAD DATASETS: TRAIN (38 DAYS), VAL (JULY 5, 2025), TEST (JULY 12-14, 2023)
# ==============================================================================
print("\n>>> Loading Verified Multi-Year Datasets...")
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

val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


# ==============================================================================
# 2. DEFINE REPLICATION RUN MATRIX (6 RUNS)
# ==============================================================================
run_specs = []
for seed in SEEDS:
    # Exp A: Baseline Control
    run_specs.append({
        "run_id": f"ExpA_Seed{seed}",
        "config_id": "ExpA",
        "config_name": "Baseline Control",
        "seed": seed,
        "loss_fn_builder": lambda: CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.0, use_weighted_l1=False, use_soft_fss=False),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.0,
        "use_weighted_l1": False,
        "use_soft_fss": False,
    })
    # Exp D: Combined Objective
    run_specs.append({
        "run_id": f"ExpD_Seed{seed}",
        "config_id": "ExpD",
        "config_name": "Combined Objective",
        "seed": seed,
        "loss_fn_builder": lambda: CompoundNowcastingLoss(w_bce=1.0, w_qpf=0.5, w_fss=0.5, use_weighted_l1=True, use_soft_fss=True, radii_cells=[1, 2, 3], threshold=5.0),
        "w_bce": 1.0,
        "w_qpf": 0.5,
        "w_fss": 0.5,
        "use_weighted_l1": True,
        "use_soft_fss": True,
    })

# Save configuration JSONs
for spec in run_specs:
    cfg_file = CFG_DIR / f"{spec['run_id']}_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": spec["run_id"],
            "config_id": spec["config_id"],
            "config_name": spec["config_name"],
            "seed": spec["seed"],
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
# 3. TRAINING FUNCTION WITH DETERMINISTIC SEEDING
# ==============================================================================
def train_single_run(spec: Dict[str, Any]) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Use reproducible data loader generator
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=g)

    model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5, num_translator_blocks=3).to(DEVICE)
    loss_fn = spec["loss_fn_builder"]().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    epochs = 15
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    training_logs = []

    print(f"\n>>> Running {spec['run_id']} ({spec['config_name']}, Seed {seed}) on {DEVICE}...")
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

        # Validation loop (July 5, 2025)
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
            "run_id": spec["run_id"],
            "seed": seed,
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
    ckpt_path = CKPT_DIR / f"{spec['run_id']}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Completed in {train_duration:.1f}s | Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}. Saved to: {ckpt_path.name}")

    # Write training log CSV
    log_csv = LOG_DIR / f"{spec['run_id']}_training_log.csv"
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
# 4. COMPREHENSIVE EVALUATION FUNCTION (ALL METRICS & INTENSITY DIAGNOSTICS)
# ==============================================================================
def evaluate_run_on_cohort(
    model: nn.Module,
    loader: DataLoader,
    run_spec: Dict[str, Any],
    eval_set_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Calculate QPF, detection contingency, circular FSS, and intensity diagnostics."""
    all_preds_qpf = []
    all_preds_cat = []
    all_obs_qpf = []
    all_obs_bin = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            bx = batch["x"].to(DEVICE)
            byc = batch["yc"]
            byb = batch["yb"]
            bm = batch["m"]

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                p_cat, p_qpf = model(bx)

            all_preds_qpf.append(p_qpf.cpu().float().numpy())
            all_preds_cat.append(torch.sigmoid(p_cat).cpu().float().numpy())
            all_obs_qpf.append(byc.float().numpy())
            all_obs_bin.append(byb.float().numpy())
            all_masks.append(bm.float().numpy())

    preds_arr = np.concatenate(all_preds_qpf, axis=0)  # (N, Leads, H, W)
    probs_arr = np.concatenate(all_preds_cat, axis=0)  # (N, Leads, H, W)
    obs_arr = np.concatenate(all_obs_qpf, axis=0)      # (N, Leads, H, W)
    bin_arr = np.concatenate(all_obs_bin, axis=0)      # (N, Leads, H, W)
    mask_arr = np.concatenate(all_masks, axis=0)      # (N, Leads, H, W)

    spatial_rows = []
    diagnostic_rows = []
    detection_rows = []

    for h_idx, lead in enumerate(LEADS):
        obs_h = obs_arr[:, h_idx]
        fcst_h = preds_arr[:, h_idx]
        mask_h = mask_arr[:, h_idx]
        prob_h = probs_arr[:, h_idx]

        # Valid pixel masking
        val_idx = mask_h > 0
        o_val = obs_h[val_idx]
        f_val = fcst_h[val_idx]

        # 1. QPF Continuous Metrics
        rmse_val = float(np.sqrt(np.mean((f_val - o_val) ** 2)))
        mae_val = float(np.mean(np.abs(f_val - o_val)))
        r_val, _ = pearsonr(f_val, o_val) if np.std(f_val) > 1e-6 else (0.0, 1.0)
        mean_bias = float(np.mean(f_val) - np.mean(o_val))

        # 2. Intensity Diagnostics
        obs_max = float(np.max(o_val)) if len(o_val) > 0 else 0.0
        fcst_max = float(np.max(f_val)) if len(f_val) > 0 else 0.0
        fcst_p95 = float(np.percentile(f_val, 95.0)) if len(f_val) > 0 else 0.0
        fcst_p99 = float(np.percentile(f_val, 99.0)) if len(f_val) > 0 else 0.0
        obs_p95 = float(np.percentile(o_val, 95.0)) if len(o_val) > 0 else 0.0
        obs_p99 = float(np.percentile(o_val, 99.0)) if len(o_val) > 0 else 0.0

        freq_obs_5mm = int(np.sum(o_val >= 5.0))
        freq_fcst_5mm = int(np.sum(f_val >= 5.0))
        freq_fcst_10mm = int(np.sum(f_val >= 10.0))
        freq_fcst_20mm = int(np.sum(f_val >= 20.0))
        heavy_rain_freq_bias = float(freq_fcst_5mm / max(freq_obs_5mm, 1))
        peak_intensity_bias = float(fcst_max - obs_max)

        # Matched convective core displacement
        core_displacements = []
        for t in range(len(obs_h)):
            o_frame = obs_h[t] * mask_h[t]
            f_frame = fcst_h[t] * mask_h[t]
            if np.max(o_frame) >= 5.0 and np.max(f_frame) >= 5.0:
                o_y, o_x = np.unravel_index(np.argmax(o_frame), o_frame.shape)
                f_y, f_x = np.unravel_index(np.argmax(f_frame), f_frame.shape)
                # Euclidean distance in km
                dist_km = np.sqrt(((o_y - f_y) * 11.08) ** 2 + ((o_x - f_x) * 10.03) ** 2)
                core_displacements.append(dist_km)
        mean_core_disp_km = float(np.mean(core_displacements)) if core_displacements else -1.0

        diag_record = {
            "run_id": run_spec["run_id"],
            "config_id": run_spec["config_id"],
            "config_name": run_spec["config_name"],
            "seed": run_spec["seed"],
            "eval_set": eval_set_name,
            "horizon": lead,
            "obs_max_mm_hr": round(obs_max, 2),
            "fcst_max_mm_hr": round(fcst_max, 2),
            "peak_intensity_bias": round(peak_intensity_bias, 2),
            "fcst_p95_mm_hr": round(fcst_p95, 2),
            "fcst_p99_mm_hr": round(fcst_p99, 2),
            "obs_p95_mm_hr": round(obs_p95, 2),
            "obs_p99_mm_hr": round(obs_p99, 2),
            "freq_obs_ge5mm": freq_obs_5mm,
            "freq_fcst_ge5mm": freq_fcst_5mm,
            "freq_fcst_ge10mm": freq_fcst_10mm,
            "freq_fcst_ge20mm": freq_fcst_20mm,
            "heavy_rain_freq_bias": round(heavy_rain_freq_bias, 4),
            "matched_core_disp_km": round(mean_core_disp_km, 2),
            "qpf_rmse": round(rmse_val, 4),
            "qpf_mae": round(mae_val, 4),
            "pearson_r": round(r_val, 4),
            "mean_rainfall_bias": round(mean_bias, 4),
        }
        diagnostic_rows.append(diag_record)

        # 3. Detection Contingency at 5.0 mm/hr
        o_bin_5 = (o_val >= 5.0).astype(int)
        f_bin_5 = (f_val >= 5.0).astype(int)
        tp = int(np.sum((o_bin_5 == 1) & (f_bin_5 == 1)))
        fp = int(np.sum((o_bin_5 == 0) & (f_bin_5 == 1)))
        fn = int(np.sum((o_bin_5 == 1) & (f_bin_5 == 0)))
        tn = int(np.sum((o_bin_5 == 0) & (f_bin_5 == 0)))

        csi_pt = float(tp / max(tp + fp + fn, 1))
        pod_pt = float(tp / max(tp + fn, 1))
        far_pt = float(fp / max(tp + fp, 1))
        precision_pt = float(tp / max(tp + fp, 1))
        recall_pt = pod_pt

        detection_rows.append({
            "run_id": run_spec["run_id"],
            "config_id": run_spec["config_id"],
            "config_name": run_spec["config_name"],
            "seed": run_spec["seed"],
            "eval_set": eval_set_name,
            "horizon": lead,
            "threshold_mm_hr": 5.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pointwise_csi": round(csi_pt, 4),
            "pointwise_pod": round(pod_pt, 4),
            "pointwise_far": round(far_pt, 4),
            "precision": round(precision_pt, 4),
            "recall": round(recall_pt, 4),
            "qpf_rmse": round(rmse_val, 4),
            "qpf_mae": round(mae_val, 4),
            "pearson_r": round(r_val, 4),
            "mean_rainfall_bias": round(mean_bias, 4),
        })

        # 4. Spatial Verification (FSS across radii & thresholds)
        for th in THRESHOLDS:
            for r_km in RADII_KM:
                k_cells = SVE.km_to_grid_radius(r_km, GRID_KM)
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

                spatial_rows.append({
                    "run_id": run_spec["run_id"],
                    "config_id": run_spec["config_id"],
                    "config_name": run_spec["config_name"],
                    "seed": run_spec["seed"],
                    "eval_set": eval_set_name,
                    "horizon": lead,
                    "threshold_mm_hr": th,
                    "neighborhood_radius_km": r_km,
                    "grid_radius_cells": k_cells,
                    "circular_fss": res_cir["accumulated_fss"],
                    "circular_n_csi": round(n_csi, 4),
                    "circular_n_pod": round(n_pod, 4),
                    "circular_n_far": round(n_far, 4),
                    "qpf_rmse": round(rmse_val, 4),
                    "qpf_mae": round(mae_val, 4),
                })

    return spatial_rows, diagnostic_rows, detection_rows


# ==============================================================================
# 5. EXECUTE 6 REPLICATION RUNS & EVALUATION
# ==============================================================================
all_val_spatial = []
all_val_diags = []
all_val_detection = []

all_test_spatial = []
all_test_diags = []
all_test_detection = []

run_metas = []

for spec in run_specs:
    model, history, meta = train_single_run(spec)
    run_metas.append({**spec, **meta})

    # Validation evaluation (July 5, 2025)
    print(f"Evaluating {spec['run_id']} on Validation Set...")
    v_spat, v_diag, v_det = evaluate_run_on_cohort(model, val_loader, spec, "Validation_20250705")
    all_val_spatial.extend(v_spat)
    all_val_diags.extend(v_diag)
    all_val_detection.extend(v_det)

    # Locked Independent Test evaluation (July 12-14, 2023)
    print(f"Evaluating {spec['run_id']} on Confirmatory Test Set...")
    t_spat, t_diag, t_det = evaluate_run_on_cohort(model, test_loader, spec, "ConfirmatoryTest_20230712_14")
    all_test_spatial.extend(t_spat)
    all_test_diags.extend(t_diag)
    all_test_detection.extend(t_det)


# ==============================================================================
# 6. STATISTICAL AGGREGATION & PAIRED SEED COMPARISON
# ==============================================================================
# 1. Validation Results Per Seed CSV
p_val_per_seed = EXP_DIR / "VALIDATION_RESULTS_PER_SEED.csv"
with open(p_val_per_seed, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_val_detection[0].keys()))
    writer.writeheader()
    writer.writerows(all_val_detection)
print(f"\nSaved {len(all_val_detection)} rows to: {p_val_per_seed.name}")

# 2. Validation Mean & Std across Seeds
import pandas as pd
df_val_det = pd.DataFrame(all_val_detection)
metrics_to_agg = ["pointwise_csi", "pointwise_pod", "pointwise_far", "precision", "recall", "qpf_rmse", "qpf_mae", "pearson_r", "mean_rainfall_bias"]

mean_std_rows = []
for config in ["ExpA", "ExpD"]:
    for lead in LEADS:
        sub = df_val_det[(df_val_det["config_id"] == config) & (df_val_det["horizon"] == lead)]
        row = {
            "config_id": config,
            "config_name": "Baseline Control" if config == "ExpA" else "Combined Objective",
            "eval_set": "Validation_20250705",
            "horizon": lead,
            "threshold_mm_hr": 5.0,
            "n_seeds": len(sub),
        }
        for m in metrics_to_agg:
            row[f"{m}_mean"] = round(float(sub[m].mean()), 4)
            row[f"{m}_std"] = round(float(sub[m].std(ddof=1)), 4) if len(sub) > 1 else 0.0
        mean_std_rows.append(row)

p_val_agg = EXP_DIR / "VALIDATION_RESULTS_MEAN_STD.csv"
with open(p_val_agg, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(mean_std_rows[0].keys()))
    writer.writeheader()
    writer.writerows(mean_std_rows)
print(f"Saved {len(mean_std_rows)} rows to: {p_val_agg.name}")

# 3. Paired Seed Differences (Exp D minus Exp A for matching seeds)
# Also include Circular FSS at 30 km radius
df_val_spat = pd.DataFrame(all_val_spatial)
paired_rows = []

for seed in SEEDS:
    for lead in LEADS:
        a_det = df_val_det[(df_val_det["config_id"] == "ExpA") & (df_val_det["seed"] == seed) & (df_val_det["horizon"] == lead)].iloc[0]
        d_det = df_val_det[(df_val_det["config_id"] == "ExpD") & (df_val_det["seed"] == seed) & (df_val_det["horizon"] == lead)].iloc[0]

        a_fss = df_val_spat[(df_val_spat["config_id"] == "ExpA") & (df_val_spat["seed"] == seed) & (df_val_spat["horizon"] == lead) & (df_val_spat["threshold_mm_hr"] == 5.0) & (df_val_spat["neighborhood_radius_km"] == 30.0)].iloc[0]["circular_fss"]
        d_fss = df_val_spat[(df_val_spat["config_id"] == "ExpD") & (df_val_spat["seed"] == seed) & (df_val_spat["horizon"] == lead) & (df_val_spat["threshold_mm_hr"] == 5.0) & (df_val_spat["neighborhood_radius_km"] == 30.0)].iloc[0]["circular_fss"]

        paired_rows.append({
            "seed": seed,
            "eval_set": "Validation_20250705",
            "horizon": lead,
            "threshold_mm_hr": 5.0,
            "expA_pointwise_csi": a_det["pointwise_csi"],
            "expD_pointwise_csi": d_det["pointwise_csi"],
            "delta_pointwise_csi": round(d_det["pointwise_csi"] - a_det["pointwise_csi"], 4),
            "expA_circular_fss_30km": a_fss,
            "expD_circular_fss_30km": d_fss,
            "delta_circular_fss_30km": round(d_fss - a_fss, 4),
            "expA_qpf_rmse": a_det["qpf_rmse"],
            "expD_qpf_rmse": d_det["qpf_rmse"],
            "delta_qpf_rmse": round(d_det["qpf_rmse"] - a_det["qpf_rmse"], 4),
            "expA_pearson_r": a_det["pearson_r"],
            "expD_pearson_r": d_det["pearson_r"],
            "delta_pearson_r": round(d_det["pearson_r"] - a_det["pearson_r"], 4),
        })

# Compute mean paired difference across the 3 seeds
for lead in LEADS:
    sub_p = [r for r in paired_rows if r["horizon"] == lead and isinstance(r["seed"], int)]
    mean_delta_csi = round(float(np.mean([r["delta_pointwise_csi"] for r in sub_p])), 4)
    std_delta_csi = round(float(np.std([r["delta_pointwise_csi"] for r in sub_p], ddof=1)), 4)
    mean_delta_fss = round(float(np.mean([r["delta_circular_fss_30km"] for r in sub_p])), 4)
    std_delta_fss = round(float(np.std([r["delta_circular_fss_30km"] for r in sub_p], ddof=1)), 4)
    mean_delta_rmse = round(float(np.mean([r["delta_qpf_rmse"] for r in sub_p])), 4)

    paired_rows.append({
        "seed": "Mean_3_Seeds",
        "eval_set": "Validation_20250705",
        "horizon": lead,
        "threshold_mm_hr": 5.0,
        "expA_pointwise_csi": round(float(np.mean([r["expA_pointwise_csi"] for r in sub_p])), 4),
        "expD_pointwise_csi": round(float(np.mean([r["expD_pointwise_csi"] for r in sub_p])), 4),
        "delta_pointwise_csi": mean_delta_csi,
        "expA_circular_fss_30km": round(float(np.mean([r["expA_circular_fss_30km"] for r in sub_p])), 4),
        "expD_circular_fss_30km": round(float(np.mean([r["expD_circular_fss_30km"] for r in sub_p])), 4),
        "delta_circular_fss_30km": mean_delta_fss,
        "expA_qpf_rmse": round(float(np.mean([r["expA_qpf_rmse"] for r in sub_p])), 4),
        "expD_qpf_rmse": round(float(np.mean([r["expD_qpf_rmse"] for r in sub_p])), 4),
        "delta_qpf_rmse": mean_delta_rmse,
        "expA_pearson_r": round(float(np.mean([r["expA_pearson_r"] for r in sub_p])), 4),
        "expD_pearson_r": round(float(np.mean([r["expD_pearson_r"] for r in sub_p])), 4),
        "delta_pearson_r": round(float(np.mean([r["delta_pearson_r"] for r in sub_p])), 4),
    })

p_paired_csv = EXP_DIR / "PAIRED_SEED_COMPARISON.csv"
with open(p_paired_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
    writer.writeheader()
    writer.writerows(paired_rows)
print(f"Saved {len(paired_rows)} rows to: {p_paired_csv.name}")

# 4. PREDICTION_INTENSITY_DIAGNOSTICS.csv (Validation + Test)
p_diag_all = EXP_DIR / "PREDICTION_INTENSITY_DIAGNOSTICS.csv"
combined_diags = all_val_diags + all_test_diags
with open(p_diag_all, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(combined_diags[0].keys()))
    writer.writeheader()
    writer.writerows(combined_diags)
print(f"Saved {len(combined_diags)} rows to: {p_diag_all.name}")

# 5. SPATIAL_VERIFICATION_RESULTS.csv (Validation + Test across all radii)
p_spat_all = EXP_DIR / "SPATIAL_VERIFICATION_RESULTS.csv"
combined_spat = all_val_spatial + all_test_spatial
with open(p_spat_all, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(combined_spat[0].keys()))
    writer.writeheader()
    writer.writerows(combined_spat)
print(f"Saved {len(combined_spat)} rows to: {p_spat_all.name}")

# 6. TEST_DIAGNOSTICS_PER_SEED.csv (Confirmatory test set metrics per seed)
p_test_diag = EXP_DIR / "TEST_DIAGNOSTICS_PER_SEED.csv"
with open(p_test_diag, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_test_detection[0].keys()))
    writer.writeheader()
    writer.writerows(all_test_detection)
print(f"Saved {len(all_test_detection)} rows to: {p_test_diag.name}")

print("\n" + "=" * 80)
print("STEP 5 THREE-SEED REPLICATION COMPLETE.")
print("=" * 80)
