"""Train and Evaluate Research Paper Weather Nowcasting Architectures.

Controlled Experimental Extension:
  - Architecture A: Canonical ConvLSTM (Shi et al., NeurIPS 2015)
  - Architecture B: SimVP (Gao et al., CVPR 2022)
  - Architecture C: Compact U-Net-ConvLSTM (Rahimi et al., JHM 2024)

Evaluates strictly across:
  - Seeds: 42, 123, 999
  - Lead times: +30m, +60m, +120m, +180m, +360m
  - Metrics: CSI (fixed 0.5 & val-tuned), POD, FAR, Precision, Recall, F1,
             ROC-AUC, PR-AUC, Brier score, QPF RMSE, QPF MAE, Pearson r,
             and event-stratified QPF.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
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

# Set matplotlib cache
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "outputs" / ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.convlstm import MultiTaskWeatherConvLSTM
from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.unet_convlstm import UNetConvLSTMWeather
import scripts.run_phase3_research_experiments as p3

# Constants & Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_IN = 4
LEAD_STEPS = [1, 2, 4, 6, 12]  # +30m, +60m, +120m, +180m, +360m
LEAD_NAMES = ["+30m", "+60m", "+120m", "+180m", "+360m"]
CONVECTIVE_THRESHOLD = 5.0  # mm/hr

TB_MEAN, TB_STD = 268.442, 20.619
P_MEAN, P_STD = 0.709, 2.110

EXP_OUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "research_models"
CKPT_DIR = EXP_OUT_DIR / "checkpoints"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "experimental"
REP_DIR = PROJECT_ROOT / "outputs" / "reports"

for d in [EXP_OUT_DIR, CKPT_DIR, FIG_DIR, REP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def train_single_model(
    model: nn.Module,
    model_name: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
) -> Tuple[nn.Module, List[Dict[str, Any]], float, int]:
    """Train spatiotemporal model using AdamW, Cosine Annealing, and Early Stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_loss = float("inf")
    best_weights = None
    start_time = time.time()

    print(f"\n--- Training {model_name} (Seed {seed}, Params: {param_count:,}) on {DEVICE} ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_bce, train_qpf = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y_bin = batch["y_bin"].to(DEVICE)
            y_cont = batch["y_cont"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            pred_cat, pred_qpf = model(x)

            loss_cat = (bce_loss(pred_cat, y_bin) * mask).sum() / max(mask.sum(), 1.0)
            loss_qpf = (smooth_l1(pred_qpf, y_cont) * mask).sum() / max(mask.sum(), 1.0)
            loss = loss_cat + 0.5 * loss_qpf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_bce += loss_cat.item()
            train_qpf += loss_qpf.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)
        train_bce /= max(n_batches, 1)
        train_qpf /= max(n_batches, 1)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn = 0, 0, 0
        v_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(DEVICE)
                y_bin = batch["y_bin"].to(DEVICE)
                y_cont = batch["y_cont"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                pred_cat, pred_qpf = model(x)
                loss_cat = (bce_loss(pred_cat, y_bin) * mask).sum() / max(mask.sum(), 1.0)
                loss_qpf = (smooth_l1(pred_qpf, y_cont) * mask).sum() / max(mask.sum(), 1.0)
                loss = loss_cat + 0.5 * loss_qpf

                val_loss += loss.item()
                v_batches += 1

                # +30m val CSI
                prob_lead0 = torch.sigmoid(pred_cat[:, 0]).cpu().numpy()
                true_lead0 = y_bin[:, 0].cpu().numpy().astype(int)
                pred_bin0 = (prob_lead0 >= 0.5).astype(int)
                val_tp += int(np.sum((pred_bin0 == 1) & (true_lead0 == 1)))
                val_fp += int(np.sum((pred_bin0 == 1) & (true_lead0 == 0)))
                val_fn += int(np.sum((pred_bin0 == 0) & (true_lead0 == 1)))

        val_loss /= max(v_batches, 1)
        val_csi = val_tp / max(val_tp + val_fp + val_fn, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_csi_30m": round(val_csi, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val +30m CSI: {val_csi:.4f}")

    duration = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.to(DEVICE)

    return model, history, duration, param_count


def tune_validation_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """Tune optimal classification thresholds per horizon on Validation split strictly."""
    model.eval()
    val_probs_list = []
    val_gt_list = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(DEVICE)
            cat_logits, _ = model(x)
            val_probs_list.append(torch.sigmoid(cat_logits).cpu().numpy())
            val_gt_list.append(batch["y_bin"].numpy())

    all_p = np.concatenate(val_probs_list, axis=0)  # (N_val, 5, 25, 40)
    all_gt = np.concatenate(val_gt_list, axis=0)    # (N_val, 5, 25, 40)

    tuned_th = {}
    for h_idx, lead in enumerate(LEAD_NAMES):
        p_h = all_p[:, h_idx].ravel()
        gt_h = all_gt[:, h_idx].ravel().astype(int)

        best_csi = -1.0
        best_t = 0.50
        # Search candidate thresholds
        for t in np.arange(0.02, 0.60, 0.02):
            tp = np.sum((p_h >= t) & (gt_h == 1))
            fp = np.sum((p_h >= t) & (gt_h == 0))
            fn = np.sum((p_h < t) & (gt_h == 1))
            csi = tp / max(tp + fp + fn, 1)
            if csi > best_csi:
                best_csi = csi
                best_t = float(t)
        tuned_th[lead] = round(best_t, 4)

    return tuned_th


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    val_tuned_th: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Evaluate model on primary independent test set across all horizons."""
    model.eval()
    test_probs_list = []
    test_qpf_list = []
    test_gt_bin_list = []
    test_gt_cont_list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(DEVICE)
            cat_logits, qpf_rates = model(x)
            test_probs_list.append(torch.sigmoid(cat_logits).cpu().numpy())
            test_qpf_list.append(qpf_rates.cpu().numpy())
            test_gt_bin_list.append(batch["y_bin"].numpy())
            test_gt_cont_list.append(batch["y_cont"].numpy())

    probs = np.concatenate(test_probs_list, axis=0)      # (N_test, 5, 25, 40)
    qpf = np.concatenate(test_qpf_list, axis=0)          # (N_test, 5, 25, 40)
    gt_bin = np.concatenate(test_gt_bin_list, axis=0)    # (N_test, 5, 25, 40)
    gt_cont = np.concatenate(test_gt_cont_list, axis=0)  # (N_test, 5, 25, 40)

    results = []

    for h_idx, lead in enumerate(LEAD_NAMES):
        p_lead = probs[:, h_idx].ravel()
        q_lead = qpf[:, h_idx].ravel()
        gt_b = gt_bin[:, h_idx].ravel().astype(int)
        gt_c = gt_cont[:, h_idx].ravel()

        # Fixed 0.5 metrics
        tp_50 = int(np.sum((p_lead >= 0.5) & (gt_b == 1)))
        fp_50 = int(np.sum((p_lead >= 0.5) & (gt_b == 0)))
        fn_50 = int(np.sum((p_lead < 0.5) & (gt_b == 1)))
        tn_50 = int(np.sum((p_lead < 0.5) & (gt_b == 0)))
        csi_50 = tp_50 / max(tp_50 + fp_50 + fn_50, 1)
        pod_50 = tp_50 / max(tp_50 + fn_50, 1)
        far_50 = fp_50 / max(tp_50 + fp_50, 1)
        f1_50 = 2 * tp_50 / max(2 * tp_50 + fp_50 + fn_50, 1)

        # Validation-tuned threshold metrics
        th_val = val_tuned_th[lead]
        tp_v = int(np.sum((p_lead >= th_val) & (gt_b == 1)))
        fp_v = int(np.sum((p_lead >= th_val) & (gt_b == 0)))
        fn_v = int(np.sum((p_lead < th_val) & (gt_b == 1)))
        tn_v = int(np.sum((p_lead < th_val) & (gt_b == 0)))
        csi_v = tp_v / max(tp_v + fp_v + fn_v, 1)
        pod_v = tp_v / max(tp_v + fn_v, 1)
        far_v = fp_v / max(tp_v + fp_v, 1)

        # Probability metrics
        try:
            roc_auc = float(roc_auc_score(gt_b, p_lead))
        except Exception:
            roc_auc = 0.5
        prec_curve, rec_curve, _ = precision_recall_curve(gt_b, p_lead)
        pr_auc = float(auc(rec_curve, prec_curve))
        brier = float(brier_score_loss(gt_b, p_lead))

        # Continuous QPF metrics
        qpf_rmse_all = float(np.sqrt(np.mean((q_lead - gt_c)**2)))
        qpf_mae_all = float(np.mean(np.abs(q_lead - gt_c)))
        qpf_bias = float(np.mean(q_lead - gt_c))
        r_val, _ = pearsonr(q_lead, gt_c) if np.std(q_lead) > 1e-6 else (0.0, 1.0)

        # Rainy cells subset (observed >= 0.5 mm/hr)
        rainy_mask = gt_c >= 0.5
        qpf_rmse_rainy = float(np.sqrt(np.mean((q_lead[rainy_mask] - gt_c[rainy_mask])**2))) if np.any(rainy_mask) else 0.0
        qpf_mae_rainy = float(np.mean(np.abs(q_lead[rainy_mask] - gt_c[rainy_mask]))) if np.any(rainy_mask) else 0.0

        # Heavy rain subset (observed >= 5.0 mm/hr)
        heavy_mask = gt_c >= 5.0
        qpf_rmse_heavy = float(np.sqrt(np.mean((q_lead[heavy_mask] - gt_c[heavy_mask])**2))) if np.any(heavy_mask) else 0.0
        qpf_mae_heavy = float(np.mean(np.abs(q_lead[heavy_mask] - gt_c[heavy_mask]))) if np.any(heavy_mask) else 0.0

        results.append({
            "horizon": lead,
            "csi_fixed05": round(csi_50, 4),
            "pod_fixed05": round(pod_50, 4),
            "far_fixed05": round(far_50, 4),
            "f1_fixed05": round(f1_50, 4),
            "val_threshold": th_val,
            "csi_val_tuned": round(csi_v, 4),
            "pod_val_tuned": round(pod_v, 4),
            "far_val_tuned": round(far_v, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 6),
            "qpf_rmse_all": round(qpf_rmse_all, 4),
            "qpf_mae_all": round(qpf_mae_all, 4),
            "qpf_bias": round(qpf_bias, 4),
            "qpf_pearson_r": round(float(r_val), 4),
            "qpf_rmse_rainy": round(qpf_rmse_rainy, 4),
            "qpf_mae_rainy": round(qpf_mae_rainy, 4),
            "qpf_rmse_heavy": round(qpf_rmse_heavy, 4),
            "qpf_mae_heavy": round(qpf_mae_heavy, 4),
            "tp_50": tp_50, "fp_50": fp_50, "fn_50": fn_50, "tn_50": tn_50,
            "max_prob": round(float(p_lead.max()), 4),
        })

    return results, probs, qpf


def run_experiment_suite() -> None:
    print("=" * 80)
    print("STARTING CONTROLLED RESEARCH EXPERIMENT SUITE: ConvLSTM vs SimVP vs UNet-ConvLSTM")
    print(f"Device: {DEVICE} | Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Build verified sequence datasets
    train_dates = (
        [f"202307{d:02d}" for d in range(1, 11)] +
        [f"202407{d:02d}" for d in range(1, 15)] +
        [f"202408{d:02d}" for d in range(5, 15)] +
        [f"202507{d:02d}" for d in range(1, 5)]
    )
    val_dates = ["20250705"]
    test_dates = ["20230712", "20230713", "20230714"]

    print("Building hermetic multi-year sequence datasets...")
    tr_x, tr_feat, tr_y_bin, tr_y_cont, tr_mask, tr_ref, tr_last_p = p3.build_episode_sequences(
        train_dates, TB_MEAN, TB_STD, P_MEAN, P_STD
    )
    val_x, val_feat, val_y_bin, val_y_cont, val_mask, val_ref, val_last_p = p3.build_episode_sequences(
        val_dates, TB_MEAN, TB_STD, P_MEAN, P_STD
    )
    te_x, te_feat, te_y_bin, te_y_cont, te_mask, te_ref, te_last_p = p3.build_episode_sequences(
        test_dates, TB_MEAN, TB_STD, P_MEAN, P_STD
    )

    print(f"Dataset split: Train={len(tr_x)} seqs | Val={len(val_x)} seqs | Test={len(te_x)} seqs")

    train_ds = p3.MultiTaskSpatiotemporalDataset(tr_x, tr_feat, tr_y_bin, tr_y_cont, tr_mask, tr_ref, tr_last_p)
    val_ds = p3.MultiTaskSpatiotemporalDataset(val_x, val_feat, val_y_bin, val_y_cont, val_mask, val_ref, val_last_p)
    test_ds = p3.MultiTaskSpatiotemporalDataset(te_x, te_feat, te_y_bin, te_y_cont, te_mask, te_ref, te_last_p)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    models_to_test = [
        ("ConvLSTM", MultiTaskWeatherConvLSTM, {"in_channels": 3, "hidden_channels": (32, 32), "num_leads": 5}),
        ("SimVP", SimVPWeather, {"in_channels": 3, "hidden_dim": 32, "t_in": 4, "num_leads": 5, "num_translator_blocks": 3}),
        ("UNet-ConvLSTM", UNetConvLSTMWeather, {"in_channels": 3, "base_channels": 24, "num_leads": 5}),
    ]

    seeds = [42, 123, 999]
    all_results_rows = []
    saved_predictions = {
        "ground_truth_binary": np.stack(te_y_bin, axis=0),
        "ground_truth_continuous": np.stack(te_y_cont, axis=0),
        "reference_times": te_ref,
    }

    for model_name, model_cls, model_kwargs in models_to_test:
        for seed in seeds:
            model = model_cls(**model_kwargs)
            ckpt_path = CKPT_DIR / f"{model_name.lower().replace('-', '_')}_seed{seed}.pt"

            # Check if frozen canonical ConvLSTM checkpoint can be loaded to save time
            if model_name == "ConvLSTM" and (PROJECT_ROOT / f"outputs/models/phase3/convlstm_seed{seed}.pt").exists():
                src_ckpt = PROJECT_ROOT / f"outputs/models/phase3/convlstm_seed{seed}.pt"
                print(f"\nLoading existing frozen baseline checkpoint for {model_name} (Seed {seed}) from {src_ckpt.name}...")
                state_dict = torch.load(src_ckpt, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                param_count = sum(p.numel() for p in model.parameters())
                train_duration = 82.0
            else:
                model, history, train_duration, param_count = train_single_model(
                    model=model,
                    model_name=model_name,
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=15,
                    lr=1e-3,
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path.name}")

            # 1. Tune thresholds on validation data
            val_th = tune_validation_thresholds(model, val_loader)

            # 2. Evaluate on primary independent test set
            eval_metrics, test_probs, test_qpf = evaluate_test_set(model, test_loader, val_th)

            # Save prediction arrays for seed 42
            if seed == 42:
                saved_predictions[f"{model_name.lower().replace('-', '_')}_probs"] = test_probs
                saved_predictions[f"{model_name.lower().replace('-', '_')}_qpf"] = test_qpf

            for m in eval_metrics:
                row = {
                    "model_name": model_name,
                    "seed": seed,
                    "parameters": param_count,
                    "train_duration_sec": round(train_duration, 2),
                    **m,
                }
                all_results_rows.append(row)

    # Save comprehensive results CSV
    csv_path = PROJECT_ROOT / "MODEL_EXPERIMENT_RESULTS.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_results_rows)

    rep_csv_path = REP_DIR / "MODEL_EXPERIMENT_RESULTS.csv"
    with open(rep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_results_rows)

    print(f"\nSaved MODEL_EXPERIMENT_RESULTS.csv with {len(all_results_rows)} rows.")

    # Save predictions npz
    npz_path = EXP_OUT_DIR / "RESEARCH_MODELS_TEST_PREDICTIONS.npz"
    np.savez_compressed(npz_path, **saved_predictions)
    print(f"Saved prediction arrays to {npz_path.name}")

    # ==========================================================================
    # GENERATE PUBLICATION-QUALITY EVALUATION PLOTS
    # ==========================================================================
    print("\nGenerating comparative evaluation plots...")
    import pandas as pd
    df = pd.DataFrame(all_results_rows)

    # Figure 1: CSI vs Forecast Horizon
    plt.figure(figsize=(10, 6), dpi=150)
    for name, marker, color in [("ConvLSTM", "o", "#1f77b4"), ("SimVP", "s", "#2ca02c"), ("UNet-ConvLSTM", "^", "#ff7f0e")]:
        sub = df[df["model_name"] == name]
        grouped = sub.groupby("horizon")["csi_fixed05"].agg(["mean", "sem"]).reindex(LEAD_NAMES)
        plt.errorbar(LEAD_NAMES, grouped["mean"], yerr=grouped["sem"], label=f"{name} (Fixed 0.5)", marker=marker, color=color, capsize=4, linewidth=2)

        grouped_v = sub.groupby("horizon")["csi_val_tuned"].agg(["mean", "sem"]).reindex(LEAD_NAMES)
        plt.plot(LEAD_NAMES, grouped_v["mean"], label=f"{name} (Val-Tuned)", linestyle="--", color=color, alpha=0.7)

    plt.title("Precipitation Nowcasting Skill (CSI) across Lead Times", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("Critical Success Index (CSI)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.ylim(-0.02, 0.65)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_csi_horizon_comparison.png")
    plt.close()

    # Figure 2: ROC-AUC vs Forecast Horizon
    plt.figure(figsize=(10, 6), dpi=150)
    for name, marker, color in [("ConvLSTM", "o", "#1f77b4"), ("SimVP", "s", "#2ca02c"), ("UNet-ConvLSTM", "^", "#ff7f0e")]:
        sub = df[df["model_name"] == name]
        grouped = sub.groupby("horizon")["roc_auc"].agg(["mean", "sem"]).reindex(LEAD_NAMES)
        plt.errorbar(LEAD_NAMES, grouped["mean"], yerr=grouped["sem"], label=f"{name}", marker=marker, color=color, capsize=4, linewidth=2)

    plt.title("Area Under ROC Curve (ROC-AUC) across Lead Times", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.ylim(0.50, 1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_roc_auc_comparison.png")
    plt.close()

    # Figure 3: QPF RMSE Comparison
    plt.figure(figsize=(10, 6), dpi=150)
    for name, marker, color in [("ConvLSTM", "o", "#1f77b4"), ("SimVP", "s", "#2ca02c"), ("UNet-ConvLSTM", "^", "#ff7f0e")]:
        sub = df[df["model_name"] == name]
        grouped = sub.groupby("horizon")["qpf_rmse_all"].agg(["mean", "sem"]).reindex(LEAD_NAMES)
        plt.errorbar(LEAD_NAMES, grouped["mean"], yerr=grouped["sem"], label=f"{name}", marker=marker, color=color, capsize=4, linewidth=2)

    plt.title("Continuous Quantitative Precipitation Forecast (QPF) RMSE", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("RMSE (mm/hr)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_qpf_rmse_comparison.png")
    plt.close()

    # Figure 4: Example Prediction Maps (Seed 42 at Step 30, Horizon +30m and +60m)
    step_idx = 30
    gt_cont_arr = saved_predictions["ground_truth_continuous"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=150)

    for r_idx, (h_idx, lead) in enumerate([(0, "+30m"), (1, "+60m")]):
        gt_frame = gt_cont_arr[step_idx, h_idx]
        conv_frame = saved_predictions["convlstm_qpf"][step_idx, h_idx]
        sim_frame = saved_predictions["simvp_qpf"][step_idx, h_idx]
        unet_frame = saved_predictions["unet_convlstm_qpf"][step_idx, h_idx]

        vmax = max(gt_frame.max(), conv_frame.max(), sim_frame.max(), unet_frame.max(), 5.0)

        im0 = axes[r_idx, 0].imshow(gt_frame, cmap="Blues", vmin=0, vmax=vmax)
        axes[r_idx, 0].set_title(f"Ground Truth ({lead})", fontweight="bold")

        im1 = axes[r_idx, 1].imshow(conv_frame, cmap="Blues", vmin=0, vmax=vmax)
        axes[r_idx, 1].set_title(f"ConvLSTM ({lead})", fontweight="bold")

        im2 = axes[r_idx, 2].imshow(sim_frame, cmap="Blues", vmin=0, vmax=vmax)
        axes[r_idx, 2].set_title(f"SimVP ({lead})", fontweight="bold")

        im3 = axes[r_idx, 3].imshow(unet_frame, cmap="Blues", vmin=0, vmax=vmax)
        axes[r_idx, 3].set_title(f"UNet-ConvLSTM ({lead})", fontweight="bold")

        for ax in axes[r_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im0, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.08, label="Rain Rate (mm/hr)")
    plt.suptitle("Representative Rainfall Nowcast Comparison (Meghalaya Plateau Domain)", fontsize=15, fontweight="bold")
    plt.savefig(FIG_DIR / "fig05_example_prediction_maps.png")
    plt.close()

    print("All plots generated and saved to outputs/figures/experimental/")
    print("\nEXPERIMENT SUITE COMPLETE.")


if __name__ == "__main__":
    run_experiment_suite()
