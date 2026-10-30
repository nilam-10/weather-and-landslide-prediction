"""Step 5: Stationary Block Bootstrap Statistical Significance Analysis (Optimized Vectorized Engine).

Precomputes sequence-level contingency, squared errors, and spatial fraction fields once,
then evaluates 1,000 stationary block bootstrap resamples (block size = 6 seqs = 3 hours)
to derive 95% Confidence Intervals and paired permutation p-values for:
  - Exp D (Combined Objective) vs Exp A (Baseline Control)
Across Seeds 42, 123, 999 on:
  1. July 5, 2025 Validation Cohort (33 sequences)
  2. July 12-14, 2023 Confirmatory Test Cohort (129 sequences)
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
import scipy.ndimage as ndi
from scipy.stats import pearsonr
import torch

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.experimental.simvp import SimVPWeather
from src.evaluation.spatial_metrics import SpatialVerificationEngine as SVE
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = PROJECT_ROOT / "outputs/experiments/model_research/step5_three_seed_replication"
CKPT_DIR = EXP_DIR / "CHECKPOINTS"

LEADS = ["+30m", "+60m", "+120m", "+180m", "+360m"]
SEEDS = [42, 123, 999]
BLOCK_SIZE = 6  # 3 hours (6 * 30 min)
N_BOOTSTRAP = 1000
GRID_KM = 10.55


def load_cohorts():
    val_dates = ["20250705"]
    test_dates = ["20230712", "20230713", "20230714"]

    val_x, val_yb, val_yc, val_m, val_t = build_advanced_sequences(val_dates, t_in=4)
    te_x, te_yb, te_yc, te_m, te_t = build_advanced_sequences(test_dates, t_in=4)

    return (
        {"x": torch.from_numpy(np.stack(val_x, axis=0)), "yc": np.stack(val_yc, axis=0), "yb": np.stack(val_yb, axis=0), "m": np.stack(val_m, axis=0), "name": "Validation_20250705"},
        {"x": torch.from_numpy(np.stack(te_x, axis=0)), "yc": np.stack(te_yc, axis=0), "yb": np.stack(te_yb, axis=0), "m": np.stack(te_m, axis=0), "name": "ConfirmatoryTest_20230712_14"},
    )


def get_model_predictions(model_path: Path, x_tensor: torch.Tensor) -> np.ndarray:
    model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5, num_translator_blocks=3).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_qpf = []
    batch_size = 16
    n = len(x_tensor)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bx = x_tensor[i : i + batch_size].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                _, p_qpf = model(bx)
            all_qpf.append(p_qpf.cpu().float().numpy())
    return np.concatenate(all_qpf, axis=0)  # (N, Leads, H, W)


def generate_block_bootstrap_indices(n_samples: int, block_size: int, n_boot: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_blocks_needed = int(np.ceil(n_samples / block_size))
    valid_starts = np.arange(max(1, n_samples - block_size + 1))

    boot_indices = np.zeros((n_boot, n_samples), dtype=int)
    for b in range(n_boot):
        starts = rng.choice(valid_starts, size=n_blocks_needed, replace=True)
        blocks = [np.arange(st, min(st + block_size, n_samples)) for st in starts]
        idx = np.concatenate(blocks)[:n_samples]
        boot_indices[b] = idx
    return boot_indices


def precompute_sequence_stats(obs_sequences: np.ndarray, fcst_sequences: np.ndarray, mask_sequences: np.ndarray, threshold: float = 5.0, radius_cells: int = 3):
    """Precompute per-sequence scalar components for instant bootstrap aggregation."""
    N = len(obs_sequences)
    tp_per_seq = np.zeros(N)
    fp_per_seq = np.zeros(N)
    fn_per_seq = np.zeros(N)
    se_per_seq = np.zeros(N)
    n_valid_per_seq = np.zeros(N)
    fbs_num_per_seq = np.zeros(N)
    fbs_den_per_seq = np.zeros(N)

    kernel = SVE.create_spatial_kernel(radius_cells, kernel_type="circular")

    for t in range(N):
        o_field = obs_sequences[t]
        f_field = fcst_sequences[t]
        m_field = mask_sequences[t] > 0

        # Valid cells
        o_val = o_field[m_field]
        f_val = f_field[m_field]
        n_valid = len(o_val)
        n_valid_per_seq[t] = n_valid

        # QPF squared error
        se_per_seq[t] = np.sum((f_val - o_val) ** 2)

        # Binary contingency
        o_bin = o_val >= threshold
        f_bin = f_val >= threshold
        tp_per_seq[t] = np.sum(o_bin & f_bin)
        fp_per_seq[t] = np.sum(~o_bin & f_bin)
        fn_per_seq[t] = np.sum(o_bin & ~f_bin)

        # Spatial fractions
        p_obs, m_o = SVE.compute_fraction_field(o_field >= threshold, mask=m_field, radius_cells=radius_cells, kernel_type="circular")
        p_fcst, m_f = SVE.compute_fraction_field(f_field >= threshold, mask=m_field, radius_cells=radius_cells, kernel_type="circular")
        val_frac = m_o & m_f

        diff_sq = (p_fcst[val_frac] - p_obs[val_frac]) ** 2
        sum_sq = p_fcst[val_frac] ** 2 + p_obs[val_frac] ** 2
        fbs_num_per_seq[t] = np.sum(diff_sq)
        fbs_den_per_seq[t] = np.sum(sum_sq)

    return {
        "tp": tp_per_seq,
        "fp": fp_per_seq,
        "fn": fn_per_seq,
        "se": se_per_seq,
        "n_valid": n_valid_per_seq,
        "fbs_num": fbs_num_per_seq,
        "fbs_den": fbs_den_per_seq,
    }


def aggregate_bootstrap_sample(stats: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    tp = np.sum(stats["tp"][idx])
    fp = np.sum(stats["fp"][idx])
    fn = np.sum(stats["fn"][idx])
    csi = float(tp / max(tp + fp + fn, 1))

    se_tot = np.sum(stats["se"][idx])
    n_tot = np.sum(stats["n_valid"][idx])
    rmse = float(np.sqrt(se_tot / max(n_tot, 1)))

    fbs_num = np.sum(stats["fbs_num"][idx])
    fbs_den = np.sum(stats["fbs_den"][idx])
    if fbs_den < 1e-12:
        fss = 1.0 if (tp + fp + fn) == 0 else 0.0
    else:
        fss = float(np.clip(1.0 - (fbs_num / fbs_den), 0.0, 1.0))

    return {"csi": csi, "fss": fss, "rmse": rmse}


def main():
    print("=" * 80)
    print("STARTING VECTORIZED BLOCK BOOTSTRAP SIGNIFICANCE ENGINE (1,000 RESAMPLES)")
    print("=" * 80)

    val_cohort, test_cohort = load_cohorts()
    cohorts = [val_cohort, test_cohort]
    k_cells_30km = SVE.km_to_grid_radius(30.0, GRID_KM)

    results = []

    for cohort in cohorts:
        n_samples = len(cohort["yc"])
        c_name = cohort["name"]
        print(f"\nProcessing Cohort: {c_name} (N = {n_samples} sequences, Block Size = {BLOCK_SIZE})...")

        boot_idx = generate_block_bootstrap_indices(n_samples, BLOCK_SIZE, N_BOOTSTRAP, seed=42)

        # Precompute predictions for all 6 models
        preds_expA = {}
        preds_expD = {}
        for seed in SEEDS:
            ckpt_a = CKPT_DIR / f"ExpA_Seed{seed}_best.pt"
            ckpt_d = CKPT_DIR / f"ExpD_Seed{seed}_best.pt"
            preds_expA[seed] = get_model_predictions(ckpt_a, cohort["x"])
            preds_expD[seed] = get_model_predictions(ckpt_d, cohort["x"])

        for h_idx, lead in enumerate(LEADS):
            print(f"  Analyzing Horizon {lead}...")
            obs_h = cohort["yc"][:, h_idx]
            mask_h = cohort["m"][:, h_idx]

            for seed in SEEDS:
                f_a = preds_expA[seed][:, h_idx]
                f_d = preds_expD[seed][:, h_idx]

                stats_a = precompute_sequence_stats(obs_h, f_a, mask_h, threshold=5.0, radius_cells=k_cells_30km)
                stats_d = precompute_sequence_stats(obs_h, f_d, mask_h, threshold=5.0, radius_cells=k_cells_30km)

                # Point estimates on full cohort
                full_idx = np.arange(n_samples)
                pt_a = aggregate_bootstrap_sample(stats_a, full_idx)
                pt_d = aggregate_bootstrap_sample(stats_d, full_idx)

                # Bootstrap resampling
                delta_csi = np.zeros(N_BOOTSTRAP)
                delta_fss = np.zeros(N_BOOTSTRAP)
                delta_rmse = np.zeros(N_BOOTSTRAP)

                for b in range(N_BOOTSTRAP):
                    idx = boot_idx[b]
                    ma = aggregate_bootstrap_sample(stats_a, idx)
                    md = aggregate_bootstrap_sample(stats_d, idx)

                    delta_csi[b] = md["csi"] - ma["csi"]
                    delta_fss[b] = md["fss"] - ma["fss"]
                    delta_rmse[b] = md["rmse"] - ma["rmse"]

                for metric_name, pa, pd, deltas, higher_better in [
                    ("pointwise_csi_5mm", pt_a["csi"], pt_d["csi"], delta_csi, True),
                    ("circular_fss_30km_5mm", pt_a["fss"], pt_d["fss"], delta_fss, True),
                    ("qpf_rmse", pt_a["rmse"], pt_d["rmse"], delta_rmse, False),
                ]:
                    mean_delta = float(np.mean(deltas))
                    ci_lower = float(np.percentile(deltas, 2.5))
                    ci_upper = float(np.percentile(deltas, 97.5))

                    if higher_better:
                        p_val = float(np.mean(deltas <= 0.0))
                    else:
                        p_val = float(np.mean(deltas >= 0.0))

                    sig = "YES (p < 0.05)" if p_val < 0.05 else "NO (p >= 0.05)"

                    results.append({
                        "cohort": c_name,
                        "seed": seed,
                        "horizon": lead,
                        "metric": metric_name,
                        "point_estimate_ExpA": round(pa, 4),
                        "point_estimate_ExpD": round(pd, 4),
                        "point_estimate_delta": round(pd - pa, 4),
                        "bootstrap_mean_delta": round(mean_delta, 4),
                        "ci95_lower": round(ci_lower, 4),
                        "ci95_upper": round(ci_upper, 4),
                        "p_value": round(p_val, 4),
                        "significant_at_05": sig,
                    })

    # Save to CSV
    out_csv = EXP_DIR / "BOOTSTRAP_STATISTICAL_ANALYSIS.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SUCCESS] Saved {len(results)} rows to: {out_csv.name}")


if __name__ == "__main__":
    main()
