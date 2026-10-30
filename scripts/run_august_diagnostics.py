"""Diagnose August 1-3, 2024 Prediction Behavior (Stage 3).

Examines:
1. Target event prevalence vs predicted positive prevalence across sequences.
2. Probabilities on True Positives vs True Negatives under:
   - Bugged Gate Order (cc_i, cc_f, cc_o, cc_g)
   - Corrected Canonical Gate Order (i, f, g, o)
   - Analytical Value Inversion (1 - p_bugged)
3. Sequence-by-sequence ROC-AUC, Brier score, and QPF error.

Saves:
  outputs/experiments/august_prediction_diagnostics.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_frozen_models_on_independent_test import MultiTaskWeatherConvLSTM as BuggedModel
from run_controlled_dataset_size_experiment import MultiTaskWeatherConvLSTM as CorrectModel

OUTPUTS_EXP = PROJECT_ROOT / "outputs" / "experiments"
OUTPUTS_EXP.mkdir(parents=True, exist_ok=True)
WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"


def run_august_diagnostics() -> List[Dict[str, Any]]:
    print("=" * 80)
    print("STAGE 3: AUGUST PREDICTION DIAGNOSTICS (BUGGED VS CORRECTED)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load August aligned data
    import h5py
    files = [WEATHER_DIR / f"aligned_weather_mergir_2024080{d}.nc4" for d in [1, 2, 3]]
    p_list, tb_list, mask_list, ref_times = [], [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            p_list.append(h["input_precip"][:])
            tb_list.append(h["satellite_tir1_mean"][:])
            mask_list.append(h["satellite_mask"][:])
            ref_times.extend([
                s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                for s in h["reference_time_iso"][:]
            ])

    p_all = np.concatenate(p_list, axis=0)
    tb_all = np.concatenate(tb_list, axis=0)
    mask_all = np.concatenate(mask_list, axis=0)

    with open(PROJECT_ROOT / "outputs" / "models" / "weather_convlstm" / "convlstm_normalizer.json") as f:
        stats = json.load(f)

    tb_norm = np.where((mask_all == 1) & np.isfinite(tb_all), (tb_all - stats["tb_mean"]) / stats["tb_std"], 0.0).astype(np.float32)
    log_p = np.log1p(np.maximum(p_all, 0.0))
    p_norm = np.where(np.isfinite(log_p), (log_p - stats["log_p_mean"]) / stats["log_p_std"], 0.0).astype(np.float32)
    feats = np.stack([tb_norm, p_norm, mask_all.astype(np.float32)], axis=1)

    seq_x, seq_y_b, seq_y_c, seq_ref = [], [], [], []
    for t in range(3, len(feats) - 12):
        seq_x.append(feats[t - 3 : t + 1])
        seq_y_b.append(p_all[t + 1] >= 5.0)
        seq_y_c.append(p_all[t + 1])
        seq_ref.append(ref_times[t])

    seq_x = torch.from_numpy(np.stack(seq_x, axis=0)).to(device)
    seq_y_b = np.stack(seq_y_b, axis=0)
    seq_y_c = np.stack(seq_y_c, axis=0)
    n_seq = len(seq_x)

    ckpt_path = PROJECT_ROOT / "outputs" / "models" / "weather_convlstm" / "controlled_expB_21day_seed42.pt"

    bugged_mod = BuggedModel(in_channels=3, hidden_channels=[32, 32]).to(device)
    bugged_mod.load_state_dict(torch.load(ckpt_path, map_location=device))
    bugged_mod.eval()

    correct_mod = CorrectModel(in_channels=3, hidden_channels=[32, 32]).to(device)
    correct_mod.load_state_dict(torch.load(ckpt_path, map_location=device))
    correct_mod.eval()

    with torch.no_grad():
        c_bug, q_bug = bugged_mod(seq_x)
        p_bug = torch.sigmoid(c_bug[:, 0]).cpu().numpy()
        qpf_bug = q_bug[:, 0].cpu().numpy()

        c_cor, q_cor = correct_mod(seq_x)
        p_cor = torch.sigmoid(c_cor[:, 0]).cpu().numpy()
        qpf_cor = q_cor[:, 0].cpu().numpy()

    diag_rows = []
    for i in range(n_seq):
        yb_i = seq_y_b[i].ravel()
        yc_i = seq_y_c[i].ravel()
        pb_i = p_bug[i].ravel()
        pc_i = p_cor[i].ravel()
        qb_i = qpf_bug[i].ravel()
        qc_i = qpf_cor[i].ravel()

        n_pos = int(np.sum(yb_i))
        if n_pos > 0 and n_pos < len(yb_i):
            roc_bug = float(roc_auc_score(yb_i, pb_i))
            roc_cor = float(roc_auc_score(yb_i, pc_i))
            roc_inv = float(roc_auc_score(yb_i, 1.0 - pb_i))
            pb_tp = float(np.mean(pb_i[yb_i == 1]))
            pb_tn = float(np.mean(pb_i[yb_i == 0]))
            pc_tp = float(np.mean(pc_i[yb_i == 1]))
            pc_tn = float(np.mean(pc_i[yb_i == 0]))
            qb_tp = float(np.mean(qb_i[yb_i == 1]))
            qb_tn = float(np.mean(qb_i[yb_i == 0]))
            qc_tp = float(np.mean(qc_i[yb_i == 1]))
            qc_tn = float(np.mean(qc_i[yb_i == 0]))
        else:
            roc_bug = 0.5
            roc_cor = 0.5
            roc_inv = 0.5
            pb_tp = float(np.mean(pb_i))
            pb_tn = float(np.mean(pb_i))
            pc_tp = float(np.mean(pc_i))
            pc_tn = float(np.mean(pc_i))
            qb_tp = float(np.mean(qb_i))
            qb_tn = float(np.mean(qb_i))
            qc_tp = float(np.mean(qc_i))
            qc_tn = float(np.mean(qc_i))

        bs_bug = float(brier_score_loss(yb_i, pb_i))
        bs_cor = float(brier_score_loss(yb_i, pc_i))
        rmse_q_bug = float(np.sqrt(np.mean((yc_i - qb_i) ** 2)))
        rmse_q_cor = float(np.sqrt(np.mean((yc_i - qc_i) ** 2)))

        diag_rows.append({
            "sequence_index": i,
            "reference_time_iso": seq_ref[i],
            "target_event_count": n_pos,
            "target_event_prevalence": f"{n_pos / len(yb_i):.4f}",
            "bugged_roc_auc": f"{roc_bug:.4f}",
            "bugged_prob_tp": f"{pb_tp:.4f}",
            "bugged_prob_tn": f"{pb_tn:.4f}",
            "bugged_qpf_tp": f"{qb_tp:.2f}",
            "bugged_qpf_tn": f"{qb_tn:.2f}",
            "bugged_brier_score": f"{bs_bug:.4f}",
            "bugged_qpf_rmse": f"{rmse_q_bug:.3f}",
            "corrected_roc_auc": f"{roc_cor:.4f}",
            "corrected_prob_tp": f"{pc_tp:.4f}",
            "corrected_prob_tn": f"{pc_tn:.4f}",
            "corrected_qpf_tp": f"{qc_tp:.2f}",
            "corrected_qpf_tn": f"{qc_tn:.2f}",
            "corrected_brier_score": f"{bs_cor:.4f}",
            "corrected_qpf_rmse": f"{rmse_q_cor:.3f}",
            "inversion_diagnostic_roc": f"{roc_inv:.4f}",
        })

    csv_diag = OUTPUTS_EXP / "august_prediction_diagnostics.csv"
    with open(csv_diag, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)
    print(f"Wrote August prediction diagnostics ({len(diag_rows)} sequences) to: {csv_diag}")

    # Overall Summary
    all_yb = seq_y_b.ravel()
    all_pb = p_bug.ravel()
    all_pc = p_cor.ravel()
    all_qb = qpf_bug.ravel()
    all_qc = qpf_cor.ravel()
    all_yc = seq_y_c.ravel()

    print("\nOVERALL AUGUST 1-3, 2024 EVALUATION SUMMARY (+30m Horizon):")
    print(f"  Total Pixels:             {len(all_yb)} (129 sequences x 1,000 pixels)")
    print(f"  Positive Event Pixels:    {int(np.sum(all_yb))} ({np.mean(all_yb)*100:.2f}%)")
    print(f"  Bugged Model ROC-AUC:     {roc_auc_score(all_yb, all_pb):.4f}")
    print(f"  Bugged Mean Prob (TP/TN): {all_pb[all_yb == 1].mean():.4f} / {all_pb[all_yb == 0].mean():.4f}")
    print(f"  Bugged Mean QPF (TP/TN):  {all_qb[all_yb == 1].mean():.2f} mm/hr / {all_qb[all_yb == 0].mean():.2f} mm/hr")
    print(f"  Bugged QPF Overall RMSE:  {np.sqrt(np.mean((all_yc - all_qb)**2)):.3f} mm/hr")
    print(f"  -------------------------------------------------------------")
    print(f"  CORRECTED MODEL ROC-AUC:  {roc_auc_score(all_yb, all_pc):.4f}")
    print(f"  Corrected Mean Prob (TP): {all_pc[all_yb == 1].mean():.4f} / {all_pc[all_yb == 0].mean():.4f}")
    print(f"  Corrected Mean QPF (TP):  {all_qc[all_yb == 1].mean():.2f} mm/hr / {all_qc[all_yb == 0].mean():.2f} mm/hr")
    print(f"  Corrected QPF Overall RMSE:{np.sqrt(np.mean((all_yc - all_qc)**2)):.3f} mm/hr")

    return diag_rows


if __name__ == "__main__":
    run_august_diagnostics()
