import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

from src.models.convlstm import MultiTaskWeatherConvLSTM
from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.unet_convlstm import UNetConvLSTMWeather
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences, tune_val_thresholds, DEVICE

OUT_DIR = Path("outputs/reports/model_research/step1_forensic_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Primary Test Data (July 12-14, 2023, 129 sequences)
test_primary_dates = ["20230712", "20230713", "20230714"]
val_dates = ["20250705"]

te_x, te_y_b, te_y_c, te_m, te_t = build_advanced_sequences(test_primary_dates, t_in=4)
va_x, va_y_b, va_y_c, va_m, va_t = build_advanced_sequences(val_dates, t_in=4)

gt_bin = np.stack(te_y_b, axis=0)      # (129, 5, 25, 40)
gt_cont = np.stack(te_y_c, axis=0)     # (129, 5, 25, 40)

# Validation targets for tuning thresholds
val_gt_bin = np.stack(va_y_b, axis=0)  # (33, 5, 25, 40)

# Also load PHASE3_TEST_PREDICTIONS_2023.npz for existing saved models
npz = np.load("outputs/experiments/PHASE3_TEST_PREDICTIONS/PHASE3_TEST_PREDICTIONS_2023.npz")

leads = ["+30m", "+60m", "+120m", "+180m", "+360m"]

models_to_eval = {}

# Model 1: Persistence
models_to_eval["Persistence"] = {
    "probs": npz["persistence_binary"].astype(np.float32),
    "qpf": npz["persistence_continuous"],
    "val_th": {l: 0.5 for l in leads}
}

# Model 2: Dense Optical Flow
models_to_eval["Optical_Flow_Dense"] = {
    "probs": npz["dense_flow_binary"].astype(np.float32),
    "qpf": npz["dense_flow_continuous"],
    "val_th": {l: 0.5 for l in leads}
}

# Model 3: Canonical ConvLSTM (From NPZ)
conv_p = npz["convlstm_probs"]
conv_q = npz["convlstm_qpf"]
# Determine val threshold for ConvLSTM
# Load ConvLSTM model to evaluate on val set for exact threshold
conv_model = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=5)
conv_model.load_state_dict(torch.load("outputs/models/phase3/convlstm_seed42.pt", map_location=DEVICE))
conv_model.to(DEVICE)
conv_model.eval()

with torch.no_grad():
    v_c, _ = conv_model(torch.from_numpy(np.stack(va_x, axis=0)).to(DEVICE))
    v_p_conv = torch.sigmoid(v_c).cpu().numpy()

conv_val_th = {}
for h_idx, lead in enumerate(leads):
    p_v = v_p_conv[:, h_idx].ravel()
    g_v = val_gt_bin[:, h_idx].ravel().astype(int)
    best_c, best_t = -1.0, 0.50
    for t in np.arange(0.02, 0.60, 0.02):
        tp = np.sum((p_v >= t) & (g_v == 1))
        fp = np.sum((p_v >= t) & (g_v == 0))
        fn = np.sum((p_v < t) & (g_v == 1))
        csi = tp / max(tp + fp + fn, 1)
        if csi > best_c:
            best_c = csi
            best_t = float(t)
    conv_val_th[lead] = round(best_t, 4)

models_to_eval["ConvLSTM_Baseline"] = {
    "probs": conv_p,
    "qpf": conv_q,
    "val_th": conv_val_th
}

# Model 4: SimVP Optimal
simvp_model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5)
simvp_model.load_state_dict(torch.load("outputs/experiments/model_research/checkpoints/simvp_optimal_best.pt", map_location=DEVICE))
simvp_model.to(DEVICE)
simvp_model.eval()

with torch.no_grad():
    v_c, _ = simvp_model(torch.from_numpy(np.stack(va_x, axis=0)).to(DEVICE))
    v_p_simvp = torch.sigmoid(v_c).cpu().numpy()
    
    t_c, t_q = simvp_model(torch.from_numpy(np.stack(te_x, axis=0)).to(DEVICE))
    simvp_p = torch.sigmoid(t_c).cpu().numpy()
    simvp_q = t_q.cpu().numpy()

simvp_val_th = {}
for h_idx, lead in enumerate(leads):
    p_v = v_p_simvp[:, h_idx].ravel()
    g_v = val_gt_bin[:, h_idx].ravel().astype(int)
    best_c, best_t = -1.0, 0.50
    for t in np.arange(0.02, 0.60, 0.02):
        tp = np.sum((p_v >= t) & (g_v == 1))
        fp = np.sum((p_v >= t) & (g_v == 0))
        fn = np.sum((p_v < t) & (g_v == 1))
        csi = tp / max(tp + fp + fn, 1)
        if csi > best_c:
            best_c = csi
            best_t = float(t)
    simvp_val_th[lead] = round(best_t, 4)

models_to_eval["SimVP_Optimal"] = {
    "probs": simvp_p,
    "qpf": simvp_q,
    "val_th": simvp_val_th
}

# Model 5: UNet-ConvLSTM
unet_model = UNetConvLSTMWeather(in_channels=3, base_channels=24, num_leads=5)
unet_model.load_state_dict(torch.load("outputs/experiments/model_research/checkpoints/unet_convlstm_best.pt", map_location=DEVICE))
unet_model.to(DEVICE)
unet_model.eval()

with torch.no_grad():
    v_c, _ = unet_model(torch.from_numpy(np.stack(va_x, axis=0)).to(DEVICE))
    v_p_unet = torch.sigmoid(v_c).cpu().numpy()
    
    t_c, t_q = unet_model(torch.from_numpy(np.stack(te_x, axis=0)).to(DEVICE))
    unet_p = torch.sigmoid(t_c).cpu().numpy()
    unet_q = t_q.cpu().numpy()

unet_val_th = {}
for h_idx, lead in enumerate(leads):
    p_v = v_p_unet[:, h_idx].ravel()
    g_v = val_gt_bin[:, h_idx].ravel().astype(int)
    best_c, best_t = -1.0, 0.50
    for t in np.arange(0.02, 0.60, 0.02):
        tp = np.sum((p_v >= t) & (g_v == 1))
        fp = np.sum((p_v >= t) & (g_v == 0))
        fn = np.sum((p_v < t) & (g_v == 1))
        csi = tp / max(tp + fp + fn, 1)
        if csi > best_c:
            best_c = csi
            best_t = float(t)
    unet_val_th[lead] = round(best_t, 4)

models_to_eval["UNet_ConvLSTM"] = {
    "probs": unet_p,
    "qpf": unet_q,
    "val_th": unet_val_th
}

verified_rows = []

for m_name, m_data in models_to_eval.items():
    probs = m_data["probs"]
    qpf = m_data["qpf"]
    val_th = m_data["val_th"]
    
    for h_idx, lead in enumerate(leads):
        p = probs[:, h_idx].ravel()
        q = qpf[:, h_idx].ravel()
        gb = gt_bin[:, h_idx].ravel().astype(int)
        gc = gt_cont[:, h_idx].ravel()
        
        # 1. Fixed threshold 0.50
        tp_50 = int(np.sum((p >= 0.5) & (gb == 1)))
        fp_50 = int(np.sum((p >= 0.5) & (gb == 0)))
        fn_50 = int(np.sum((p < 0.5) & (gb == 1)))
        tn_50 = int(np.sum((p < 0.5) & (gb == 0)))
        csi_50 = tp_50 / max(tp_50 + fp_50 + fn_50, 1)
        pod_50 = tp_50 / max(tp_50 + fn_50, 1)
        far_50 = fp_50 / max(tp_50 + fp_50, 1)
        prec_50 = tp_50 / max(tp_50 + fp_50, 1)
        rec_50 = tp_50 / max(tp_50 + fn_50, 1)
        
        # 2. Validation-tuned threshold
        tau_val = val_th[lead]
        tp_v = int(np.sum((p >= tau_val) & (gb == 1)))
        fp_v = int(np.sum((p >= tau_val) & (gb == 0)))
        fn_v = int(np.sum((p < tau_val) & (gb == 1)))
        tn_v = int(np.sum((p < tau_val) & (gb == 0)))
        csi_v = tp_v / max(tp_v + fp_v + fn_v, 1)
        pod_v = tp_v / max(tp_v + fn_v, 1)
        far_v = fp_v / max(tp_v + fp_v, 1)
        prec_v = tp_v / max(tp_v + fp_v, 1)
        rec_v = tp_v / max(tp_v + fn_v, 1)
        
        # 3. Probabilistic metrics
        try:
            roc_auc = float(roc_auc_score(gb, p))
        except Exception:
            roc_auc = 0.5
        prec_curve, rec_curve, _ = precision_recall_curve(gb, p)
        pr_auc = float(auc(rec_curve, prec_curve))
        brier = float(brier_score_loss(gb, p))
        
        # 4. Continuous QPF metrics
        rmse_all = float(np.sqrt(np.mean((q - gc)**2)))
        mae_all = float(np.mean(np.abs(q - gc)))
        r_val, _ = pearsonr(q, gc) if np.std(q) > 1e-6 else (0.0, 1.0)
        
        heavy_mask = gc >= 5.0
        rmse_heavy = float(np.sqrt(np.mean((q[heavy_mask] - gc[heavy_mask])**2))) if np.any(heavy_mask) else 0.0
        
        verified_rows.append({
            "model_name": m_name,
            "horizon": lead,
            "csi_fixed05": round(csi_50, 4),
            "pod_fixed05": round(pod_50, 4),
            "far_fixed05": round(far_50, 4),
            "precision_fixed05": round(prec_50, 4),
            "recall_fixed05": round(rec_50, 4),
            "tp_fixed05": tp_50,
            "fp_fixed05": fp_50,
            "fn_fixed05": fn_50,
            "tn_fixed05": tn_50,
            "tau_val_tuned": tau_val,
            "csi_val_tuned": round(csi_v, 4),
            "pod_val_tuned": round(pod_v, 4),
            "far_val_tuned": round(far_v, 4),
            "precision_val_tuned": round(prec_v, 4),
            "recall_val_tuned": round(rec_v, 4),
            "tp_val_tuned": tp_v,
            "fp_val_tuned": fp_v,
            "fn_val_tuned": fn_v,
            "tn_val_tuned": tn_v,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 6),
            "qpf_rmse_all": round(rmse_all, 4),
            "qpf_mae_all": round(mae_all, 4),
            "pearson_r": round(float(r_val), 4),
            "rmse_heavy_rain": round(rmse_heavy, 4),
            "verification_status": "VERIFIED_FROM_RAW_PREDICTION_ARRAYS"
        })

csv_file = OUT_DIR / "VERIFIED_BASELINE_METRICS.csv"
fieldnames = list(verified_rows[0].keys())
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(verified_rows)

print(f"Successfully computed and saved {len(verified_rows)} verified metric rows to {csv_file}")
