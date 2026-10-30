import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import pearsonr
import torch
from src.models.experimental.simvp import SimVPWeather
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences, DEVICE

test_primary_dates = ["20230712", "20230713", "20230714"]
te_x, te_y_b, te_y_c, te_m, te_t = build_advanced_sequences(test_primary_dates, t_in=4)

model = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5)
ckpt_path = Path("outputs/experiments/model_research/checkpoints/simvp_optimal_best.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

x_tensor = torch.from_numpy(np.stack(te_x, axis=0)).to(DEVICE)
with torch.no_grad():
    c_logits, qpf = model(x_tensor)
    simvp_probs = torch.sigmoid(c_logits).cpu().numpy()
    simvp_qpf = qpf.cpu().numpy()

gt_bin = np.stack(te_y_b, axis=0)
gt_cont = np.stack(te_y_c, axis=0)

leads = ["+30m", "+60m", "+120m", "+180m", "+360m"]

print("=== FORECAST DEGRADATION DIAGNOSIS ACROSS 5 HORIZONS (SIMVP) ===")

for h_idx, lead in enumerate(leads):
    p = simvp_probs[:, h_idx]
    q = simvp_qpf[:, h_idx]
    gb = gt_bin[:, h_idx]
    gc = gt_cont[:, h_idx]
    
    p_flat = p.ravel()
    p_max = float(np.max(p_flat))
    p_mean = float(np.mean(p_flat))
    p_p99 = float(np.percentile(p_flat, 99))
    p_p95 = float(np.percentile(p_flat, 95))
    p_gt_05_count = int(np.sum(p_flat >= 0.5))
    
    grad_q_y, grad_q_x = np.gradient(q, axis=(1, 2))
    grad_mag_q = np.sqrt(grad_q_y**2 + grad_q_x**2)
    grad_gc_y, grad_gc_x = np.gradient(gc, axis=(1, 2))
    grad_mag_gc = np.sqrt(grad_gc_y**2 + grad_gc_x**2)
    smoothness_ratio = float(np.mean(grad_mag_q) / max(np.mean(grad_mag_gc), 1e-6))
    
    heavy_mask = gc >= 5.0
    mean_heavy_actual = float(np.mean(gc[heavy_mask])) if np.sum(heavy_mask) > 0 else 0.0
    mean_heavy_pred = float(np.mean(q[heavy_mask])) if np.sum(heavy_mask) > 0 else 0.0
    underestimate_ratio = float(mean_heavy_pred / max(mean_heavy_actual, 1e-6))
    
    tp_50 = int(np.sum((p_flat >= 0.5) & (gb.ravel() == 1)))
    fp_50 = int(np.sum((p_flat >= 0.5) & (gb.ravel() == 0)))
    fn_50 = int(np.sum((p_flat < 0.5) & (gb.ravel() == 1)))
    tn_50 = int(np.sum((p_flat < 0.5) & (gb.ravel() == 0)))
    
    clim_map = np.mean(gc, axis=0)
    mean_pred_map = np.mean(q, axis=0)
    corr_to_clim, _ = pearsonr(mean_pred_map.ravel(), clim_map.ravel())
    
    time_variance = float(np.mean(np.var(q, axis=0)))
    spatial_variance = float(np.mean(np.var(q, axis=(1, 2))))
    
    print(f"\n--- HORIZON {lead} ---")
    print(f"  1. Probabilities: Max={p_max:.4f}, Mean={p_mean:.4f}, P99={p_p99:.4f}, P95={p_p95:.4f}, N(P>=0.5)={p_gt_05_count}")
    print(f"  2. Smoothness Ratio (Pred Grad / GT Grad): {smoothness_ratio:.4f}")
    print(f"  3. Heavy Rain Bias (>=5 mm/hr): Actual Mean={mean_heavy_actual:.2f}, Pred Mean={mean_heavy_pred:.2f} (Ratio={underestimate_ratio*100:.1f}%)")
    print(f"  4. Errors at 0.50 Threshold: TP={tp_50}, FP={fp_50}, FN={fn_50}, TN={tn_50}")
    print(f"  5. Climatology Correlation: r={corr_to_clim:.4f} | Temporal Var={time_variance:.4f}, Spatial Var={spatial_variance:.4f}")
