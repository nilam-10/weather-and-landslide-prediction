import numpy as np
import scipy.ndimage as ndi
from scipy.stats import pearsonr

npz = np.load("outputs/experiments/PHASE3_TEST_PREDICTIONS/PHASE3_TEST_PREDICTIONS_2023.npz")
gt_bin = npz["ground_truth_binary"]       # (129, 5, 25, 40)
gt_cont = npz["ground_truth_continuous"]  # (129, 5, 25, 40)
c_probs = npz["convlstm_probs"]           # (129, 5, 25, 40)
c_qpf = npz["convlstm_qpf"]               # (129, 5, 25, 40)
ref_times = npz["reference_times"]

leads = ["+30m", "+60m", "+120m", "+180m", "+360m"]

print("=== FORECAST DEGRADATION DIAGNOSIS ACROSS 5 HORIZONS (CONVLSTM) ===")

for h_idx, lead in enumerate(leads):
    p = c_probs[:, h_idx]      # (129, 25, 40)
    q = c_qpf[:, h_idx]        # (129, 25, 40)
    gb = gt_bin[:, h_idx]      # (129, 25, 40)
    gc = gt_cont[:, h_idx]     # (129, 25, 40)
    
    # 1. Probability distribution & shrinkage
    p_flat = p.ravel()
    p_max = float(np.max(p_flat))
    p_mean = float(np.mean(p_flat))
    p_p99 = float(np.percentile(p_flat, 99))
    p_p95 = float(np.percentile(p_flat, 95))
    p_p90 = float(np.percentile(p_flat, 90))
    p_gt_05_count = int(np.sum(p_flat >= 0.5))
    
    # 2. Smoothness / Spatial gradient magnitude
    # Compute spatial gradient of predictions vs ground truth
    grad_q_y, grad_q_x = np.gradient(q, axis=(1, 2))
    grad_mag_q = np.sqrt(grad_q_y**2 + grad_q_x**2)
    grad_gc_y, grad_gc_x = np.gradient(gc, axis=(1, 2))
    grad_mag_gc = np.sqrt(grad_gc_y**2 + grad_gc_x**2)
    smoothness_ratio = float(np.mean(grad_mag_q) / max(np.mean(grad_mag_gc), 1e-6))
    
    # 3. Systematic underestimation of heavy rain (gc >= 5 mm/hr)
    heavy_mask = gc >= 5.0
    n_heavy = int(np.sum(heavy_mask))
    mean_heavy_actual = float(np.mean(gc[heavy_mask])) if n_heavy > 0 else 0.0
    mean_heavy_pred = float(np.mean(q[heavy_mask])) if n_heavy > 0 else 0.0
    underestimate_ratio = float(mean_heavy_pred / max(mean_heavy_actual, 1e-6))
    
    # Extreme rain (gc >= 10 mm/hr)
    extreme_mask = gc >= 10.0
    n_extreme = int(np.sum(extreme_mask))
    mean_extreme_actual = float(np.mean(gc[extreme_mask])) if n_extreme > 0 else 0.0
    mean_extreme_pred = float(np.mean(q[extreme_mask])) if n_extreme > 0 else 0.0
    
    # 4. Contingency at 0.50: False alarms vs Misses
    tp_50 = int(np.sum((p_flat >= 0.5) & (gb.ravel() == 1)))
    fp_50 = int(np.sum((p_flat >= 0.5) & (gb.ravel() == 0)))
    fn_50 = int(np.sum((p_flat < 0.5) & (gb.ravel() == 1)))
    tn_50 = int(np.sum((p_flat < 0.5) & (gb.ravel() == 0)))
    
    # 5. Classification vs Regression Agreement
    # Does regression predict >= 5.0 when prob >= 0.5?
    q_ge5 = q.ravel() >= 5.0
    p_ge05 = p_flat >= 0.5
    agreement_count = np.sum(q_ge5 == p_ge05)
    agreement_pct = float(agreement_count / len(p_flat) * 100)
    q_ge5_count = int(np.sum(q_ge5))
    
    # 6. Spatial correlation / Climatology collapse
    # Climatological mean map of ground truth
    clim_map = np.mean(gc, axis=0) # (25, 40)
    mean_pred_map = np.mean(q, axis=0) # (25, 40)
    corr_to_clim, _ = pearsonr(mean_pred_map.ravel(), clim_map.ravel())
    
    # Variance of prediction over time vs variance over space
    time_variance = float(np.mean(np.var(q, axis=0))) # temporal variance
    spatial_variance = float(np.mean(np.var(q, axis=(1, 2)))) # spatial variance
    
    print(f"\n--- HORIZON {lead} ---")
    print(f"  1. Probabilities: Max={p_max:.4f}, Mean={p_mean:.4f}, P99={p_p99:.4f}, P95={p_p95:.4f}, N(P>=0.5)={p_gt_05_count}")
    print(f"  2. Smoothness Ratio (Pred Grad / GT Grad): {smoothness_ratio:.4f} ({'Excessively Blurred' if smoothness_ratio < 0.3 else 'Moderate Blur'})")
    print(f"  3. Heavy Rain Bias (>=5 mm/hr): Actual Mean={mean_heavy_actual:.2f}, Pred Mean={mean_heavy_pred:.2f} (Ratio={underestimate_ratio*100:.1f}%)")
    print(f"     Extreme Rain Bias (>=10 mm/hr): Actual Mean={mean_extreme_actual:.2f}, Pred Mean={mean_extreme_pred:.2f} (Ratio={mean_extreme_pred/max(mean_extreme_actual,1e-6)*100:.1f}%)")
    print(f"  4. Errors at 0.50 Threshold: TP={tp_50}, FP={fp_50}, FN={fn_50}, TN={tn_50} -> {'FN dominates (Zero Alerts)' if fn_50 > 100*max(fp_50,1) else 'Balanced'}")
    print(f"  5. Head Agreement: P(>=0.5)={p_gt_05_count} vs QPF(>=5mm/hr)={q_ge5_count} | Overall Agreement={agreement_pct:.2f}%")
    print(f"  6. Climatology Correlation: r={corr_to_clim:.4f} | Temporal Var={time_variance:.4f}, Spatial Var={spatial_variance:.4f}")
