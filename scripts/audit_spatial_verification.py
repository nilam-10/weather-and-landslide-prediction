"""Audit Spatial Verification: Geometry, Mathematical Formulations, and Seed Comparisons."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.spatial_metrics import SpatialVerificationEngine as SVE
COHORT_PATH = PROJECT_ROOT / "outputs/reports/model_research/step2_spatial_verification/test_predictions_cohort.npz"

def main():
    npz = np.load(COHORT_PATH)
    gt_cont = npz["ground_truth_continuous"]
    mask_stack = npz["observation_mask"]
    
    models = {
        "ConvLSTM_Seed42": npz["convlstm_seed42_qpf"],
        "ConvLSTM_Seed123": npz["convlstm_seed123_qpf"],
        "ConvLSTM_Seed999": npz["convlstm_seed999_qpf"],
        "SimVP_Optimal": npz["simvp_qpf"],
        "UNet_ConvLSTM": npz["unet_convlstm_qpf"],
        "Persistence": npz["persistence_qpf"],
        "Dense_Optical_Flow": npz["dense_flow_qpf"],
    }
    
    leads = ["+30m", "+60m", "+120m", "+180m", "+360m"]
    radii_km = [10.0, 20.0, 30.0, 50.0]
    thresholds = [0.1, 1.0, 5.0]
    
    print("=== KERNEL GEOMETRY ANALYSIS ===")
    for r_km in radii_km:
        k = SVE.km_to_grid_radius(r_km)
        k_sq = SVE.create_spatial_kernel(k, "square")
        k_ci = SVE.create_spatial_kernel(k, "circular")
        
        # Grid specs: dx = 10.03 km, dy = 11.08 km
        dx, dy = 10.03, 11.08
        sq_width = (2 * k + 1) * dx
        sq_height = (2 * k + 1) * dy
        sq_corner_dist = np.sqrt((k * dx)**2 + (k * dy)**2)
        sq_area = np.sum(k_sq) * (dx * dy)
        
        ci_area = np.sum(k_ci) * (dx * dy)
        true_circle_area = np.pi * (r_km ** 2)
        
        print(f"Radius {r_km:4.1f} km (k={k}):")
        print(f"  Square:   cells={int(np.sum(k_sq))}, size={2*k+1}x{2*k+1}, width={sq_width:.1f} km, max_reach={sq_corner_dist:.1f} km, area={sq_area:.0f} km²")
        print(f"  Circular: cells={int(np.sum(k_ci))}, area={ci_area:.0f} km², true_circle_area={true_circle_area:.0f} km²")

    print("\n=== COMPARING SQUARE VS CIRCULAR KERNELS (ConvLSTM vs SimVP at 5.0 mm/hr) ===")
    for lead_idx, lead in enumerate(leads):
        obs_h = gt_cont[:, lead_idx]
        mask_h = mask_stack[:, lead_idx]
        print(f"\n--- Lead {lead} ---")
        for r_km in radii_km:
            k = SVE.km_to_grid_radius(r_km)
            res_c_sq = SVE.evaluate_sequence_dataset(obs_h, models["ConvLSTM_Seed42"][:, lead_idx], threshold=5.0, radius_cells=k, mask=mask_h, kernel_type="square")
            res_c_ci = SVE.evaluate_sequence_dataset(obs_h, models["ConvLSTM_Seed42"][:, lead_idx], threshold=5.0, radius_cells=k, mask=mask_h, kernel_type="circular")
            res_s_sq = SVE.evaluate_sequence_dataset(obs_h, models["SimVP_Optimal"][:, lead_idx], threshold=5.0, radius_cells=k, mask=mask_h, kernel_type="square")
            res_s_ci = SVE.evaluate_sequence_dataset(obs_h, models["SimVP_Optimal"][:, lead_idx], threshold=5.0, radius_cells=k, mask=mask_h, kernel_type="circular")
            
            print(f"r={r_km:4.1f} km | ConvLSTM Sq={res_c_sq['accumulated_fss']:.4f}, Cir={res_c_ci['accumulated_fss']:.4f} | SimVP Sq={res_s_sq['accumulated_fss']:.4f}, Cir={res_s_ci['accumulated_fss']:.4f}")

if __name__ == "__main__":
    main()
