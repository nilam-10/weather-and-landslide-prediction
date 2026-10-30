"""Audit script to verify scientific consistency in Phase 6 XAI outputs.
"""
import csv
import json
from pathlib import Path
import numpy as np

def audit():
    report_dir = Path("outputs/reports")
    val_csv = report_dir / "PHASE6_XAI_VALIDATION.csv"
    res_csv = report_dir / "PHASE6_XAI_EXPERIMENT_RESULTS.csv"
    npz_file = Path("outputs/experiments/xai/phase6_attributions_step316.npz")

    print("=== AUDIT: PHASE6_XAI_VALIDATION.csv ===")
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    for idx, r in enumerate(val_rows):
        exp_id = r["experiment_id"]
        if "COMPLETENESS" in exp_id or "BASELINE" in exp_id:
            fin = float(r["f_input"])
            fb = float(r["f_baseline"])
            df = float(r["delta_f"])
            s_attr = float(r["sum_attribution"])
            err_abs = float(r["completeness_error_abs"])
            err_pct = float(r["completeness_error_pct"])
            
            calc_df = round(fin - fb, 4)
            calc_err_abs = round(abs(s_attr - df), 4)
            calc_err_pct = round((calc_err_abs / abs(df)) * 100, 4)
            
            # Check numerical consistency
            df_match = abs(calc_df - df) < 1e-3
            err_match = abs(calc_err_abs - err_abs) < 1e-3
            pct_match = abs(calc_err_pct - err_pct) < 0.1
            
            print(f"[{exp_id}] fin={fin}, fb={fb}, df={df} (calc {calc_df}), sum={s_attr}, err_abs={err_abs} (calc {calc_err_abs}), err_pct={err_pct}% (calc {calc_err_pct:.3f}%) | Match: df={df_match}, err={err_match}, pct={pct_match}")

    print("\n=== AUDIT: npz file arrays ===")
    if npz_file.exists():
        data = np.load(npz_file)
        print("NPZ keys:", list(data.keys()))
        print("latitudes shape:", data["latitudes"].shape)
        print("longitudes shape:", data["longitudes"].shape)
        print("target_cell_row:", data["target_cell_row"], "target_cell_col:", data["target_cell_col"])
        print("ig_prob_spatial_total shape:", data["ig_prob_spatial_total"].shape, "sum:", np.sum(data["ig_prob_spatial_total"]))
        print("ig_qpf_spatial_total shape:", data["ig_qpf_spatial_total"].shape, "sum:", np.sum(data["ig_qpf_spatial_total"]))
        print("occ_prob_spatial shape:", data["occ_prob_spatial"].shape)
        if "ig_prob_raw_4d" in data:
            print("ig_prob_raw_4d shape:", data["ig_prob_raw_4d"].shape)
    else:
        print("NPZ file not found!")

    print("\n=== AUDIT: EXPERIMENT RESULTS CSV (Row counts & Top Channels) ===")
    with open(res_csv, "r", encoding="utf-8") as f:
        res_rows = list(csv.DictReader(f))
    print(f"Total experiment rows: {len(res_rows)}")
    cells = set(r["cell_name"] for r in res_rows)
    print("Cells tested:", cells)
    leads = set(r["lead_time"] for r in res_rows)
    print("Leads tested:", leads)
    methods = set(r["method"] for r in res_rows)
    print("Methods tested:", methods)
    targets = set(r["target_type"] for r in res_rows)
    print("Target types tested:", targets)

if __name__ == "__main__":
    audit()
