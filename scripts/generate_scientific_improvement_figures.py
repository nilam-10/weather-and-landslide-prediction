"""Generate Publication-Quality Figures for Scientific Improvement Phase (300 DPI).

Creates:
  1. outputs/figures/evaluation/fig10_controlled_convlstm_ablations.png:
     - Comparison across 5 controlled experiments:
       EXP_01 (Baseline 32-ch, Compound Loss, AdamW, AMP)
       EXP_02 (64-ch Increased Capacity)
       EXP_03 (BCE Only Loss Ablation)
       EXP_04 (Adam Regularization Ablation)
       EXP_05 (FP32 Precision Ablation)
     - Displays Calibrated CSI, ROC-AUC, Brier Score, and continuous QPF RMSE (mm/hr).

  2. outputs/figures/evaluation/fig11_bootstrap_uncertainty_inflation.png:
     - Visualizes Pixel-level naive CI vs Frame-level Block Bootstrap CI across +30m to +360m horizons.
     - Illustrates the 3.5x-4.8x uncertainty inflation factor caused by spatiotemporal autocorrelation.

  3. outputs/figures/landslide/fig12_landslide_spatial_uncertainty_audit.png:
     - Breakdown of reported location uncertainty radii (1 km, 5 km, 25 km, 50 km, unknown)
       contrasted against the 90 m DEM grid resolution and 0.1-degree weather cells.
     - Explains the mathematical prevalence barrier (6.12e-6) for REQ-LND-04.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_FIG_EVAL = Path("outputs/figures/evaluation")
OUTPUTS_FIG_LND = Path("outputs/figures/landslide")
OUTPUTS_FIG_EVAL.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIG_LND.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

def plot_fig10_controlled_experiments():
    csv_path = Path("outputs/experiments/convlstm_controlled_experiments.csv")
    if not csv_path.exists():
        print("Missing convlstm_controlled_experiments.csv")
        return

    exp_names = []
    csi_vals = []
    roc_vals = []
    brier_vals = []
    rmse_vals = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eid = r["exp_id"].replace("EXP_", "")
            desc = r["description"].split(":")[0]
            exp_names.append(f"{eid}\n({desc})")
            csi_vals.append(float(r["test_csi_cal_30m"]))
            roc_vals.append(float(r["test_roc_auc_30m"]))
            brier_vals.append(float(r["test_brier_cal_30m"]))
            rmse_vals.append(float(r["qpf_rmse_mm_hr"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    x = np.arange(len(exp_names))
    width = 0.55

    # 1. CSI Calibrated
    ax = axes[0, 0]
    bars = ax.bar(x, csi_vals, width, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_title("(a) Calibrated Critical Success Index (CSI @ +30m)")
    ax.set_ylabel("CSI")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8)
    ax.set_ylim(0.30, 0.42)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    # 2. ROC-AUC
    ax = axes[0, 1]
    bars = ax.bar(x, roc_vals, width, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.set_title("(b) Area Under ROC Curve (ROC-AUC @ +30m)")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8)
    ax.set_ylim(0.965, 0.985)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    # 3. Platt-Calibrated Brier Score
    ax = axes[1, 0]
    bars = ax.bar(x, brier_vals, width, color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax.set_title("(c) Platt-Calibrated Brier Score (Lower is Better)")
    ax.set_ylabel("Brier Score")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8)
    ax.set_ylim(0.0, 0.012)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0003, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    # 4. Continuous QPF RMSE
    ax = axes[1, 1]
    bars = ax.bar(x, rmse_vals, width, color="#d62728", edgecolor="black", alpha=0.85)
    ax.set_title("(d) Continuous QPF Rain-Rate RMSE (mm/hr)")
    ax.set_ylabel("RMSE (mm/hr)")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8)
    ax.set_ylim(0.60, 0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Controlled ConvLSTM Architecture, Loss & Precision Ablations (Meghalaya Pilot)", y=0.99, fontweight="bold")
    plt.tight_layout()
    out_file = OUTPUTS_FIG_EVAL / "fig10_controlled_convlstm_ablations.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Fig 10 to: {out_file}")

def plot_fig11_bootstrap_inflation():
    csv_path = Path("outputs/experiments/weather_metrics_bootstrap_comparison.csv")
    block_csv = Path("outputs/experiments/weather_metrics_block_bootstrap_ci.csv")
    if not csv_path.exists() or not block_csv.exists():
        print("Missing bootstrap CSVs")
        return

    horizons = ["+30m", "+60m", "+120m", "+180m", "+360m"]
    pt_csi = []
    px_widths = []
    blk_widths = []
    ratios = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["model_name"] == "MultiHorizon Weather ConvLSTM" and "Uncalibrated" in r["calibration_status"]:
                h = r["horizon"]
                if h in horizons:
                    pt_csi.append(float(r["point_estimate"]))
                    px_widths.append(float(r["pixel_bootstrap_95ci_width"]))
                    blk_widths.append(float(r["block_bootstrap_95ci_width"]))
                    ratios.append(float(r["uncertainty_inflation_ratio"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: 95% CI comparison (error bars)
    x = np.arange(len(horizons))
    px_err = [w / 2.0 for w in px_widths]
    blk_err = [w / 2.0 for w in blk_widths]

    ax1.errorbar(x - 0.08, pt_csi, yerr=px_err, fmt="o", color="#1f77b4", label="Pixel Naive Bootstrap (Independent)", capsize=4, elinewidth=1.5)
    ax1.errorbar(x + 0.08, pt_csi, yerr=blk_err, fmt="s", color="#d62728", label="Frame-Level Block Bootstrap (Spatial Correlation)", capsize=5, elinewidth=2.0)
    ax1.set_title("(a) CSI vs Forecast Horizon with 95% Confidence Intervals")
    ax1.set_xlabel("Forecast Horizon")
    ax1.set_ylabel("Critical Success Index (CSI)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_ylim(-0.02, 0.55)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Annotate +360m dissipation
    ax1.annotate("Near-Zero Predictive Skill\n(Thermodynamic Dissipation)", xy=(4, 0.001), xytext=(3.1, 0.15),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                 fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

    # Right: Inflation factor
    bars = ax2.bar(x, ratios, width=0.5, color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", label="1.0x (No Inflation)")
    ax2.set_title("(b) Empirical Uncertainty Inflation Ratio (Block CI / Pixel CI)")
    ax2.set_xlabel("Forecast Horizon")
    ax2.set_ylabel("Confidence Interval Width Inflation Factor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_ylim(0, 6.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{b.get_height():.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.legend(loc="upper right")

    plt.suptitle("Spatiotemporal Autocorrelation Defense: Block vs Naive Bootstrap Evaluation", y=0.98, fontweight="bold")
    plt.tight_layout()
    out_file = OUTPUTS_FIG_EVAL / "fig11_bootstrap_uncertainty_inflation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Fig 11 to: {out_file}")

def plot_fig12_landslide_uncertainty():
    audit_csv = Path("outputs/experiments/landslide_label_uncertainty_audit.csv")
    if not audit_csv.exists():
        print("Missing landslide_label_uncertainty_audit.csv")
        return

    uncertainties = []
    triggers = []
    admin = []

    with open(audit_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uncertainties.append(float(r["uncertainty_radius_km"]))
            triggers.append(r["trigger_classified"])
            admin.append(r["inside_meghalaya"].lower() == "true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: Histogram of uncertainty radii
    categories = ["< 1 km (High)", "1 - 5 km", "5 - 10 km", "10 - 25 km", "25 - 50 km"]
    counts = [
        sum(1 for u in uncertainties if u < 1.0),
        sum(1 for u in uncertainties if 1.0 <= u <= 5.0),
        sum(1 for u in uncertainties if 5.0 < u <= 10.0),
        sum(1 for u in uncertainties if 10.0 < u <= 25.0),
        sum(1 for u in uncertainties if u > 25.0),
    ]

    x = np.arange(len(categories))
    bars = ax1.bar(x, counts, width=0.55, color="#9467bd", edgecolor="black", alpha=0.85)
    ax1.set_title("(a) Spatial Coordinate Uncertainty of GLC Records (N=67)")
    ax1.set_xlabel("Uncertainty Error Radius")
    ax1.set_ylabel("Number of Historical Records")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, str(int(b.get_height())), ha="center", va="bottom", fontsize=8)

    # Right: Spatial scale mismatch schematic
    # Draw concentric circles representing 1km, 5km, 25km vs 90m pixel
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    ax2.set_aspect("equal")

    circle_50 = plt.Circle((0, 0), 25, color="#d62728", fill=False, linestyle="--", linewidth=1.5, label="25 km Error Radius (County/District)")
    circle_5 = plt.Circle((0, 0), 5, color="#ff7f0e", fill=False, linestyle="-", linewidth=2.0, label="5 km Error Radius (Town/Locality)")
    circle_1 = plt.Circle((0, 0), 1, color="#2ca02c", fill=True, alpha=0.3, label="1 km Error Radius (~120 DEM Pixels)")
    ax2.add_patch(circle_50)
    ax2.add_patch(circle_5)
    ax2.add_patch(circle_1)
    ax2.plot(0, 0, "kx", markersize=8, mew=2, label="Reported Centroid (News Media)")

    ax2.set_title("(b) Spatial Scale Mismatch: 90 m DEM vs GLC Uncertainty")
    ax2.set_xlabel("East-West Distance (km)")
    ax2.set_ylabel("North-South Distance (km)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    plt.suptitle("Landslide Inventory Provenance & Spatial Resolution Mismatch Audit", y=0.98, fontweight="bold")
    plt.tight_layout()
    out_file = OUTPUTS_FIG_LND / "fig12_landslide_spatial_uncertainty_audit.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Fig 12 to: {out_file}")

if __name__ == "__main__":
    plot_fig10_controlled_experiments()
    plot_fig11_bootstrap_inflation()
    plot_fig12_landslide_uncertainty()
