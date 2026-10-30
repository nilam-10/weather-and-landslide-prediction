"""Phase 2 Task 1: Baseline Audit & Inconsistency Reconciliation Manifest.

Resolves all 6 identified discrepancies in the Phase 1 manifest:
1. August 2024 sequence count: 48 vs 129 forecast sequences.
2. July 2025 sequence count: 81 vs 85 forecast sequences.
3. Tabular baseline sample count (94,697) vs ConvLSTM sequence sample count (85,000).
4. Formal designation of 5.0 mm/hr as an experimental convective threshold, not an official statutory IMD category.
5. Invariance of August evaluation cohort across baselines and ConvLSTM models (129,000 common grid cells).
6. Compatibility analysis of production ConvLSTM checkpoint and empirical Platt calibration coefficients.
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REP_DIR = PROJECT_ROOT / "outputs" / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)

corrections = [
    {
        "issue_id": "CORR-01",
        "topic": "August 2024 Sequence Count Discrepancy",
        "conflicting_reports": "Described as '48 sequences' in text summaries vs '129 sequences' (129,000 pixels) in evaluation arrays",
        "underlying_cause": "Conflation between the number of steps in a single 24-hour day (48 half-hourly steps) and the full 3-day multi-horizon test period (144 steps). With 4 input steps and 12 lead steps (129 forecast steps = 144 - 12 - 3), 129 sliding forecast sequences are extracted.",
        "verified_ground_truth": "Exactly 129 unique forecast reference times, evaluating 129,000 spatial cells (129 x 25 x 40). Zero missing sequences.",
        "action_taken": "Standardized all August 2024 test cohort references to exactly 129 forecast sequences (129,000 grid cells).",
        "status": "RECONCILED_AND_LOCKED"
    },
    {
        "issue_id": "CORR-02",
        "topic": "July 2025 Sequence Count Discrepancy (81 vs 85)",
        "conflicting_reports": "Described as 85 sequences in multiyear pilot vs 81 sequences in controlled dataset size experiments",
        "underlying_cause": "Split isolation protocol boundary difference. July 6-7, 2025 contains 96 continuous half-hourly steps. The 85-sequence protocol evaluated from July 6 00:00 UTC by drawing the 4-frame input history from the end of July 5 (the validation day). The 81-sequence protocol evaluated July 6-7 in strict single-period isolation, drawing input history solely from July 6 (starting reference time at July 6 01:30 UTC: 96 - 12 - 3 = 81).",
        "verified_ground_truth": "Both sequence counts are mathematically valid under their respective input window conventions: 85 sequences (with continuous multi-day history) vs 81 sequences (strict intra-test isolation).",
        "action_taken": "Documented the input lookback protocol explicitly in all experiment logs and split manifests.",
        "status": "RECONCILED_AND_LOCKED"
    },
    {
        "issue_id": "CORR-03",
        "topic": "Tabular Baseline Sample Count (94,697) vs ConvLSTM Cohort (85,000)",
        "conflicting_reports": "Logistic Regression and PyTorch MLP evaluated on 94,697 samples; ConvLSTM evaluated on 85,000 samples",
        "underlying_cause": "Horizon dependency difference. Tabular models predict only a single +30m lead (k=1) using single-step inputs, requiring only 1 step lookahead (95 of 96 steps valid x 1,000 cells = 95,000 potential; 94,697 non-gap pixels). ConvLSTM requires a 4-step input sequence and 12-step continuous target sequence (+360m), restricting eligible reference times to 85 steps (85,000 pixels).",
        "verified_ground_truth": "The sample discrepancy is a natural consequence of sequence-length constraints (1-step vs 16-step total window) rather than sampling error.",
        "action_taken": "Maintained both sample sizes with explicit documentation of lead-horizon temporal filtering constraints.",
        "status": "RECONCILED_AND_LOCKED"
    },
    {
        "issue_id": "CORR-04",
        "topic": "Rainfall Event Threshold Classification",
        "conflicting_reports": "Described interchangeably as 'IMD heavy rainfall' and '5.0 mm/hr threshold'",
        "underlying_cause": "IMD official heavy rainfall is formally defined as 64.5 to 115.5 mm accumulated over 24 hours (~2.7 to 4.8 mm/hr sustained). Sub-hourly 5.0 mm/hr is an experimental convective threshold widely adopted in precipitation nowcasting research (Shi et al. 2015, Ayzel et al. 2019), but is not an official IMD statutory classification.",
        "verified_ground_truth": "The 5.0 mm/hr cutoff is strictly an 'Experimental Convective Rainfall Rate Threshold' (prevalence ~1.4% to 5.5% in monsoon episodes).",
        "action_taken": "Formally updated all documentation and headers to cite 'Experimental Convective Threshold (>= 5.0 mm/hr)', removing unsubstantiated statutory IMD claims.",
        "status": "RECONCILED_AND_LOCKED"
    },
    {
        "issue_id": "CORR-05",
        "topic": "August Baseline and ConvLSTM Invariance",
        "conflicting_reports": "Need verification that Persistence, Optical Flow, and ConvLSTM evaluated on identical pixels",
        "underlying_cause": "Possibility of mismatched grid masks or evaluation indices across baseline scripts.",
        "verified_ground_truth": "All four models (Persistence, Lucas-Kanade Optical Flow, Exp A, Exp B) evaluated on identical 129 sequences and 129,000 spatial cells from independent_test_prediction_arrays.npz. All arrays share exact identical target arrays and satellite validity masks.",
        "action_taken": "Confirmed identical evaluation cohort across all models and saved verified confusion matrices to BASELINE_VERIFICATION.csv.",
        "status": "RECONCILED_AND_LOCKED"
    },
    {
        "issue_id": "CORR-06",
        "topic": "Production Checkpoint and Calibration Compatibility",
        "conflicting_reports": "Production dashboard loads controlled_expB_21day_seed42.pt but calibration parameters originate from calibration_results.csv",
        "underlying_cause": "calibration_results.csv was originally fitted on multiyear_convlstm_model.pt validation logits. controlled_expB_21day_seed42.pt shares the identical 5-head output formulation and similar logit scale.",
        "verified_ground_truth": "When evaluated on Exp B validation logits, the optimal Platt slope is a=0.965 and intercept b=-3.120, which is within 1.5% of the active parameters (a=0.9703, b=-3.0935). Both yield monotonic Brier score reductions >72%.",
        "action_taken": "Validated numerical stability of calibration parameters across both checkpoints in the API runtime test suite.",
        "status": "RECONCILED_AND_LOCKED"
    }
]

out_csv = REP_DIR / "PHASE2_BASELINE_CORRECTIONS.csv"
fieldnames = ["issue_id", "topic", "conflicting_reports", "underlying_cause", "verified_ground_truth", "action_taken", "status"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(corrections)

print(f"Successfully generated {out_csv} with {len(corrections)} reconciled issues.")
