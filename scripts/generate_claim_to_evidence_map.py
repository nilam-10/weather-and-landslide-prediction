"""Generate CLAIM_TO_EVIDENCE_MAP.csv linking every quantitative claim to source artifacts."""
import csv
from pathlib import Path

OUT_DIR = Path("outputs/reports/thesis_weather_model_integration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

claims = [
    {
        "claim_id": "CLM-001",
        "claim_domain": "Intermediate Convective Nowcasting (+180m)",
        "statement_in_thesis": "Compound loss Exp D improves +180m Pointwise CSI from 0.0183 to 0.1111 (6.07x) on validation.",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+180m, threshold >= 5.0 mm/hr",
        "reconciled_numerical_value": "ExpA: 0.0183 +/- 0.0107, ExpD: 0.1111 +/- 0.0184, Delta: +0.0928 +/- 0.0089",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/VALIDATION_RESULTS_MEAN_STD.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Report as mean +/- std across N=3 seeds. Do not claim universal generalization."
    },
    {
        "claim_id": "CLM-002",
        "claim_domain": "Intermediate Spatial Verification (+180m)",
        "statement_in_thesis": "Compound loss Exp D improves +180m Circular FSS (30 km) from 0.0740 to 0.4016 (5.43x) on validation.",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+180m, radius 30 km (k=3), threshold >= 5.0 mm/hr",
        "reconciled_numerical_value": "ExpA: 0.0740 +/- 0.0535, ExpD: 0.4016 +/- 0.0523, Delta: +0.3276 +/- 0.0055",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/PAIRED_SEED_COMPARISON.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Evaluated using corrected circular Euclidean kernel (29 cells). Paired across matching seeds."
    },
    {
        "claim_id": "CLM-003",
        "claim_domain": "Intermediate Convective Nowcasting (+120m)",
        "statement_in_thesis": "Compound loss Exp D improves +120m Pointwise CSI from 0.0796 to 0.1685 (2.12x) on validation.",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+120m, threshold >= 5.0 mm/hr",
        "reconciled_numerical_value": "ExpA: 0.0796 +/- 0.0300, ExpD: 0.1685 +/- 0.0149, Delta: +0.0889 +/- 0.0223",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/VALIDATION_RESULTS_MEAN_STD.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Positive paired gains in all 3 seeds (42, 123, 999)."
    },
    {
        "claim_id": "CLM-004",
        "claim_domain": "Intermediate Spatial Verification (+120m)",
        "statement_in_thesis": "Compound loss Exp D improves +120m Circular FSS (30 km) from 0.3232 to 0.5560 (+72%) on validation, crossing the operational useful threshold.",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+120m, radius 30 km (k=3), threshold >= 5.0 mm/hr",
        "reconciled_numerical_value": "ExpA: 0.3232 +/- 0.1158, ExpD: 0.5560 +/- 0.0307, Delta: +0.2328 +/- 0.0893",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/SPATIAL_VERIFICATION_RESULTS.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Useful threshold FSS_useful = 0.5 + f0/2 ~ 0.52."
    },
    {
        "claim_id": "CLM-005",
        "claim_domain": "Out-of-Sample Benchmark Generalization (+180m)",
        "statement_in_thesis": "On 3-day confirmatory benchmark, +180m CSI improves from 0.0120 to 0.1300 (10.8x) and FSS (30 km) from 0.0527 to 0.4333 (8.2x).",
        "primary_split_cohort": "ConfirmatoryTest_20230712_14 (N=129)",
        "horizon_and_threshold": "+180m, threshold >= 5.0 mm/hr, radius 30 km",
        "reconciled_numerical_value": "CSI: 0.0120 +/- 0.0132 -> 0.1300 +/- 0.0231; FSS: 0.0527 +/- 0.0556 -> 0.4333 +/- 0.0424",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/TEST_DIAGNOSTICS_PER_SEED.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Must disclose that July 12-14, 2023 is a locked historical benchmark, not a blind unseen test."
    },
    {
        "claim_id": "CLM-006",
        "claim_domain": "Bulk Regression Error Preservation",
        "statement_in_thesis": "Compound loss Exp D maintains bulk QPF RMSE parity with baseline Exp A across all horizons, with no degradation.",
        "primary_split_cohort": "ConfirmatoryTest_20230712_14 (N=129)",
        "horizon_and_threshold": "+180m, all valid grid cells",
        "reconciled_numerical_value": "ExpA RMSE: 1.5224 +/- 0.0122 mm/hr, ExpD RMSE: 1.5216 +/- 0.0165 mm/hr, Delta: -0.0008 mm/hr",
        "primary_source_artifact": "outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Bootstrap difference is statistically insignificant (p = 0.4490), confirming exact bulk error parity."
    },
    {
        "claim_id": "CLM-007",
        "claim_domain": "Long Horizon Nowcasting (+360m / 6 Hours)",
        "statement_in_thesis": "At +360m, Exp D restores non-zero convective structure over baseline extinction, but absolute skill remains weak.",
        "primary_split_cohort": "ConfirmatoryTest_20230712_14 (N=129)",
        "horizon_and_threshold": "+360m, threshold >= 5.0 mm/hr, radius 30 km",
        "reconciled_numerical_value": "CSI: 0.0000 -> 0.0125 +/- 0.0129; FSS: 0.0000 -> 0.0526 +/- 0.0495; Mean peak: 7.47 mm/hr",
        "primary_source_artifact": "outputs/experiments/model_research/step6_statistical_integrity_audit/STEP6_STATISTICAL_AUDIT.md",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Never describe +360m as operationally reliable. State clearly that FSS ~ 0.05 is far below FSS_useful ~ 0.53."
    },
    {
        "claim_id": "CLM-008",
        "claim_domain": "Convective Intensity Retention",
        "statement_in_thesis": "Exp D prevents conditional mean blur, retaining +180m forecast maximum rain rate of 13.15 mm/hr vs baseline 7.06 mm/hr.",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+180m, domain-wide distribution diagnostics",
        "reconciled_numerical_value": "Obs Max: 48.58 mm/hr, ExpA Max: 7.06 mm/hr, ExpD Max: 13.15 mm/hr, ExpD 99th pct: 6.17 mm/hr",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/PREDICTION_INTENSITY_DIAGNOSTICS.csv",
        "audit_verification_status": "Supported but Limited",
        "mandatory_caveats_and_wording_rules": "Explicitly state that maxima are domain-wide distribution diagnostics, not proof of predicting exact storm cell peak location."
    },
    {
        "claim_id": "CLM-009",
        "claim_domain": "Storm Core Centroid Displacement",
        "statement_in_thesis": "Matched convective storm core displacement averages approximately 110 km at +180m on validation, with median displacement of 15 km at +30m and 24 km at +60m on test.",
        "primary_split_cohort": "Validation_20250705 & ConfirmatoryTest_20230712_14",
        "horizon_and_threshold": "+30m to +180m, argmax matched frames >= 5.0 mm/hr",
        "reconciled_numerical_value": "Val +180m Mean: 109.9 km, Med: 73.3 km; Test +30m Med: 14.9 km; Test +60m Med: 24.3 km; Test +180m Med: 157.1 km",
        "primary_source_artifact": "outputs/reports/thesis_weather_model_integration/METHODOLOGY_AND_STATISTICAL_VALIDATION.md",
        "audit_verification_status": "Supported but Limited",
        "mandatory_caveats_and_wording_rules": "Must disclose that this is a single-peak argmax heuristic on matched frames, not an object-based multi-cell tracking algorithm."
    },
    {
        "claim_id": "CLM-010",
        "claim_domain": "Statistical Inferential Significance",
        "statement_in_thesis": "Stationary temporal block bootstrapping (L=12 sequences = 6.0 hours) confirms paired CSI and FSS gains at +120m and +180m are statistically significant (p < 0.001).",
        "primary_split_cohort": "ConfirmatoryTest_20230712_14 (N=129)",
        "horizon_and_threshold": "+120m and +180m, L=12 blocks, B=1000 resamples",
        "reconciled_numerical_value": "Pooled +180m CSI 95% CI: [+0.0652, +0.1654], p < 0.001; Pooled +180m FSS 95% CI: [+0.2241, +0.4789], p < 0.001",
        "primary_source_artifact": "outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv",
        "audit_verification_status": "Clarified & Bounded (Step 6 Audit)",
        "mandatory_caveats_and_wording_rules": "Never report p = 0.000. State that significance applies to temporal episode sampling, not across-seed initialization variance."
    },
    {
        "claim_id": "CLM-011",
        "claim_domain": "Across-Seed Initialization Reproducibility",
        "statement_in_thesis": "Convective gains replicate across all three independent random seeds (42, 123, 999) with small across-seed variance (s_delta = 0.0089 for CSI at +180m).",
        "primary_split_cohort": "Validation_20250705 (N=33)",
        "horizon_and_threshold": "+180m, paired differences across Seeds 42, 123, 999",
        "reconciled_numerical_value": "Delta Seed 42: +0.0835, Delta Seed 123: +0.0936, Delta Seed 999: +0.1013, Mean Delta: +0.0928, Std: 0.0089",
        "primary_source_artifact": "outputs/experiments/model_research/step5_three_seed_replication/PAIRED_SEED_COMPARISON.csv",
        "audit_verification_status": "Verified Directly from Source Data",
        "mandatory_caveats_and_wording_rules": "Do not claim that N=3 seeds alone establish inferential hypothesis test proof across the space of initializations."
    },
    {
        "claim_id": "CLM-012",
        "claim_domain": "Downstream Landslide Coupling Boundary",
        "statement_in_thesis": "Weather nowcasting improvements have not been evaluated through the downstream Phase 4 landslide susceptibility engine.",
        "primary_split_cohort": "System Integration Boundary",
        "horizon_and_threshold": "Next-24-hour landslide hazard integration",
        "reconciled_numerical_value": "Weather model evaluated independently; downstream coupling unevaluated in Step 5",
        "primary_source_artifact": "outputs/reports/thesis_weather_model_integration/LIMITATIONS_AND_FUTURE_WORK.md",
        "audit_verification_status": "Clarified Boundary (Methodological Rule)",
        "mandatory_caveats_and_wording_rules": "Prohibit claims that weather nowcasting improvements improved landslide prediction accuracy or warning lead time."
    }
]

out_csv = OUT_DIR / "CLAIM_TO_EVIDENCE_MAP.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(claims[0].keys()))
    writer.writeheader()
    writer.writerows(claims)

print(f"Generated {len(claims)} claim-to-evidence mappings in {out_csv.name}")
