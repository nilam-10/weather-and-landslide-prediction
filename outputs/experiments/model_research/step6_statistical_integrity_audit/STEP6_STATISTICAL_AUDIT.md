# Step 6 Master Audit: Statistical and Scientific Integrity Audit Report

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Audit Scope:** Verification of Step 4 (Loss Pilot) and Step 5 (Three-Seed Replication) Experiments  
**Auditor:** Senior Research Engineer & Scientific Integrity Reviewer  
**Audit Date:** October 30, 2026  
**Artifact Directory:** `outputs/experiments/model_research/step6_statistical_integrity_audit/`

---

## 1. Executive Audit Verdict

### Overall Status: ACCEPTED WITH DOCUMENTED LIMITATIONS & METHODOLOGICAL CORRECTIONS

The Step 5 three-seed replication experiment successfully establishes that **Compound Spatial Loss (Exp D: Intensity-Weighted L1 + Multiscale Differentiable Soft-FSS + BCE)** substantially and consistently outperforms the standard Smooth-L1 baseline (Exp A) at intermediate nowcasting horizons (+120m and +180m). The performance improvements replicate across all three evaluated random initializations (Seeds 42, 123, 999) without degrading bulk precipitation root-mean-squared error (RMSE).

However, this audit identifies **four critical methodological limitations** in how the Step 5 results were analyzed and reported:
1. **Uncalibrated Empirical $p$-values:** Step 5 reported exact values of `p = 0.000` from a 1,000-iteration Monte Carlo bootstrap. Empirical resampling cannot yield $p=0.000$; finite resolution bounds the minimum probability to $p < 0.001$.
2. **Conflation of Temporal and Seed Uncertainty:** The block bootstrap resampled temporal weather sequences, not random model initializations. Statistical significance applies to **temporal episode sampling**, while across-seed consistency is supported by empirical replication across $N=3$ seeds ($s_\Delta = 0.0089$ at +180m). Three seeds alone cannot establish a non-parametric $p < 0.05$.
3. **Under-Decorrelated Block Size for 6-Hour Forecasts:** A block length of 3.0 hours ($L=6$) was used across all horizons. Because the +360m forecast horizon spans 6.0 hours, adjacent blocks overlapped in their target verification trajectories. Full decorrelation requires a 6.0-hour block ($L=12$).
4. **Cohort Transparency:** The confirmatory test cohort (July 12–14, 2023, 129 sequences) was evaluated in earlier project phases and must be described as a locked benchmark rather than an unexposed blind test.

All four issues have been forensically recomputed, corrected, and documented in the accompanying audit artifacts:
- [`SOURCE_VALUE_RECONCILIATION.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/SOURCE_VALUE_RECONCILIATION.csv)
- [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv)
- [`BOOTSTRAP_METHOD_AUDIT.md`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/BOOTSTRAP_METHOD_AUDIT.md)
- [`DATA_SPLIT_AND_TEMPORAL_DEPENDENCE_AUDIT.md`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/DATA_SPLIT_AND_TEMPORAL_DEPENDENCE_AUDIT.md)
- [`THESIS_SAFE_CLAIMS.md`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/THESIS_SAFE_CLAIMS.md)
- [`FINAL_RESEARCH_DECISION.md`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/FINAL_RESEARCH_DECISION.md)

---

## 2. Source Data Verification & Recomputation

### 2.1 Recomputation of Aggregate Statistics
All aggregate figures reported in `VALIDATION_RESULTS_MEAN_STD.csv`, `PAIRED_SEED_COMPARISON.csv`, and `STEP5_REPLICATION_REPORT.md` were recomputed directly from the 30 per-seed records in `VALIDATION_RESULTS_PER_SEED.csv` using sample statistics ($\text{ddof} = 1$).

**Findings:**
- **Zero Mathematical Discrepancy:** All recalculated means and standard deviations match the reported Step 5 source tables to within machine precision ($\le 3.3 \times 10^{-5}$, due solely to 4-decimal-place rounding).
- **Paired Gain Exactness:** The paired CSI differences ($\Delta_s = \text{ExpD}_s - \text{ExpA}_s$) are strictly positive for all three seeds at +30m, +60m, +120m, and +180m. At +180m on Validation, the paired differences are $+0.0835$ (Seed 42), $+0.0936$ (Seed 123), and $+0.1013$ (Seed 999), yielding an exact mean gain of $+0.0928 \pm 0.0089$.
- **FSS Recalculation:** Spatial Fractions Skill Scores recomputed via the corrected circular Euclidean kernel ($k=3$ cells = 29 grid points, $30\text{ km}$ nominal radius) match the reported values in `SPATIAL_VERIFICATION_RESULTS.csv` perfectly.

---

### 2.2 Forensic Reconciliation: Step 4 Pilot vs. Step 5 Replication (Seed 42)
An apparent shift was noted between the Step 4 pilot and the Step 5 replication for Seed 42:
- In Step 4, Exp A Seed 42 achieved +30m $\text{CSI} = 0.3739$ and +180m $\text{CSI} = 0.0374$.
- In Step 5, Exp A Seed 42 achieved +30m $\text{CSI} = 0.3203$ and +180m $\text{CSI} = 0.0135$.
- In Step 4, Exp D Seed 42 achieved +180m $\text{CSI} = 0.1352$.
- In Step 5, Exp D Seed 42 achieved +180m $\text{CSI} = 0.0970$.

**Forensic Investigation:**
1. In Step 4 (`scripts/run_step4_loss_pilot_experiment.py`), a single global PyTorch RNG was used across sequential training of four model configurations (Exp A, B, C, D). As a result, the mini-batch shuffling sequence across epochs was influenced by prior training loops.
2. In Step 5 (`scripts/run_step5_three_seed_replication.py`), a dedicated `torch.Generator().manual_seed(seed)` was instantiated inside each run loop to enforce strict, isolated determinism.
3. For Exp A, this slight difference in batch order caused the minimum validation loss checkpoint to be selected at **Epoch 6** in Step 5 ($\text{val\_loss} = 0.4461$) versus **Epoch 14** in Step 4 ($\text{val\_loss} = 0.4411$), representing a trivial loss difference of $0.0050$.
4. For Exp D, both Step 4 and Step 5 converged to the exact same best epoch: **Epoch 9** (Step 4 loss = $0.8303$, Step 5 loss = $0.8299$).
5. **Conclusion:** This shift is entirely attributable to standard stochastic mini-batch trajectory variance under isolated seeding. It falls within normal stochastic optimization bounds and confirms the importance of multi-seed replication.

---

## 3. Bootstrap Statistical Analysis Audit

### 3.1 Audit Findings on Step 5 Bootstrap
1. **Resampling Alignment:** Pairing was rigorously maintained. Within each bootstrap replicate, Exp A and Exp D were evaluated on identical resampled temporal blocks.
2. **P-Value Bounding:** Step 5's report of `p = 0.000` is invalid. With $B = 1,000$ iterations, the maximum resolution is $1/B$. The continuity-corrected bound is $p < 0.001$.
3. **Inferential Scope:** The bootstrap quantified temporal sampling uncertainty across weather sequences for fixed trained checkpoints. It did not test variance across random initializations.
4. **Decorrelation Block Length:** For +360m (6 hours ahead), 3.0-hour blocks ($L=6$) suffer from residual overlap correlation. Re-evaluating with 6.0-hour blocks ($L=12$) decorrelates the entire forecast horizon.

### 3.2 Summary of Corrected Recomputations
From [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv) on the 129-sequence confirmatory test benchmark with complete 6.0-hour block decorrelation ($L=12$):

| Horizon | Metric | Exp A Point | Exp D Point | Point $\Delta$ | 95% Bootstrap CI | Contrary Replicates ($k$) | Corrected $p$-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+120m** | Pointwise CSI (Seed 42) | $0.1057$ | $0.1829$ | $+0.0772$ | $[+0.0461, +0.1039]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ |
| | Pointwise CSI (Seed 123) | $0.0937$ | $0.2395$ | $+0.1458$ | $[+0.1057, +0.1772]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ |
| | Pointwise CSI (Seed 999) | $0.1704$ | $0.2176$ | $+0.0472$ | $[+0.0151, +0.0818]$ | $1 / 1000$ | $\mathbf{p = 0.0020}$ |
| | **Three-Seed Pooled CSI** | **N/A** | **N/A** | $\mathbf{+0.0901}$ | $\mathbf{[+0.0633, +0.1139]}$ | $\mathbf{0 / 1000}$ | $\mathbf{p < 0.001}$ |
| **+180m** | Pointwise CSI (Seed 42) | $0.0050$ | $0.1050$ | $+0.1000$ | $[+0.0385, +0.1557]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ |
| | Pointwise CSI (Seed 123) | $0.0038$ | $0.1506$ | $+0.1468$ | $[+0.0784, +0.2078]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ |
| | Pointwise CSI (Seed 999) | $0.0272$ | $0.1343$ | $+0.1071$ | $[+0.0526, +0.1528]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ |
| | **Three-Seed Pooled CSI** | **N/A** | **N/A** | $\mathbf{+0.1180}$ | $\mathbf{[+0.0652, +0.1654]}$ | $\mathbf{0 / 1000}$ | $\mathbf{p < 0.001}$ |
| **+180m** | **Three-Seed Pooled FSS** | **N/A** | **N/A** | $\mathbf{+0.3807}$ | $\mathbf{[+0.2241, +0.4789]}$ | $\mathbf{0 / 1000}$ | $\mathbf{p < 0.001}$ |
| **+180m** | **Three-Seed Pooled RMSE**| **N/A** | **N/A** | $\mathbf{-0.0008}$ | $\mathbf{[-0.0617, +0.0545]}$ | $\mathbf{449 / 1000}$ | $\mathbf{p = 0.449}$ |

---

## 4. Evaluation Cohort and Data Split Audit

1. **Manifest Integrity:**
   - Training split: 38 days ($N = 1,764$ sequences), spanning active monsoon periods in 2023, 2024, and 2025.
   - Validation split: July 5, 2025 ($N = 33$ sequences, 16.5 active rain hours).
   - Confirmatory test split: July 12–14, 2023 ($N = 129$ sequences, 72.0 total observation hours).
2. **64.5-Hour Figure Reconciliation:**
   - The test set reference timestamps span from `2023-07-12T02:00:00Z` to `2023-07-14T18:00:00Z`, an elapsed duration of **64.0 hours**.
   - Multiplying 129 sequences $\times 0.5\text{ hours/step} = 64.5\text{ hours}$.
   - Including the 6-hour forecast horizon of the last sequence, the total contiguous meteorological observation window is **72.0 hours (3 full calendar days)**.
3. **Zero Information Leakage:**
   - Causal sequence builders enforce strict timestamp continuity checks. No sequence spans gaps between non-contiguous dates, and no input or target window crosses split boundaries.
   - Normalization constants ($\mu_{T_b} = 268.442, \sigma_{T_b} = 20.619, \mu_P = 0.709, \sigma_P = 2.110$) were computed once strictly on training data. Target precipitation values were maintained in raw $\text{mm/hr}$.
4. **Cohort Status Disclosure:**
   - July 12–14, 2023 is a fixed, locked benchmark that was evaluated in earlier project phases. It must be disclosed as a previously inspected confirmatory benchmark rather than a blind, unseen dataset.

---

## 5. Intensity and Spatial Skill Audit

1. **Intensity Attenuation Cure:**
   - Exp A suffers severe conditional mean blur, attenuating +180m forecast maxima to $7.06\text{ mm/hr}$ (observed peak: $48.58\text{ mm/hr}$) and reducing 99th percentile rainfall to $3.05\text{ mm/hr}$ (observed: $10.31\text{ mm/hr}$).
   - Exp D retains sharp convective peaks: average forecast peak of $13.15\text{ mm/hr}$ and 99th percentile of $6.17\text{ mm/hr}$.
2. **Spatial Core Displacement:**
   - At +180m, the average displacement between observed and predicted convective cores is **$109.8\text{ km}$** (~10 grid cells).
   - This large displacement explains why Pointwise CSI at +180m is $0.1111$, while Circular FSS at 30 km is $0.4016$. The model successfully predicts convective cells within the general regional domain, but with an offset of ~100 km over 3 hours of advection.
3. **+360m Forecast Horizon Reality:**
   - Exp D prevents total zero-out at +360m (restoring maximum rain rate to $7.47\text{ mm/hr}$ and $\text{FSS} = 0.0193$ to $0.0526$, whereas Exp A collapsed to $0.0000$).
   - However, an FSS of $0.05$ is far below the useful forecasting threshold of $\sim 0.50$. The model has weak absolute skill at 6 hours and cannot be considered an operational 6-hour convective nowcasting system.

---

## 6. Thesis-Safe Reporting Protocol

To ensure academic rigor, all findings incorporated into the B.Tech Honours thesis must adhere to the following standards:
1. **Never use hyperbolic terminology:** Eliminate words like "unequivocally", "proven", "flawless", or "solved".
2. **Never report $p = 0.000$:** Report $p < 0.001$ with continuity-corrected fractions.
3. **Distinguish temporal significance from seed variance:** State that temporal episode significance was established via block bootstrapping, while seed consistency was demonstrated across $N=3$ random initializations.
4. **Disclose benchmark status:** Acknowledge that the test split is a locked historical benchmark rather than an unseen blind test.
5. **No coupled landslide claims:** Disclose that nowcasting improvements have not yet been evaluated through the downstream Phase 4 landslide engine.
