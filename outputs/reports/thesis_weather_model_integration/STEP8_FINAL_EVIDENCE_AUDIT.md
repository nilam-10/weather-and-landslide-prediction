# STEP 8: FINAL SCIENTIFIC AND STATISTICAL EVIDENCE AUDIT

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Domain:** Spatiotemporal Meteorological Nowcasting & Scientific Integrity Audit  
**Target Submission:** B.Tech (Honours) Major Project Thesis  
**Date:** October 30, 2026  
**Auditor:** Independent Research Integrity Auditor  
**Audit Status:** Complete and Fully Reconciled  
**Directory:** `outputs/reports/thesis_weather_model_integration/`

---

## 1. EXECUTIVE AUDIT SUMMARY

This audit independently scrutinizes the scientific claims, mathematical formulations, statistical inferences, and dataset structures assembled in Step 7 and previous research phases. It enforces strict academic integrity: no model retraining, no checkpoint alteration, zero tolerance for fabricated significance or false equivalence, and rigorous disclosure of physical and methodological boundaries.

### Summary of Audit Findings:

1. **FSS Operational-Utility Threshold (Claim A):**
   - **Status:** **DISCREPANCY RESOLVED — THRESHOLD CLAIM FORMALLY REMOVED.**
   - The assertion that a +120-minute FSS of $0.5560$ crosses a pre-established "operational-utility threshold" of $\approx 0.52$ was an *ex post* invocation of Roberts & Lean's (2008) theoretical rule of thumb ($0.5 + f_0/2$). No formal operational threshold was pre-specified in the project synopsis or experimental protocol. The operational certification claim has been removed, while the measured comparative gain ($+72.0\%$, $0.3232 \to 0.5560$) is preserved.
2. **Bulk QPF RMSE Interpretation (Claim B):**
   - **Status:** **PHRASING CORRECTED — EQUIVALENCE CLAIM QUALIFIED.**
   - At +180m on the historical benchmark, Exp A RMSE is $1.5224 \pm 0.0122\text{ mm/hr}$ and Exp D RMSE is $1.5216 \pm 0.0165\text{ mm/hr}$ ($\bar{\Delta} = -0.0008\text{ mm/hr}$, 95% bootstrap CI: $[-0.0616, +0.0483]$, $p = 0.4166$ under $L=12$). Failing to reject the null hypothesis does *not* prove equivalence. Claims stating "proves no increase in error" have been replaced with precise phrasing: "no statistically significant difference was detected ($p \ge 0.40$), indicating bulk error was preserved without detectable degradation at the evaluated sample size."
3. **Storm-Core Centroid Displacement Diagnostic (Claim C):**
   - **Status:** **VERIFIED & METHODOLOGICALLY CLARIFIED.**
   - Verified that the metric evaluates the single domain-wide peak coordinate (`argmax`), *not* matched storm objects. Detailed re-analysis confirmed that **100% of cases (9 of 9 on validation)** with large displacements ($> 100\text{ km}$) exhibited multiple active convective cores (4 to 22 observed peaks, 1 to 9 forecast peaks). At short horizons, median displacement is tightly localized ($14.9\text{ km}$ at +30m, $27.2\text{ km}$ at +60m on test), while at +180m mean displacement is $109.9\text{ km}$ on validation and $156.6\text{ km}$ on test. Fractions Skill Scores across 10, 20, 30, and 50 km were retrieved directly from source CSVs.
4. **Three-Seed Replication & Statistical Uncertainty (Claim D):**
   - **Status:** **VERIFIED & PROPERLY DISTINGUISHED.**
   - Clarified that the three random seeds ($42, 123, 999$) measure initialization variance, while temporal block bootstrapping ($L=12$ frames / 6.0 hours, $B=1,000$ resamples) measures episode sampling uncertainty conditional on the evaluated checkpoints. Bootstrap significance ($p < 0.001$) cannot be generalized across all possible initializations.
5. **Six-Hour Predictability Ceiling (+360m) (Claim E):**
   - **Status:** **VERIFIED & RIGOROUSLY BOUNDED.**
   - While Exp D overcomes total convective extinction ($\text{CSI} = 0.0125 \pm 0.0129$, $\text{FSS} = 0.0526 \pm 0.0495$, mean maximum $7.47\text{ mm/hr}$), an FSS of $\approx 0.05$ represents very weak absolute skill. Operational viability at 6 hours is explicitly rejected.
6. **Maximum Predicted Rain Rates (Claim F):**
   - **Status:** **RECONCILED ACROSS STEPS 4, 5, AND 6.**
   - Differences between Step 4 pilot figures and Step 5 multi-seed figures are traced to isolated PyTorch generator seeding in Step 5 vs. global RNG in Step 4. Maximum rain rate is defined strictly as a distribution heavy-tail diagnostic, not storm-cell accuracy.
7. **Dataset Splits and Generalization (Claim G):**
   - **Status:** **VERIFIED & CHRONOLOGICAL LIMITATION NOTED.**
   - Confirmed training split ($N=1,764$ sequences, 38 days across 2023–2025), validation split ($N=33$ sequences, July 5, 2025), and historical benchmark ($N=129$ sequences, July 12–14, 2023). Explicitly disclosed that July 2023 was evaluated in earlier phases and chronologically precedes some training data.

---

## 2. AUDIT ITEM A: FRACTIONS SKILL SCORE OPERATIONAL-UTILITY THRESHOLD

### 2.1 The Issue
Step 7 claimed that a +120-minute circular FSS of $0.5560$ on the July 5, 2025 validation set exceeded an "operational-utility threshold" of approximately $0.52$.

### 2.2 Forensic Tracing and Literature Origin
- **Source Formula:** In the spatial verification literature, Roberts and Lean (2008) (*Monthly Weather Review*, 136(1), 78–97) derived a theoretical target skill criterion:
  $$\text{FSS}_{\text{useful}} = 0.5 + \frac{f_0}{2}$$
  where $f_0$ is the domain-wide base rate (observed event fractional coverage). For convective events with low fractional coverage ($f_0 \approx 0.02\text{--}0.04$), this yields $0.5 + 0.02 = 0.52$.
- **Project Context:** Roberts and Lean derived this criterion for high-resolution numerical weather prediction (NWP) model verification against radar composites, where a random forecast with the same base rate achieves $\text{FSS}_{\text{random}} = f_0$, and $\text{FSS}_{\text{useful}}$ represents halfway between a random forecast and a perfect forecast ($1.0$).
- **Discrepancy:** This threshold was never pre-registered in the approved project synopsis or the experimental design protocol as an operational requirement. Furthermore, as noted by Mittermaier (2014) and Skok (2015), the applicability of $0.5 + f_0/2$ as an "operational threshold" is highly sensitive to domain size, grid resolution, and spatial sparsity. Applying it post hoc to claim that the model has crossed into "certified operational utility" is scientifically untenable.

### 2.3 Correction Made
1. The claim of exceeding an "operational-utility threshold" has been **completely removed** from all thesis text and claim maps.
2. The measured FSS values are reported strictly as comparative model performance metrics:
   - Baseline Exp A achieved $0.3232 \pm 0.1158$ at +120m ($30\text{ km}$, $\ge 5.0\text{ mm/hr}$).
   - Compound Loss Exp D achieved $0.5560 \pm 0.0307$, representing an absolute gain of $+0.2328$ ($+72.0\%$).
3. The Roberts & Lean (2008) value of $\approx 0.52$ is cited solely as a *theoretical reference benchmark* from the atmospheric literature to provide context, with an explicit disclaimer that no operational standard was pre-specified.

---

## 3. AUDIT ITEM B: BULK QPF RMSE PARITY AND STATISTICAL EQUIVALENCE

### 3.1 The Issue
Step 7 reported that at +180 minutes on the test benchmark, Exp D achieved an RMSE of $1.5216\text{ mm/hr}$ vs. Exp A's $1.5224\text{ mm/hr}$ with bootstrap $p = 0.449$, and concluded that this "proves zero bulk regression degradation" or "proves exact parity".

### 3.2 Underlying Data Verification
Inspecting `outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv` and `outputs/experiments/model_research/step5_three_seed_replication/TEST_DIAGNOSTICS_PER_SEED.csv`:
- **Exp A RMSE (+180m, Test Benchmark, $N=129$):**
  - Seed 42: $1.5262\text{ mm/hr}$
  - Seed 123: $1.5322\text{ mm/hr}$
  - Seed 999: $1.5088\text{ mm/hr}$
  - Sample Mean: **$1.5224\text{ mm/hr}$** | Sample Std: **$0.0122\text{ mm/hr}$**
- **Exp D RMSE (+180m, Test Benchmark, $N=129$):**
  - Seed 42: $1.5026\text{ mm/hr}$
  - Seed 123: $1.5329\text{ mm/hr}$
  - Seed 999: $1.5292\text{ mm/hr}$
  - Sample Mean: **$1.5216\text{ mm/hr}$** | Sample Std: **$0.0165\text{ mm/hr}$**
- **Paired Bootstrap Difference ($\Delta = \text{ExpD} - \text{ExpA}$):**
  - Point Estimate: $\bar{\Delta} = -0.0008\text{ mm/hr}$
  - 95% Bootstrap Confidence Interval ($L=12$ sequences = 6.0 hours, $B=1,000$ resamples): **$[-0.0616, +0.0483]\text{ mm/hr}$**
  - Continuity-Corrected $p$-value: **$p = 0.4166$** (raw contrary replicates: 416 of 1,000)

### 3.3 Statistical Integrity Assessment
In classical null-hypothesis significance testing, a non-significant result ($p = 0.417 \ge 0.05$) indicates only that the null hypothesis (that there is no difference in means) cannot be rejected. It is an elementary logical fallacy to assert that failing to reject the null "proves equivalence" or "proves no increase in error." Establishing statistical equivalence requires defining an *a priori* equivalence margin $\pm \delta_{\text{equiv}}$ and demonstrating that the two-sided 90% or 95% confidence interval falls entirely within that margin (e.g., via Two One-Sided Tests, TOST). Because no equivalence margin was established, claiming "proven equivalence" is invalid.

### 3.4 Correction Made
1. Replaced "proves zero bulk regression degradation" and "proves exact parity" with:
   *"No statistically significant difference in bulk QPF RMSE was detected between Exp D and baseline Exp A ($1.5216\text{ mm/hr}$ vs $1.5224\text{ mm/hr}$, $p = 0.417$, 95% CI: $[-0.0616, +0.0483]\text{ mm/hr}$ under $L=12$ block bootstrap), indicating that intermediate-horizon convective gains were achieved without detectable degradation in domain-wide bulk error at the evaluated sample size."*
2. Added an explicit note distinguishing lack of detectable difference from formal proof of equivalence.

---

## 4. AUDIT ITEM C: STORM-CORE CENTROID DISPLACEMENT AND MULTI-SCALE FSS

### 4.1 Implementation and Distance Formulation
The displacement diagnostic in `scripts/run_step5_three_seed_replication.py` (lines 388–398) was inspected:
```python
# Identification of Domain-Wide Maximum Pixels
o_y, o_x = np.unravel_index(np.argmax(o_frame), o_frame.shape)
f_y, f_x = np.unravel_index(np.argmax(f_frame), f_frame.shape)

# Euclidean Distance in Physical Kilometers
dist_km = np.sqrt(((o_y - f_y) * 11.08) ** 2 + ((o_x - f_x) * 10.03) ** 2)
```
- **Physical Grid Spacing:** The latitudinal spacing is $\Delta y = 0.10^\circ \times 110.8\text{ km/deg} = 11.08\text{ km}$. The longitudinal spacing at $\approx 25.5^\circ\text{N}$ is $\Delta x = 0.10^\circ \times 111.32\text{ km/deg} \times \cos(25.5^\circ) = 10.03\text{ km}$. This physical Euclidean projection is geometrically correct for the Meghalaya grid.
- **Argmax vs. Storm Objects:** The function uses `np.argmax(o_frame)` and `np.argmax(f_frame)`. It does **not** identify, segment, or track multi-cell storm objects (such as in the Method for Object-Based Diagnostic Evaluation, MODE, or Contiguous Rain Area, CRA). It evaluates strictly the single highest pixel in the entire $25 \times 40$ grid.
- **Handling of Ties, Dry Frames, and Unmatched Events:**
  - Dry frames ($\max(o\_frame) < 5.0\text{ mm/hr}$) are excluded from displacement evaluation.
  - "Matched Frame" definition: A frame is eligible if $\max(o\_frame) \ge 5.0\text{ mm/hr}$, and is matched if and only if $\max(f\_frame) \ge 5.0\text{ mm/hr}$.
  - If the forecast fails to reach $5.0\text{ mm/hr}$ anywhere in the domain, the frame is recorded as an *unmatched storm miss* (displacement cannot be computed).
  - Ties: In the event of identical maximum values, `np.argmax` returns the first occurring index in row-major order.
  - Missing observations: Masked out via $M(x, y)$.

### 4.2 Comprehensive Event Breakdown: Eligible, Matched, and Missed Frames

An exhaustive recomputation across all frames on Validation ($N=33$) and Test ($N=129$) confirms the following breakdown:

| Split Cohort | Forecast Lead | Total Frames | Eligible Storms ($\ge 5\text{ mm/hr}$) | Exp A Matched | Exp D Matched | Exp A Missed (%) | Exp D Missed (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | +30m | 33 | 26 | 24 | 24 | 2 (7.7%) | 2 (7.7%) |
| ($N=33$) | +60m | 33 | 27 | 24 | 24 | 3 (11.1%) | 3 (11.1%) |
| | +120m | 33 | 29 | 8 | **23** | 21 (72.4%) | **6 (20.7%)** |
| | **+180m** | 33 | 31 | **1** | **22** | **30 (96.8%)** | **9 (29.0%)** |
| | +360m | 33 | 33 | **0** | **10** | **33 (100.0%)** | **23 (69.7%)** |
| **Test Benchmark**| +30m | 129 | 117 | 107 | 112 | 10 (8.5%) | 5 (4.3%) |
| ($N=129$) | +60m | 129 | 117 | 94 | 102 | 23 (19.7%) | 15 (12.8%) |
| | +120m | 129 | 117 | 50 | **78** | 67 (57.3%) | **39 (33.3%)** |
| | **+180m** | 129 | 117 | **6** | **62** | **111 (94.9%)** | **55 (47.0%)** |
| | +360m | 129 | 116 | **0** | **35** | **116 (100.0%)** | **81 (69.8%)** |

### 4.3 Displacement Distribution Statistics

Across the matched events (evaluated on Seed 42 checkpoints):
- **+30m:**
  - Test ($N=112$): Mean = $73.0\text{ km}$, **Median = $14.9\text{ km}$**, 25th percentile = $0.0\text{ km}$, 75th percentile = $134.4\text{ km}$, Min = $0.0\text{ km}$, Max = $338.5\text{ km}$.
  - Val ($N=24$): Mean = $84.8\text{ km}$, Median = $47.2\text{ km}$, Min = $0.0\text{ km}$, Max = $275.3\text{ km}$.
- **+60m:**
  - Test ($N=102$): Mean = $83.1\text{ km}$, **Median = $27.2\text{ km}$**, 25th percentile = $10.0\text{ km}$, 75th percentile = $153.9\text{ km}$, Min = $0.0\text{ km}$, Max = $351.8\text{ km}$.
  - Val ($N=24$): Mean = $80.0\text{ km}$, Median = $43.0\text{ km}$, Min = $11.1\text{ km}$, Max = $275.3\text{ km}$.
- **+120m:**
  - Test ($N=78$): Mean = $123.0\text{ km}$, Median = $83.1\text{ km}$, 25th percentile = $26.9\text{ km}$, 75th percentile = $188.7\text{ km}$, Min = $0.0\text{ km}$, Max = $342.7\text{ km}$.
  - Val ($N=23$): Mean = $99.2\text{ km}$, Median = $68.7\text{ km}$, Min = $20.1\text{ km}$, Max = $269.1\text{ km}$.
- **+180m:**
  - Test ($N=62$): Mean = **$156.6\text{ km}$**, Median = **$157.1\text{ km}$**, 25th percentile = $52.5\text{ km}$, 75th percentile = $238.7\text{ km}$, Min = $22.9\text{ km}$, Max = $323.7\text{ km}$.
  - Val ($N=22$): Mean = **$109.9\text{ km}$**, Median = **$73.3\text{ km}$**, 25th percentile = $49.9\text{ km}$, 75th percentile = $161.0\text{ km}$, Min = $32.1\text{ km}$, Max = $245.4\text{ km}$.

### 4.4 Verification of the Multi-Peak Ambiguity Hypothesis
We executed an explicit diagnostic on all validation frames at +180m where the displacement offset exceeded $100\text{ km}$:
- **Finding:** In **100% of the large-displacement cases (9 out of 9)**, multiple distinct local convective peaks ($\ge 5.0\text{ mm/hr}$) were active simultaneously across the domain.
  - The observation fields contained between **4 and 22 distinct convective peaks**.
  - The forecast fields contained between **1 and 9 distinct convective peaks**.
- **Case Evidence:** In sequence $t=29$, the observation had 22 convective peaks, with the global peak ($21.2\text{ mm/hr}$) in the western Garo Hills at $(14, 11)$. The model predicted 5 convective peaks, with its global peak ($10.8\text{ mm/hr}$) in the eastern Khasi Hills at $(11, 35)$. The distance between these two coordinates is $243.0\text{ km}$. However, the model also actively predicted rain in the western cluster. The large offset is an artifact of taking the domain-wide argmax across multiple clusters.
- **Conclusion:** The multi-cell switching phenomenon is **empirically confirmed by the source data**. However, this diagnostic remains a distributional peak offset, not an object tracker.

### 4.5 Multi-Scale Fractions Skill Score (10, 20, 30, and 50 km)
FSS was verified across all four scales directly from `outputs/experiments/model_research/step5_three_seed_replication/SPATIAL_VERIFICATION_RESULTS.csv` (threshold $\ge 5.0\text{ mm/hr}$, mean across seeds):
- **+120m (Test Benchmark, $N=129$):**
  - $10\text{ km}$ ($k=1$): Exp A = $0.3092$, Exp D = $0.4755$ ($\Delta = +0.1663$)
  - $20\text{ km}$ ($k=2$): Exp A = $0.3633$, Exp D = $0.5424$ ($\Delta = +0.1791$)
  - $30\text{ km}$ ($k=3$): Exp A = $0.4113$, Exp D = $0.6000$ ($\Delta = +0.1887$)
  - $50\text{ km}$ ($k=5$): Exp A = $0.4531$, Exp D = $0.6645$ ($\Delta = +0.2114$)
- **+180m (Test Benchmark, $N=129$):**
  - $10\text{ km}$ ($k=1$): Exp A = $0.0370$, Exp D = $0.3266$ ($\Delta = +0.2896$)
  - $20\text{ km}$ ($k=2$): Exp A = $0.0451$, Exp D = $0.3831$ ($\Delta = +0.3380$)
  - $30\text{ km}$ ($k=3$): Exp A = $0.0527$, Exp D = $0.4333$ ($\Delta = +0.3806$)
  - $50\text{ km}$ ($k=5$): Exp A = $0.0592$, Exp D = $0.4928$ ($\Delta = +0.4336$)

### 4.6 Complementary Nature of FSS and Displacement
FSS and argmax displacement measure fundamentally different, non-conflicting aspects of predictive quality:
- **FSS** evaluates spatial frequency alignment within a neighborhood tolerance ($30\text{ km}$). It is tolerant of slight advective offsets and captures whether convective activity is present in the proper geographic region.
- **Argmax displacement** evaluates the precision of locating the single most intense convective cell in the entire domain.
- The two metrics do not invalidate each other: FSS demonstrates that Exp D produces actionable convective coverage within regional tolerances ($FSS \approx 0.43$ at $30\text{ km}$), while the displacement diagnostic demonstrates that pinpointing the single heaviest storm core becomes increasingly uncertain over 3 hours.

---

## 5. AUDIT ITEM D: THREE-SEED REPLICATION AND STATISTICAL UNCERTAINTY

### 5.1 Distinction Between Across-Seed and Temporal Resampling
- **Across-Seed Variation ($N=3$ Initializations):** Models were trained with seeds $42, 123,$ and $999$. Across-seed standard deviation measures sensitivity to stochastic gradient descent trajectory and weight initialization:
  - For +180m Pointwise CSI on validation, paired gains were: Seed 42 ($+0.0835$), Seed 123 ($+0.0936$), Seed 999 ($+0.1013$). Sample standard deviation of the paired gain is $s_\Delta = 0.0089$.
  - With $N=3$, degrees of freedom is $df = 2$. This establishes empirical consistency across the sampled seeds, but cannot support formal asymptotic inferential hypothesis testing across the distribution of initializations.
- **Temporal Block Bootstrapping ($B=1,000$ Resamples):**
  - Evaluates temporal episode sampling variability conditional on the evaluated checkpoints.
  - Uses $L=12$ sequences ($6.0\text{ hours}$) on the 129-sequence test benchmark, ensuring complete independence between adjacent resampled blocks.
  - The continuity-corrected bound $p < 0.001$ reflects the consistency of model outperformance across synoptic weather episodes, *not* generalization across random initializations.

---

## 6. AUDIT ITEM E: SIX-HOUR FORECAST REALITIES (+360m)

### 6.1 Reconciled Figures from Source Data
- **Baseline Exp A (+360m):**
  - Pointwise CSI = $0.0000 \pm 0.0000$ (Validation & Test)
  - Circular FSS ($30\text{ km}$) = $0.0000 \pm 0.0000$ (Validation & Test)
  - Domain Max Rain Rate: $1.75\text{ mm/hr}$ on Val, $1.47\text{ mm/hr}$ on Test.
  - *Extinction Verdict:* Complete convective collapse. The model predicts a completely flat, washed-out field below the $5.0\text{ mm/hr}$ threshold.
- **Combined Loss Exp D (+360m):**
  - Pointwise CSI: $0.0037 \pm 0.0037$ (Val), $0.0125 \pm 0.0129$ (Test).
  - Circular FSS ($30\text{ km}$): $0.0193 \pm 0.0172$ (Val), $0.0526 \pm 0.0495$ (Test).
  - Domain Max Rain Rate: $7.47\text{ mm/hr}$ on Val, $7.10\text{ mm/hr}$ on Test.
  - Bootstrap significance: $p < 0.001$ over zero extinction.

### 6.2 Scientific Boundary Framing
While Exp D statistically prevents complete convective collapse, an FSS of **$0.0526$** represents **very weak absolute skill**. The thesis explicitly states that at 6 hours, purely satellite-driven spatiotemporal deep learning reaches a physical predictability boundary in the absence of numerical weather prediction (NWP) coupling or atmospheric thermodynamic soundings. The 6-hour convective forecasting problem remains unsolved.

---

## 7. AUDIT ITEM F: MAXIMUM PREDICTED RAIN RATES

### 7.1 Source Reconciliation Across Steps 4, 5, and 6
- **Step 5 Multi-Seed Figures (Authoritative):**
  - Evaluated on July 5, 2025 validation ($N=33$), observed maximum is $48.58\text{ mm/hr}$.
  - At +180m, Exp A mean max across 3 seeds is **$7.06\text{ mm/hr}$** (99th percentile: $3.05\text{ mm/hr}$).
  - At +180m, Exp D mean max across 3 seeds is **$13.15\text{ mm/hr}$** (99th percentile: $6.17\text{ mm/hr}$).
- **Step 4 Pilot Figures (Seed 42 only):**
  - Exp A max was $10.64\text{ mm/hr}$; Exp D max was $17.61\text{ mm/hr}$.
- **Resolution:** As audited in `outputs/experiments/model_research/step6_statistical_integrity_audit/SOURCE_VALUE_RECONCILIATION.csv`, Step 4 used a single global random seed across models, whereas Step 5 instantiated an isolated `torch.Generator(seed)` for each run. Best validation loss epoch for Exp A was epoch 14 in Step 4 vs epoch 6 in Step 5. Step 5 multi-seed figures are authoritative.
- **Methodological Limitation:** Maximum predicted rain rate is a domain-wide distributional tail metric. It proves that the compound loss prevents conditional mean blurring, but is *not* proof that the model predicted the exact location, timing, or shape of an observed storm core.

---

## 8. AUDIT ITEM G: DATA SPLITS AND GENERALIZATION

### 8.1 Verification of Partitions
- **Training Cohort:** 38 active monsoon days across 2023–2025, yielding **1,764 continuous sequences** ($T_{\text{in}}=4, T_{\text{out}}=5$).
  - July 1–10, 2023 (10 days)
  - July 1–14, 2024 (14 days)
  - August 5–14, 2024 (10 days)
  - July 1–4, 2025 (4 days)
- **Validation Cohort:** Single severe convective episode on **July 5, 2025** (33 sequences). Used strictly for epoch checkpoint selection.
- **Confirmatory Historical Benchmark:** Extreme synoptic depression event on **July 12–14, 2023** (129 sequences, 72.0 continuous hours).

### 8.2 Chronological and Re-Evaluation Limitations
1. *Prior Evaluation History:* The July 12–14, 2023 benchmark was evaluated during earlier project milestones (Phase 2 and 3). While it was locked and never accessed during SimVP training or loss optimization, it is an out-of-sample historical benchmark rather than an unseen prospective test.
2. *Chronological Non-Inversion:* The training set includes data from 2024 and 2025, while the benchmark is from 2023. This is an out-of-sample back-testing evaluation, not a strictly forward prospective operational trial.

---

## 9. DOWNSTREAM LANDSLIDE COUPLING BOUNDARY

- The Phase 4 Random Forest landslide susceptibility model and dynamic antecedent hazard tiers were calibrated on historical landslide inventories (GSI / NASA COOLR) with canonical precipitation inputs.
- The SimVP research conducted in Steps 4 through 6 focused exclusively on weather nowcasting loss optimization. The improved rainfall fields have **not been coupled** through the downstream landslide model.
- **Rule Enforced:** The thesis strictly prohibits any claim that nowcasting improvements increased landslide prediction accuracy, polygon AUC-ROC, or warning lead time.

---

## 10. CONCLUSION AND AUDIT SIGN-OFF

All 12 major claims in [`CLAIM_TO_EVIDENCE_MAP.csv`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/CLAIM_TO_EVIDENCE_MAP.csv) have been audited against the physical code and data. All identified discrepancies have been corrected:
1. Operational-utility threshold claim removed;
2. RMSE equivalence claim replaced with non-significant difference description;
3. Displacement single-peak argmax mechanics and multi-peak characteristics fully disclosed;
4. Temporal bootstrap boundaries clarified;
5. Predictability ceilings at 6 hours transparently established;
6. Downstream landslide coupling boundary strictly maintained.

**Audit Status:** **APPROVED WITH ALL QUALIFICATIONS ENFORCED**  
**Lead Auditor:** Senior Research Engineer & Antigravity Scientific Integrity Auditor  
**Date:** October 30, 2026
