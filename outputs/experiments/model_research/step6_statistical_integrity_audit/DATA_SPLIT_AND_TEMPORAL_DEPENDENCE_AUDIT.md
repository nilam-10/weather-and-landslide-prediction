# Step 6 Audit: Data Splits, Manifest Reconciliation, and Temporal Autocorrelation

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Audit Target:** Dataset Manifests, Split Boundary Integrity, Autocorrelation, and Cohort Classification  
**Audit Date:** October 30, 2026  
**Artifact Directory:** `outputs/experiments/model_research/step6_statistical_integrity_audit/`

---

## 1. Dataset Manifest Reconciliation

Forensic inspection of `scripts/run_step5_three_seed_replication.py`, `scripts/run_rigorous_model_improvement_program.py`, and underlying HDF5/NetCDF4 data archives in `data/processed/weather/` establishes the exact data manifests:

### 1.1 Complete Partition Specifications

| Split Category | Date Manifest List | Total Days | Sequence Count ($N$) | Temporal Stride | Input History ($T_{\text{in}}$) | Forecast Leads ($T_{\text{lead}}$) | First Sequence Timestamp | Last Sequence Timestamp |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | • July 1–10, 2023 (10 d)<br>• July 1–14, 2024 (14 d)<br>• Aug 5–14, 2024 (10 d)<br>• July 1–4, 2025 (4 d) | **38 days** | **1,764** | 30 min | 4 frames (2.0 h) | 5 horizons (+30m to +360m) | `2023-07-01T02:00:00Z` | `2025-07-04T18:00:00Z` |
| **Validation Split** | • July 5, 2025 (1 d) | **1 day** | **33** | 30 min | 4 frames (2.0 h) | 5 horizons (+30m to +360m) | `2025-07-05T02:00:00Z` | `2025-07-05T18:00:00Z` |
| **Confirmatory Test Split** | • July 12–14, 2023 (3 d) | **3 days** | **129** | 30 min | 4 frames (2.0 h) | 5 horizons (+30m to +360m) | `2023-07-12T02:00:00Z` | `2023-07-14T18:00:00Z` |

---

## 2. Reconciling the "64.5-Hour" Test Window Claim

In previous experiment reports (e.g., Step 4 and Step 5), the test cohort was described as *"64.5 hours of continuous monsoon nowcasting."* This audit cross-examined the actual timestamps to determine whether this figure is exact or approximate.

### Mathematical Breakdown of Test Set Temporal Geometry:
1. **Sequence Count:** $N = 129$ sequences.
2. **Stride Interval:** $\Delta t = 30\text{ minutes} = 0.5\text{ hours}$.
3. **Reference Timestamps:**
   - First sequence reference time: `2023-07-12T02:00:00+00:00`
   - Last sequence reference time: `2023-07-14T18:00:00+00:00`
   - Elapsed reference time duration:
     $$\Delta T_{\text{ref}} = (129 - 1) \times 0.5\text{ hours} = 64.0\text{ hours}$$
4. **Origin of the "64.5-Hour" Figure:**
   - Multiplying $129\text{ sequences} \times 0.5\text{ hours/sequence} = \mathbf{64.5\text{ hours}}$.
   - This represents the total sum of 30-minute interval bins associated with the 129 forecasts, rather than the elapsed duration between the first and last forecast initialization.
5. **Total Observation Footprint:**
   - First input frame: `2023-07-12T00:30:00+00:00` ($t - 90\text{ min}$)
   - Last forecast target at +360m: `2023-07-14T24:00:00+00:00` ($t + 360\text{ min}$)
   - Total contiguous observation window spanned: **exactly 72.0 hours (3 full calendar days)**.

> **Audit Recommendation for Thesis:** State precisely: *"The confirmatory test dataset spans 3 consecutive active monsoon days (July 12–14, 2023, 72.0 total observation hours), yielding 129 causal forecast sequences initialized every 30 minutes over a 64.0-hour initialization span."*

---

## 3. Split Boundary and Information Leakage Audit

### 3.1 Did Input or Target Windows Cross Split Boundaries?
**Finding: NO. Zero leakage occurred across split boundaries.**

**Evidence:**
1. In `scripts/run_step5_three_seed_replication.py` (lines 71–73):
   ```python
   tr_x, ... = build_advanced_sequences(train_dates, t_in=4)
   val_x, ... = build_advanced_sequences(val_dates, t_in=4)
   te_x, ... = build_advanced_sequences(test_dates, t_in=4)
   ```
   Each cohort was built via an isolated invocation of `build_advanced_sequences` with disjoint date manifests.
2. In `scripts/run_rigorous_model_improvement_program.py` (lines 184–207), strict temporal continuity validation is enforced on every sequence:
   ```python
   # Verify past window continuity
   if (dt_ref - dt_past).total_seconds() != (t_in - 1) * 1800:
     continue
   # Verify future target continuity
   for k in LEAD_STEPS:
     if (dt_future - dt_ref).total_seconds() != k * 1800:
       all_valid = False
       break
```
   Any missing timestamp or date boundary gap automatically discards the candidate window.
3. The training dates ended on July 4, 2025; validation began on July 5, 2025 at 02:00 UTC (input window starting at 00:30 UTC). No input frame of the validation set overlapped with the training set.

---

### 3.2 Did Overlapping Sequences Cause Dependence in Evaluation?
**Finding: YES. Sequences within each evaluation cohort are temporally autocorrelated.**

**Detailed Analysis:**
- Consecutive sequences are sampled with a 30-minute stride ($\Delta t = 1$ frame).
- Consecutive sequences share 3 input frames ($t-60m, t-30m, t$).
- For intermediate and long horizons (+180m and +360m), the verification windows overlap across consecutive sequences:
  - Sequence $i$ predicts +180m at time $t_i + 3\text{h}$.
  - Sequence $i+1$ predicts +180m at time $t_i + 3.5\text{h}$.
  - For +360m (6 hours ahead), twelve consecutive sequences share overlapping target verification periods!
- **Consequence:** Treating each sequence as an independent statistical sample drastically underestimates standard errors and produces artificially narrow confidence intervals.
- **Audit Mandate:** This temporal dependence makes **block bootstrapping** mandatory. Furthermore, for the +360m horizon, blocks of at least 12 sequences (6 hours) are required to eliminate overlap bias.

---

### 3.3 Normalization Parameters Audit
**Finding: VERIFIED. Fitted exclusively on historical training data.**

- Input channels are normalized using fixed parameters established in Phase 3:
  - Infrared Brightness Temperature ($T_b$): $\mu = 268.442\text{ K}, \sigma = 20.619\text{ K}$.
  - GPM IMERG Precipitation ($P$): $\mu = 0.709\text{ mm/hr}, \sigma = 2.110\text{ mm/hr}$.
- These statistics were derived from the 38-day historical training split and held constant across all scripts.
- Precipitation target values ($y_{\text{cont}}$) were **never normalized or scaled**; they were evaluated in raw, physical units ($\text{mm/hr}$).

---

## 4. Rigorous Classification of Evaluation Cohorts

To adhere to rigorous academic honesty standards in the B.Tech thesis, the experimental data cohorts must be clearly differentiated. Researchers must **never** describe previously evaluated test sets as "unseen", "blind", or "pristine".

### Formal Cohort Classification

| Cohort Name | Calendar Period | Sequence Count ($N$) | Historical Usage in Project | Formal Methodological Status | Thesis-Safe Description |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Training Split** | 38 active monsoon days (2023, 2024, 2025) | 1,764 | Used for gradient updates in Phase 3, Step 4, Step 5. | **Training Cohort** | *"Training dataset used for parameter optimization."* |
| **Validation Split** | July 5, 2025 | 33 | Used for checkpoint selection (minimum validation loss epoch). | **Validation / Model Selection Cohort** | *"Validation dataset used exclusively for checkpoint selection and loss convergence monitoring."* |
| **Confirmatory Test Split** | July 12–14, 2023 | 129 | Evaluated in Phase 3 baseline audit, Step 2 spatial verification, Step 3 audit, Step 4 pilot, and Step 5 replication. | **Previously Inspected Benchmark Cohort** | *"Locked confirmatory test benchmark previously used for architecture comparisons; never accessed during loss computation or checkpoint selection."* |
| **Untouched Independent Split** | August 1–3, 2024 | 129 | Evaluated in Step 3 `evaluate_frozen_models_on_independent_test.py` for frozen baselines, but **never evaluated on SimVP loss experiments**. | **Genuinely Untouched Out-of-Sample Cohort** | *"Independent historical monsoon test window reserved for zero-shot generalization testing."* |

---

## 5. Thesis Integrity Guidelines

1. **Do not claim the July 12–14, 2023 test cohort was "blind" or "unseen":**
   Acknowledge that this 3-day monsoon event has served as the repository's primary benchmark test episode across earlier phases.
2. **Emphasize Strictly Causal Freezing:**
   Highlight that all six SimVP models in Step 5 were selected **strictly** by the minimum validation loss on July 5, 2025, and that the test set was locked and never used for model selection, stopping criteria, or hyperparameter tuning.
3. **Disclose Single-Day Limitation for Validation:**
   Acknowledge that the validation split is a single active monsoon day ($N=33$), which provides sufficient resolution for checkpoint selection but limited statistical power for independent inference.
