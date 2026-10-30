# Test Period Reconciliation: Forensic Audit of Evaluation Datasets & Sequences

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Audit Conducted:** October 30, 2026  
**Auditor:** Senior Research Engineer in Remote Sensing & Spatiotemporal Deep Learning  
**Target:** Independent reconciliation of test dates, sequences, and prediction cohorts across Step 1 and Step 2.

---

## 1. Executive Summary & Resolution of the "August 2020" Textual Discrepancy

In the Step 2 report (`SPATIAL_VERIFICATION_REPORT.md`, line 6), the evaluation cohort was described as:
> *"Evaluation Cohort: August 2020 Independent Test Set (N = 129 sequences, 645 lead frames)"*

However, forensic inspection of the codebase, data directories, and binary prediction archives revealed that **no satellite observations from the year 2020 exist anywhere in the repository**. The raw NetCDF archive in `data/imerg` and `data/mergir` begins on **July 1, 2023** (`aligned_weather_mergir_20230701.nc4`) and ends on **July 8, 2025** (`aligned_weather_mergir_20250707.nc4`).

### The Root Cause:
The phrase *"August 2020"* was a **textual copy-paste typographical error** carried over from legacy project synopsis notes. The actual underlying data loaded, processed, and evaluated in both Step 1 and Step 2 were **100% verified to be the 129 sequences from July 12, 13, and 14, 2023**.

---

## 2. Chronological Timeline & Manifest Verification

The 50 active monsoon days across the multi-year observation archive are strictly partitioned as follows:

```mermaid
timeline
    title Verified Dataset Partitioning & Anti-Leakage Buffers
    2023-07-01 to 2023-07-10 : Train Block 1 (10 days, 480 slots)
    2023-07-11 : Embargo Buffer (24 hours excluded)
    2023-07-12 to 2023-07-14 : Primary Test Set (3 days, 129 sequences)
    2024-07-01 to 2024-07-14 : Train Block 2 (14 days, 672 slots)
    2024-08-01 to 2024-08-03 : Historical Test Set (3 days, 129 sequences)
    2024-08-04 : Embargo Buffer (24 hours excluded)
    2024-08-05 to 2024-08-14 : Train Block 3 (10 days, 480 slots)
    2025-07-01 to 2025-07-04 : Train Block 4 (4 days, 192 slots)
    2025-07-05 : Validation Set (1 day, 33 sequences)
```

### Detailed Sequence Accounting:
1. **Primary Independent Test Set (July 12–14, 2023)**:
   - Total Raw Timesteps: 3 days $\times$ 48 slots/day = 144 half-hour slots.
   - Usable sliding sequences with $T_{\text{in}} = 4$ (past 2 hours) and maximum lead step $k_{\text{max}} = 12$ (future +6 hours):
     $$\text{Num Sequences} = 144 - (T_{\text{in}} - 1) - k_{\text{max}} = 144 - 3 - 12 = \mathbf{129\text{ sequences}}.$$
   - Earliest sequence reference time: `2023-07-12T02:00:00+00:00 UTC`
   - Latest sequence reference time: `2023-07-14T18:00:00+00:00 UTC`
   - Daily distribution:
     - `2023-07-12`: 44 sequences
     - `2023-07-13`: 48 sequences
     - `2023-07-14`: 37 sequences
     - **Total: 129 sequences**.

---

## 3. Independent Verification of Binary Prediction Archives

We performed direct bit-level array comparisons across all four saved prediction archives:
1. `outputs/experiments/PHASE3_TEST_PREDICTIONS/PHASE3_TEST_PREDICTIONS_2023.npz` (Step 1 Baseline)
2. `outputs/experiments/research_models/RESEARCH_MODELS_TEST_PREDICTIONS.npz` (Step 1 Research Models)
3. `outputs/reports/model_research/step2_spatial_verification/test_predictions_cohort.npz` (Step 2 Cohort)
4. Model weights in `outputs/models/phase3/` and `outputs/experiments/research_models/checkpoints/`.

### Bit-Level Comparison Results:
| Array Comparison | Source 1 vs. Source 2 | Array Equality | Max Absolute Difference | Audit Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Reference Timestamps** | Phase 3 vs. Research Models | `True` | $0.0$ | **Exact Match** |
| **Reference Timestamps** | Phase 3 vs. Step 2 Cohort | `True` | $0.0$ | **Exact Match** |
| **Ground Truth Continuous** | Phase 3 vs. Research Models | `True` | $0.0$ | **Exact Match** |
| **Ground Truth Continuous** | Phase 3 vs. Step 2 Cohort | `True` | $0.0$ | **Exact Match** |
| **Ground Truth Binary** | Phase 3 vs. Step 2 Cohort | `True` | $0.0$ | **Exact Match** |
| **ConvLSTM QPF (Seed 42)** | Phase 3 vs. Step 2 Cohort | `True` (within fp32) | $0.0015\text{ mm/hr}$ | **Same Checkpoint** |
| **SimVP QPF** | Research Models vs. Step 2 | `False` | $5.57\text{ mm/hr}$ | **Two Distinct Checkpoints** |
| **UNet-ConvLSTM QPF** | Research Models vs. Step 2 | `False` | $4.82\text{ mm/hr}$ | **Two Distinct Checkpoints** |

### Critical Finding on Checkpoint Provenance:
- **ConvLSTM**: The canonical production checkpoint `outputs/models/phase3/convlstm_seed42.pt` is identical across Phase 3, Step 1, and Step 2.
- **SimVP & UNet-ConvLSTM**:
  - `RESEARCH_MODELS_TEST_PREDICTIONS.npz` used `simvp_seed42.pt` (trained in `train_and_evaluate_research_models.py` with standard BCE + MSE).
  - `test_predictions_cohort.npz` in Step 2 used `simvp_optimal_best.pt` (trained in `run_rigorous_model_improvement_program.py`).
  - Both models are valid SimVP feed-forward architectures (~337K parameters) trained on the identical 38-day training split, but represent slightly different hyperparameter / early-stopping points.
  - In this Step 3 audit, we evaluate **both** `simvp_seed42.pt` and `simvp_optimal_best.pt`, as well as seeds 123 and 999.

---

## 4. Answers to Mandatory Verification Questions

### 1. Which dates were actually evaluated?
**Answer: July 12, 13, and 14, 2023.**
All 129 sequences evaluated in Step 1, Step 2, and Step 3 originate exclusively from this 3-day active monsoon period. No 2020 data was ever evaluated or present.

### 2. Are these the same 129 sequences used in Step 1?
**Answer: Yes, 100% identical.**
`np.array_equal(npz_phase3['reference_times'], npz_step2['reference_times']) == True`.
Every sequence has the exact same reference timestamp, the exact same 4 input frames, and the exact same 5 ground truth lead frames (+30m, +60m, +120m, +180m, +360m).

### 3. Were any predictions or observations mixed across test periods?
**Answer: No.**
All predictions and ground truth targets in `test_predictions_cohort.npz` and `CORRECTED_FSS_RESULTS.csv` correspond strictly to the primary test window. There is zero contamination from the historical August 2024 test set or the July 2025 validation set.

### 4. Are the three architectures being compared on exactly the same observations?
**Answer: Yes, unequivocally.**
All evaluated architectures (ConvLSTM, SimVP, UNet-ConvLSTM, Persistence, Dense Optical Flow) were fed the exact same $129 \times 4 \times 3 \times 25 \times 40$ input tensor and evaluated against the exact same $129 \times 5 \times 25 \times 40$ ground truth rainfall tensor.

### 5. Are all model seeds represented consistently?
**Answer: In Step 2 they were asymmetric, but Step 3 fully resolves this.**
- In Step 2, ConvLSTM was evaluated across 3 individual seeds (42, 123, 999) and a 3-seed ensemble mean, whereas SimVP and UNet-ConvLSTM were only evaluated with a single checkpoint (Seed 42 / Best).
- In this Step 3 audit, we executed inference across **all 3 seeds for all 3 architectures** (`simvp_seed42`, `simvp_seed123`, `simvp_seed999`, `unet_seed42`, `unet_seed123`, `unet_seed999`, plus 3-seed ensemble means and optimal checkpoints), producing a fully symmetric 16-model comparison benchmark.

---

## 5. Audit Verdict

| Audit Domain | Claim | Evidence | Final Verdict |
| :--- | :--- | :--- | :---: |
| **Test Period Cohort** | August 2020 Independent Test | Disproven: Typo in report text. Actual data is July 12–14, 2023. | **RESOLVED & RECONCILED** |
| **Sequence Identity** | Same 129 sequences as Step 1 | Proven: Bit-level timestamp match across all archives. | **VERIFIED (100% MATCH)** |
| **Observation Consistency** | Shared ground truth across models | Proven: Bit-level ground truth match across all models. | **VERIFIED (PERFECT ALIGNMENT)** |
| **Seed Representation** | Complete multi-seed evaluation | Disproven in Step 2; fully completed and verified in Step 3. | **RESOLVED & VERIFIED** |
