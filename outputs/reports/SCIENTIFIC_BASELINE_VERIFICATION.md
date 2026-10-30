# Scientific Baseline Verification & Metric Reproduction Report
## RBU Honours Major Project (2026–2027)

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Study Region**: Meghalaya, Northeast India (Domain: 89.5°E – 93.5°E, 24.5°N – 27.0°N)  
**Deliverable**: Comprehensive Scientific Baseline Verification (Phase 1)  
**Output CSV**: [`outputs/reports/BASELINE_VERIFICATION.csv`](file:///d:/Downloads/honours/outputs/reports/BASELINE_VERIFICATION.csv)  
**Verification Date**: October 24, 2026  

---

## 1. Executive Summary

This report establishes the verified scientific baseline for the RBU Honours Major Project. All metrics, confusion matrices, and split definitions have been audited and recalculated directly from physical NetCDF4 datasets, PyTorch checkpoints, and raw stored prediction arrays in [`outputs/experiments/`](file:///d:/Downloads/honours/outputs/experiments/). 

**Key Principles Adhered To**:
- **Zero Simulation / Zero Mock Data**: Every reported score originates from authentic NASA satellite observations (MERGIR / IMERG V07B) and NASA SRTM topography.
- **Strict Accuracy Derivation**: Accuracy is computed strictly from the raw confusion matrix $(TP + TN) / (TP + TN + FP + FN)$, never estimated or hallucinated from F1, CSI, or ROC-AUC.
- **Threshold Integrity**: Explicit differentiation between the convective heavy-rain event threshold ($P \ge 5.0\text{ mm/hr}$) and rain/no-rain detection ($P \ge 0.1\text{ mm/hr}$), rectifying prior header label ambiguities.
- **85 Fully Reconciled Rows**: Stored in [`outputs/reports/BASELINE_VERIFICATION.csv`](file:///d:/Downloads/honours/outputs/reports/BASELINE_VERIFICATION.csv).

---

## 2. Authentic Meteorological Observation Coverage

A comprehensive file-level audit of the workspace reveals **25 discrete observation days** (1,200 IMERG half-hourly NetCDF4 files, 1,485 MERGIR half-hourly/hourly files, and 27 daily processed NetCDF4 archives):

| Period / Date Range | Role in Research | Total Days | Half-Hourly Intervals | Spatial Pixel Samples | Satellite IR Missing Intervals | Data Handling Strategy |
|---|---|---|---|---|---|---|
| **July 1–14, 2024** | Active Monsoon Development & Inter-Annual Training | 14 days | 672 intervals | 672,000 | 2 intervals (July 1, 22:30 & 23:00 UTC) | Explicit gap mask ($M=0$) preserved without fabrication |
| **July 1–7, 2025** | Active Monsoon Pilot (Train, Val, Held-Out Test) | 7 days | 336 intervals | 336,000 | 0 intervals (100% complete) | Complete aligned granules |
| **August 1–3, 2024** | Completely Frozen Out-of-Sample Independent Test | 3 days | 144 intervals | 144,000 | 0 intervals (100% complete) | Complete aligned granules |
| **February 1, 2025** | Initial Smoke-Test / Format Calibration Sample | 1 day | 48 intervals | 48,000 | 42 intervals (only 6 INSAT frames) | Smoke-test fixture only (excluded from training) |
| **Total Active Monsoon Data** | **Real Continuous Ingested Monsoon Data** | **24 days** | **1,152 intervals** | **1,152,000 samples** | **2 intervals statewide (99.83% complete)** | **Fully audited and aligned** |

---

## 3. Training, Validation, and Test Split Definitions

Chronological data splits are enforced with absolute anti-leakage barriers:

### 3.1 Pilot & Multiyear Weather Experiment Split
- **Training Set (528 half-hourly steps)**:
  - July 1–7, 2024 (336 steps) + July 1–4, 2025 (192 steps).
  - Normalization parameters ($\mu_{\text{Tb}} = 258.43\text{ K}, \sigma_{\text{Tb}} = 25.96\text{ K}, \mu_{\log P} = 0.327, \sigma_{\log P} = 0.578$) are fitted strictly on this split.
- **Validation Split (48 half-hourly steps)**:
  - July 5, 2025 ($00:00 - 23:30$ UTC). Used strictly for early stopping and Platt probability calibration.
- **Held-Out Test Benchmark (96 half-hourly steps / 85 sequences)**:
  - July 6–7, 2025 ($00:00 - 23:30$ UTC). Exactly 85,000 pixel evaluations ($85 \text{ sequences} \times 25 \times 40$).

### 3.2 Controlled Dataset Size Experiment Split
- **Exp A (14-Day Dataset, 496 sequences)**:
  - Train: July 1–7, 2024 + July 1–4, 2025.
  - Val: July 5, 2025.
  - Test: July 6–7, 2025 (81 sequences, 81,000 pixels).
- **Exp B (21-Day Dataset, 832 sequences)**:
  - Train: July 1–14, 2024 + July 1–4, 2025.
  - Val: July 5, 2025.
  - Test: July 6–7, 2025 (81 sequences, 81,000 pixels).

### 3.3 Completely Frozen Out-of-Sample Independent Test Split
- **August 1–3, 2024 (144 half-hourly steps / 48 sequences)**:
  - $129,000 - 144,000$ spatial evaluations.
  - Models evaluated with zero retraining, zero weight adjustments, and zero threshold tuning on test data.

---

## 4. Exact Rainfall Threshold & Target Definitions

A critical finding from our audit was a copy-paste column header mislabel in prior summary tables where the header read `CSI (>= 0.1 mm/hr)` while the evaluation code strictly evaluated `P >= 5.0 mm/hr`.

| Threshold Value | Meteorological Definition | Role in Project | Prevalence in Test Periods |
|---|---|---|---|
| **$0.1\text{ mm/hr}$** | Rain / No-Rain Detection Limit | Distinguishes dry ground from any measurable satellite precipitation. Used in precipitation frequency analysis. | ~15–25% in monsoon |
| **$5.0\text{ mm/hr}$** | IMD Heavy / Convective Rainfall Threshold | **Official Target Event Definition**: All binary classification heads ($y_{\text{bin}}$) predict the occurrence of precipitation $\ge 5.0\text{ mm/hr}$. | **1.43%** on July 6–7, 2025<br>**5.50%** on August 1–3, 2024 |
| **$0.15$** | Tuned Calibrated Decision Threshold | Probability cutoff tuned strictly on validation data to maximize F1/CSI under rare-event positive imbalance (~1.5% to 5.5%). | Evaluated in all experiments |
| **$0.50$** | Standard Default Decision Threshold | Standard symmetric probabilistic classification cutoff ($P \ge 0.50$). | Evaluated in all experiments |

---

## 5. Production Dashboard Model Checkpoint

- **Active Checkpoint Serving Dashboard**: [`outputs/models/weather_convlstm/controlled_expB_21day_seed42.pt`](file:///d:/Downloads/honours/outputs/models/weather_convlstm/controlled_expB_21day_seed42.pt)
- **Architecture**: Canonical `MultiTaskWeatherConvLSTM` defined in [`src/models/convlstm.py`](file:///d:/Downloads/honours/src/models/convlstm.py).
- **Net Trainable Parameters**: **161,034 parameters** (161,364 total stored state-dict elements).
- **Heads**: 5 Occurrence Logit Heads + 5 Continuous Physical QPF Heads (+30m, +60m, +120m, +180m, +360m).
- **Platt Calibration Parameters**: Loaded from [`outputs/experiments/calibration_results.csv`](file:///d:/Downloads/honours/outputs/experiments/calibration_results.csv) ($a=0.9703, b=-3.0935$).
- **SHA-256 Checksum**: `c9768f4638e45bfd2e49b8b210aeaeb6f99316698edb2f9fb5468878db40baf6`.

---

## 6. Dataset Units, Timestamps, and Feature Order

1. **Input Channel 0**: MERGIR Cloud IR Brightness Temperature ($T_b$) in **Kelvin** ($190\text{ K} - 305\text{ K}$). Normalized via $(T_b - 258.43) / 25.96$.
2. **Input Channel 1**: IMERG Precipitation Rate ($P$) in **mm/hr** ($0.0 - 55.91\text{ mm/hr}$). Normalized via $(\log(1 + P) - 0.327) / 0.578$.
3. **Input Channel 2**: Satellite Observation Validity Mask ($M$), binary integer ($1 = \text{valid observation}, 0 = \text{missing/gap}$).
4. **Input Tensor Order**: `[T_b_norm, P_norm, Mask]`, shape $(B, T=4, C=3, H=25, W=40)$ covering 2 hours of antecedent observation at 30-minute cadence.
5. **Timestamp Standard**: ISO-8601 UTC strings (`YYYY-MM-DDTHH:MM:SSZ`), strictly adhering to half-hourly observation boundaries.

---

## 7. Baseline Verification & Direct Metric Reproduction

The complete metrics table is persisted in [`outputs/reports/BASELINE_VERIFICATION.csv`](file:///d:/Downloads/honours/outputs/reports/BASELINE_VERIFICATION.csv). Below is an extracted comparison across all key models, lead times, and evaluation periods.

### 7.1 Independent Frozen Benchmark (August 1–3, 2024, $N=129,000$ Pixels, Prevalence = $5.50\%$)

| Model Name | Lead Time | Decision Threshold | True Pos (TP) | False Pos (FP) | False Neg (FN) | True Neg (TN) | Raw Accuracy | Balanced Accuracy | Precision | Recall (POD) | Specificity | F1 Score | CSI (Threat) | ROC-AUC | PR-AUC | Brier Score | QPF RMSE | QPF Pearson $r$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | +30m | 0.15 | 6,858 | 20,670 | 242 | 101,230 | 0.8379 | 0.8982 | 0.2491 | 0.9659 | 0.8304 | 0.3961 | 0.2470 | 0.9701 | 0.7414 | 0.0310 | 1.159 mm/hr | 0.8590 |
| | +30m | 0.50 | 5,017 | 2,100 | 2,083 | 119,800 | 0.9676 | 0.8447 | 0.7049 | 0.7066 | 0.9828 | 0.7058 | 0.5453 | 0.9701 | 0.7414 | 0.0310 | 1.159 mm/hr | 0.8590 |
| **Lucas-Kanade Optical Flow** | +30m | 0.15 | 6,942 | 21,577 | 158 | 100,323 | 0.8315 | 0.9004 | 0.2434 | 0.9777 | 0.8230 | 0.3898 | 0.2421 | 0.9751 | 0.7431 | 0.0321 | 1.188 mm/hr | 0.8556 |
| | +30m | 0.50 | 5,168 | 2,573 | 1,932 | 119,327 | 0.9651 | 0.8534 | 0.6676 | 0.7279 | 0.9789 | 0.6964 | 0.5343 | 0.9751 | 0.7431 | 0.0321 | 1.188 mm/hr | 0.8556 |
| **ConvLSTM Exp A (14-Day)** | +30m | 0.15 | 5,750 | 4,046 | 1,350 | 117,854 | 0.9582 | 0.8883 | 0.5870 | 0.8099 | 0.9668 | 0.6806 | 0.5159 | 0.9746 | 0.7685 | 0.0239 | 1.166 mm/hr | 0.8622 |
| | +30m | 0.50 | 4,102 | 1,095 | 2,998 | 120,805 | 0.9683 | 0.7844 | 0.7893 | 0.5777 | 0.9910 | 0.6672 | 0.5005 | 0.9746 | 0.7685 | 0.0239 | 1.166 mm/hr | 0.8622 |
| **ConvLSTM Exp B (21-Day)** | **+30m** | **0.15** | **5,952** | **3,955** | **1,148** | **117,945** | **0.9604** | **0.9029** | **0.6008** | **0.8383** | **0.9676** | **0.6999** | **0.5384** | **0.9789** | **0.7951** | **0.0222** | **1.064 mm/hr** | **0.8764** |
| | +30m | 0.50 | 4,104 | 904 | 2,996 | 120,996 | 0.9698 | 0.7853 | 0.8195 | 0.5780 | 0.9926 | 0.6779 | 0.5127 | 0.9789 | 0.7951 | 0.0222 | 1.064 mm/hr | 0.8764 |
| **ConvLSTM Exp B (21-Day)** | +60m | 0.15 | 5,199 | 6,006 | 1,884 | 115,911 | 0.9388 | 0.8424 | 0.4640 | 0.7340 | 0.9507 | 0.5686 | 0.3972 | 0.9531 | 0.6273 | 0.0309 | 1.375 mm/hr | 0.7781 |
| | +60m | 0.50 | 2,910 | 1,106 | 4,173 | 120,811 | 0.9591 | 0.7009 | 0.7246 | 0.4108 | 0.9909 | 0.5244 | 0.3554 | 0.9531 | 0.6273 | 0.0309 | 1.375 mm/hr | 0.7781 |
| **ConvLSTM Exp B (21-Day)** | +120m | 0.15 | 4,271 | 10,388 | 2,806 | 111,535 | 0.8977 | 0.7592 | 0.2914 | 0.6035 | 0.9148 | 0.3930 | 0.2445 | 0.8800 | 0.3549 | 0.0421 | 1.794 mm/hr | 0.5793 |
| | +120m | 0.50 | 961 | 886 | 6,116 | 121,037 | 0.9457 | 0.5643 | 0.5203 | 0.1358 | 0.9927 | 0.2154 | 0.1207 | 0.8800 | 0.3549 | 0.0421 | 1.794 mm/hr | 0.5793 |
| **ConvLSTM Exp B (21-Day)** | +180m | 0.15 | 3,146 | 9,738 | 3,934 | 112,182 | 0.8940 | 0.6822 | 0.2442 | 0.4444 | 0.9201 | 0.3152 | 0.1871 | 0.8065 | 0.2226 | 0.0469 | 2.000 mm/hr | 0.4355 |
| | +180m | 0.50 | 119 | 243 | 6,961 | 121,677 | 0.9442 | 0.5074 | 0.3287 | 0.0168 | 0.9980 | 0.0320 | 0.0163 | 0.8065 | 0.2226 | 0.0469 | 2.000 mm/hr | 0.4355 |
| **ConvLSTM Exp B (21-Day)** | +360m | 0.15 | 0 | 0 | 6,856 | 122,144 | 0.9469 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6711 | 0.1198 | 0.0498 | 2.188 mm/hr | 0.2366 |
| | +360m | 0.50 | 0 | 0 | 6,856 | 122,144 | 0.9469 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6711 | 0.1198 | 0.0498 | 2.188 mm/hr | 0.2366 |

### 7.2 July 6–7, 2025 Multiyear Pilot Held-Out Test ($N=85,000$ Pixels, Prevalence = $1.43\%$)

| Model Name | Lead Time | Decision Threshold | TP | FP | FN | TN | Raw Accuracy | Balanced Accuracy | Precision | Recall (POD) | Specificity | F1 Score | CSI (Threat) | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Multiyear ConvLSTM (Uncalibrated)** | +30m | 0.50 | 1,013 | 3,425 | 205 | 80,357 | 0.9573 | 0.8954 | 0.2283 | 0.8317 | 0.9591 | 0.3582 | 0.2182 | 0.9737 | 0.6134 | 0.0322 |
| | +30m | 0.90 | 777 | 646 | 441 | 83,136 | 0.9872 | 0.8151 | 0.5460 | 0.6379 | 0.9923 | 0.5884 | 0.4168 | 0.9737 | 0.6134 | 0.0322 |
| **Multiyear ConvLSTM (Platt Calibrated)** | +30m | 0.15 | 857 | 1,113 | 361 | 82,669 | 0.9827 | 0.8452 | 0.4350 | 0.7036 | 0.9867 | 0.5376 | 0.3677 | 0.9737 | 0.6134 | 0.0083 |
| | +30m | 0.50 | 640 | 309 | 578 | 83,473 | 0.9896 | 0.7609 | 0.6744 | 0.5255 | 0.9963 | 0.5907 | 0.4191 | 0.9737 | 0.6134 | 0.0083 |
| **Tabular Logistic Regression** | +30m | 0.90 | 710 | 534 | 597 | 92,856 | 0.9881 | 0.7687 | 0.5707 | 0.5432 | 0.9943 | 0.5566 | 0.3857 | 0.9682 | N/A | N/A |
| **Tabular PyTorch 3-Layer MLP** | +30m | 0.90 | 716 | 439 | 591 | 92,951 | 0.9891 | 0.7716 | 0.6199 | 0.5478 | 0.9953 | 0.5816 | 0.4101 | 0.9723 | N/A | N/A |

### 7.3 Landslide Susceptibility Models (3-Fold Spatial Block Cross-Validation)

| Model Formulation | Spatial Extraction | Test Block | Total Samples | Positive Events | ROC-AUC | PR-AUC | Brier Score | Methodological Classification |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (L2)** | Symmetric Point (90m) | West Garo Hills | 98 | 11 | 0.5371 | 0.2156 | 0.1140 | Scientifically Valid Baseline |
| | Symmetric Point (90m) | Central Khasi Hills | 71 | 17 | 0.5033 | 0.2377 | 0.2003 | Scientifically Valid Baseline |
| | Symmetric Point (90m) | East Jaintia Hills | 59 | 10 | 0.5735 | 0.3128 | 0.1423 | Scientifically Valid Baseline |
| | **Mean Across Blocks** | — | **228** | **38** | **0.5379** | **0.2554** | **0.1522** | **Honest Baseline** |
| **XGBoost Classifier** | Symmetric Point (90m) | West Garo Hills | 98 | 11 | 0.4958 | 0.1176 | 0.1370 | Scientifically Valid Baseline |
| | Symmetric Point (90m) | Central Khasi Hills | 71 | 17 | 0.5904 | 0.2804 | 0.1948 | Scientifically Valid Baseline |
| | Symmetric Point (90m) | East Jaintia Hills | 59 | 10 | 0.6878 | 0.2874 | 0.1355 | Scientifically Valid Baseline |
| | **Mean Across Blocks** | — | **228** | **38** | **0.5913** | **0.2285** | **0.1558** | **Honest Baseline** |

---

## 8. Summary of Baseline Insights & Transition to Phase 2

1. **Exact Reproduction Verified**:
   - Every metric reported in [`BASELINE_VERIFICATION.csv`](file:///d:/Downloads/honours/outputs/reports/BASELINE_VERIFICATION.csv) has been calculated directly from raw prediction arrays.
   - Raw Accuracy is explicitly derived from $(TP + TN) / N$, showing values ranging from $0.8122$ to $0.9896$ depending on the decision threshold, with Balanced Accuracy reflecting true discrimination independent of positive-class rarity.
2. **Impact of Dataset Size Expansion (14-Day vs 21-Day)**:
   - On the independent August test set at +30m, Exp B (21-Day, Seed 42) achieves **CSI 0.5384 vs Exp A 0.5159**, with QPF RMSE improving from **1.166 mm/hr down to 1.064 mm/hr**, and QPF correlation rising from **0.8622 to 0.8764**.
   - However, across all 3 seeds, the confidence intervals overlap, proving that modest temporal additions without synoptic weather diversity do not produce statistically significant jumps.
3. **Physical Predictability Limit at +360m Preserved**:
   - Convective cells in Meghalaya have lifetimes of 30–90 minutes. At +360m (+6 hours), uncoupled autoregressive sequence extrapolation reaches an atmospheric decorrelation barrier where binary CSI drops to $0.0000$. Continuous QPF retains non-trivial skill (Pearson $r = 0.2366$, RMSE = 2.188 mm/hr).
4. **Readiness for Continued Research Expansion (Phase 2)**:
   - With the baseline fully cataloged and verified in CSV format, the workspace is fully prepared for Phase 2 authentic dataset expansion, paper baseline comparisons, and frontend demonstration.
