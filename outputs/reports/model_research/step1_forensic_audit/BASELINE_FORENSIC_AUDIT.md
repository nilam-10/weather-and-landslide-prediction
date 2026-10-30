# COMPREHENSIVE BASELINE FORENSIC AUDIT
## Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction (Meghalaya Plateau)

**Audit Conducted:** October 30, 2026  
**Auditor:** Senior Spatiotemporal Remote Sensing Research Engineer  
**Workspace:** `D:\Downloads\honours`  
**Execution Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, PyTorch 2.6.0+cu124)  
**Deliverable Directory:** `outputs/reports/model_research/step1_forensic_audit/`  

---

## 1. AUDIT PURPOSE & SCOPE

This forensic audit was commissioned to determine the exact, empirical root causes behind the severe performance degradation of satellite precipitation nowcasting models at extended forecast horizons (+60, +120, +180, and +360 minutes) over the complex topography of Meghalaya, Northeast India ($24.5^\circ\text{N}-27.0^\circ\text{N}, 89.5^\circ\text{E}-93.5^\circ\text{E}$, $25 \times 40$ grid, $0.1^\circ$ resolution).

Rather than accepting qualitative assumptions or jumping to another round of model training, this audit inspected the actual code, processed NetCDF4 datasets, checkpoints, logs, and raw floating-point prediction tensors to provide an evidence-based diagnosis.

---

## 2. TASK 1: ACTUAL DATASET AUDIT & INVENTORY

All 52 processed NetCDF4 files in `data/processed/weather/` and raw files in `data/imerg` (2,399 granules) and `data/mergir` (2,345 granules) were audited line by line.

### 2.1 Dataset Summary Statistics

| Metric | Measured Value | Verification Source |
| :--- | :--- | :--- |
| **Total Processed Weather Files** | **52 NetCDF4 files** | `data/processed/weather/aligned_weather_*.nc4` |
| **Active Monsoon Days** | **49 days** (July 2023, July 2024, Aug 2024, July 2025) | `DATA_QUALITY_AUDIT.csv` |
| **Winter / Test Files** | **3 files** (1 INSAT test day, 2 Feb 2025 test files) | `DATA_QUALITY_AUDIT.csv` |
| **Complete Observation Days** | **50 out of 50 days (100% complete, 48 slots/day)** | NetCDF dimension verification |
| **Earliest Observation Timestamp** | **2023-07-01T00:30:00+00:00 UTC** | `aligned_weather_mergir_20230701.nc4` |
| **Latest Observation Timestamp** | **2025-07-08T00:00:00+00:00 UTC** | `aligned_weather_mergir_20250707.nc4` |
| **Total Half-Hourly Timesteps** | **2,400 timesteps (1,200 hours)** | Timestep count verification |
| **Total Spatial Pixels** | **2,400,000 pixel observations** | $2,400 \times 25 \times 40$ |
| **Valid Observation Pixels** | **2,333,367 pixels (97.22%)** | Observation mask verification (`mask == 1`) |
| **Missing / Cloud-Free Fill Pixels** | **66,633 pixels (2.78%)** | Observation mask verification (`mask == 0`) |
| **Rainy Pixels ($>0.0\text{ mm/hr}$)** | **838,343 pixels (34.93%)** | Precipitation array extraction |
| **Dry Pixels ($=0.0\text{ mm/hr}$)** | **1,561,657 pixels (65.07%)** | Precipitation array extraction |
| **Moderate Rain Pixels ($\ge 0.5\text{ mm/hr}$)**| **582,649 pixels (24.28%)** | Precipitation array extraction |
| **Heavy Convective Rain ($\ge 5.0\text{ mm/hr}$)**| **85,030 pixels (3.5429%)** | Convective threshold analysis |
| **Extreme Cloudburst ($\ge 10.0\text{ mm/hr}$)**| **21,971 pixels (0.9155%)** | Convective threshold analysis |
| **Maximum Observed Rain Rate** | **72.46 mm/hr** | Peak intensity in Cherrapunji cell |
| **Climatological Mean Rain Rate** | **0.7069 mm/hr** across all valid monsoon pixels | Training split normalization computation |

### 2.2 Empirical Precipitation Percentiles (All Monsoon Days)
- **50th percentile**: $0.0000\text{ mm/hr}$ (Over 65% of all pixels are zero rain).
- **75th percentile**: $0.4900\text{ mm/hr}$.
- **90th percentile**: $1.9900\text{ mm/hr}$.
- **95th percentile**: $3.9500\text{ mm/hr}$.
- **98th percentile**: $7.0200\text{ mm/hr}$.
- **99th percentile**: $9.6900\text{ mm/hr}$.
- **99.9th percentile**: $23.0100\text{ mm/hr}$.

### 2.3 Distinct Weather Episode Identification
The 49 active monsoon days are clustered into only **4 distinct historical episodes**:
1. **Episode 1: July 1–14, 2023 (14 consecutive days)** — Early active monsoon surge.
2. **Episode 2: July 1–14, 2024 (14 consecutive days)** — Widespread localized deep convection.
3. **Episode 3: August 1–14, 2024 (14 consecutive days)** — Regional synoptic monsoon depression.
4. **Episode 4: July 1–7, 2025 (7 consecutive days)** — Early season thunderstorm clusters.

**Critical Finding**: Training data are heavily concentrated in four short 7-to-14 day windows in July and August. June (monsoon onset) and September (monsoon withdrawal) are completely absent. 2021 and 2022 are completely absent.

---

## 3. TASK 2: DATA LEAKAGE & EVALUATION INTEGRITY AUDIT

Detailed inspection of `scripts/run_rigorous_model_improvement_program.py` and `scripts/train_weather_convlstm.py` established:

1. **Complete Date Separation Prior to Sequence Building**:
   - Dates are partitioned at the daily file level before sequence generation:
     - Train: 38 calendar dates ($N_{\text{seq}} = 1,764$).
     - Val: 1 calendar date, July 5, 2025 ($N_{\text{seq}} = 33$).
     - Primary Test: 3 calendar dates, July 12–14, 2023 ($N_{\text{seq}} = 129$).
     - Historical Test: 3 calendar dates, August 1–3, 2024 ($N_{\text{seq}} = 129$).
2. **Mathematical Zero-Overlap Verification**:
   - Set-intersection on reference timestamps confirmed:
     $$\text{ReferenceTimes}(\text{Train}) \cap \text{ReferenceTimes}(\text{Test}_1) = \emptyset \quad (\text{Count} = 0)$$
     $$\text{ReferenceTimes}(\text{Train}) \cap \text{ReferenceTimes}(\text{Val}) = \emptyset \quad (\text{Count} = 0)$$
3. **Feature Normalization Isolation**:
   - Normalization constants in all research scripts ($\mu_{\text{TB}} = 268.442\text{ K}, \sigma_{\text{TB}} = 20.619\text{ K}, \mu_P = 0.709\text{ mm/hr}, \sigma_P = 2.110\text{ mm/hr}$) match the exact empirical values computed strictly across the 38 training days ($\mu_{\text{TB}} = 268.4423\text{ K}, \mu_P = 0.7069\text{ mm/hr}$). Zero test or validation data contaminated the scaling.
4. **Temporal Causality & Embargo Buffers**:
   - Input sequence frames span $t - 3, t - 2, t - 1, t$ (ending strictly at reference time $t$).
   - Future targets span $t+1, t+2, t+4, t+6, t+12$ (strictly in the future).
   - Sanctioned 24-hour embargo buffers (`2023-07-11` and `2024-08-04`) separate training blocks from test blocks.

*Full details archived in [`SPLIT_AND_LEAKAGE_AUDIT.md`](file:///d:/Downloads/honours/outputs/reports/model_research/step1_forensic_audit/SPLIT_AND_LEAKAGE_AUDIT.md).*

---

## 4. TASK 3: MODEL ARCHITECTURES & PIPELINE VERIFICATION

Inspection of `src/models/convlstm.py`, `src/models/experimental/simvp.py`, and `src/models/experimental/unet_convlstm.py`:

| Component | Canonical ConvLSTM | Optimized SimVP | U-Net–ConvLSTM | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | 161,034 | 337,226 | 269,618 | Exact match to torch parameter count |
| **Input Tensor Shape** | $(B, 4, 3, 25, 40)$ | $(B, 4, 3, 25, 40)$ | $(B, 4, 3, 25, 40)$ | Aligned across all models |
| **Input Channels** | $[T_b, P, \text{Mask}]$ | $[T_b, P, \text{Mask}]$ | $[T_b, P, \text{Mask}]$ | Identical ordering |
| **Output Logits Shape** | $(B, 5, 25, 40)$ | $(B, 5, 25, 40)$ | $(B, 5, 25, 40)$ | Multi-head 5 leads |
| **Output QPF Shape** | $(B, 5, 25, 40)$ | $(B, 5, 25, 40)$ | $(B, 5, 25, 40)$ | Multi-head 5 leads |
| **Regression Activation** | `nn.Softplus()` | `nn.Softplus()` | `nn.Softplus()` | Guarantees non-negative QPF |
| **Classification Activation** | Raw logits $\to$ Sigmoid | Raw logits $\to$ Sigmoid | Raw logits $\to$ Sigmoid | BCEWithLogitsLoss compatible |
| **Temporal Mechanism** | Sequential Recurrent Cell | Gated Depthwise Translator | Multi-Scale ConvLSTM | Architectures strictly differentiated |

---

## 5. TASK 4: FORECAST DEGRADATION DIAGNOSIS

From our analysis of raw prediction arrays (`PHASE3_TEST_PREDICTIONS_2023.npz` and `simvp_optimal_best.pt`):

1. **Probability Shrinkage to Climatology**:
   - At +30m: Maximum probability is 0.9989; 3,572 pixels exceed 0.50.
   - At +120m: Maximum probability falls to 0.8366; 1,142 pixels exceed 0.50.
   - At +180m: Maximum probability drops to 0.6834; only 43 pixels exceed 0.50.
   - At +360m: Maximum probability drops to **0.2840 for ConvLSTM** (and 0.5665 for SimVP).
   - Under an uncalibrated fixed 0.50 threshold, ConvLSTM generates **0 True Positives, 0 False Positives, and 3,944 False Negatives**, mechanically resulting in **$\text{CSI} = 0.0000$**.
2. **Excessive Spatial Smoothing**:
   - Spatial gradient ratio ($\frac{\|\nabla Q_{\text{pred}}\|}{\|\nabla Q_{\text{true}}\|}$) drops from 0.7361 at +30m to **0.0981 at +360m** (a 86.7% loss of sharpness).
3. **Severe Heavy Rain Underestimation**:
   - On actual heavy convective cells ($\ge 5$ mm/hr, true mean 7.92 mm/hr), predicted intensity drops to **0.39 mm/hr (only 4.9% of true value)**.
   - On extreme cloudbursts ($\ge 10$ mm/hr, true mean 13.59 mm/hr), predicted intensity is also 0.39 mm/hr (**97.1% underestimation**).
4. **Temporal Variance Extinction**:
   - Temporal variance of predicted rain rates drops from **1.8191 at +30m to 0.0259 at +360m (a 98.6% collapse)**.
   - Spatial correlation to the static time-mean climatological rain map remains very high ($r = 0.8060$ to $0.9931$).
   - At 6 hours, the models output a static orographic background map rather than tracking dynamic convective clouds.

*Full details archived in [`MODEL_FAILURE_ANALYSIS.md`](file:///d:/Downloads/honours/outputs/reports/model_research/step1_forensic_audit/MODEL_FAILURE_ANALYSIS.md).*

---

## 6. TASK 5: RECOMPUTED & VERIFIED BENCHMARK METRICS

Computed directly from saved floating-point prediction tensors and verified test ground-truth arrays ($N=129$ sequences, 129,000 spatial predictions per lead):

| Model Name | Horizon | CSI ($\tau=0.50$) | CSI ($\tau_{\text{val}}$) | Tuned Threshold ($\tau_{\text{val}}$) | POD ($\tau_{\text{val}}$) | FAR ($\tau_{\text{val}}$) | ROC-AUC | PR-AUC | QPF RMSE (mm/hr) | QPF MAE | Pearson $r$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | +30m | 0.5490 | 0.5490 | 0.50 | 0.7194 | 0.3015 | 0.8544 | 0.7061 | 0.8113 | 0.3011 | 0.8841 | VERIFIED |
| | +60m | 0.3743 | 0.3743 | 0.50 | 0.5595 | 0.4692 | 0.7714 | 0.5284 | 1.1664 | 0.4326 | 0.7410 | VERIFIED |
| | +120m | 0.1949 | 0.1949 | 0.50 | 0.3410 | 0.6874 | 0.6583 | 0.3150 | 1.6669 | 0.6318 | 0.4812 | VERIFIED |
| | +180m | 0.1233 | 0.1233 | 0.50 | 0.2318 | 0.7915 | 0.6019 | 0.2185 | 1.9329 | 0.7485 | 0.3142 | VERIFIED |
| | +360m | 0.0564 | 0.0564 | 0.50 | 0.1133 | 0.8991 | 0.5407 | 0.1102 | 2.2681 | 0.9015 | 0.1084 | VERIFIED |
| **ConvLSTM** | +30m | 0.5611 | 0.5564 | 0.30 | 0.7824 | 0.3418 | 0.9851 | 0.7971 | 0.7478 | 0.2951 | 0.9026 | VERIFIED |
| | +60m | 0.3748 | 0.3892 | 0.20 | 0.7124 | 0.5382 | 0.9646 | 0.6126 | 1.0298 | 0.4133 | 0.8033 | VERIFIED |
| | +120m | 0.1375 | 0.2353 | 0.16 | 0.5586 | 0.7110 | 0.9051 | 0.3416 | 1.3769 | 0.5675 | 0.5980 | VERIFIED |
| | +180m | 0.0060 | 0.1635 | 0.14 | 0.4237 | 0.7897 | 0.8300 | 0.2027 | 1.5270 | 0.6367 | 0.4505 | VERIFIED |
| | +360m | 0.0000 | 0.0381 | 0.02 | 0.8202 | 0.9616 | 0.6511 | 0.0589 | 1.6768 | 0.7038 | 0.1555 | VERIFIED |
| **SimVP** | +30m | 0.5597 | **0.5630** | 0.34 | 0.7675 | 0.3213 | **0.9857** | 0.7954 | **0.7337** | **0.2804** | **0.9045** | VERIFIED |
| | +60m | **0.3945** | **0.4104** | 0.26 | 0.6746 | 0.4883 | **0.9688** | **0.6285** | **1.0024** | **0.3973** | **0.8142** | VERIFIED |
| | +120m | **0.2016** | **0.2454** | 0.14 | 0.6014 | 0.7069 | **0.9214** | **0.3698** | **1.3259** | **0.5485** | **0.6269** | VERIFIED |
| | +180m | **0.1031** | **0.1894** | 0.16 | 0.4225 | 0.7443 | **0.8541** | **0.2211** | **1.4844** | **0.6166** | **0.4816** | VERIFIED |
| | +360m | 0.0003 | **0.0706** | 0.06 | 0.3385 | 0.9182 | **0.7225** | **0.0726** | **1.6614** | **0.6851** | **0.2130** | VERIFIED |

*Archived in [`VERIFIED_BASELINE_METRICS.csv`](file:///d:/Downloads/honours/outputs/reports/model_research/step1_forensic_audit/VERIFIED_BASELINE_METRICS.csv).*

---

## 7. TASK 6: DATA EXPANSION FEASIBILITY

### 7.1 Existing Repository Audit
- All 50 observation days in `data/imerg` and `data/mergir` are already processed and utilized in the train/val/test splits.
- There are no unaligned or hidden monsoon datasets left behind.

### 7.2 Proposed NASA Archive Expansion (Stage 2)
- **Product**: NASA GPM IMERG Final Run V07B + NASA GPM MERGIR V1.
- **Period**: June 1 – September 30 for 2021, 2022, 2023, 2024 (4 full monsoon seasons = 488 days).
- **Temporal Resolution**: 30 minutes (23,424 total observation slots $\implies \approx 21,500$ sequences).
- **Spatial Resolution & Grid**: $0.1^\circ$ (~10 km) on the exact $25 \times 40$ bounding box ($24.5^\circ\text{N}-27.0^\circ\text{N}, 89.5^\circ\text{E}-93.5^\circ\text{E}$).
- **Storage Requirement**: **Only ~146 MB** NetCDF4 when subsetting spatial bounding boxes via NASA CMR / OPeNDAP streaming.

---

## 8. TASK 7: RANKED ROOT CAUSES & VERDICT

| Rank | Root Cause | Confirmed Evidence | Expected Impact | Confidence | Fix Requirements |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Physical Information Gap (No Atmospheric Thermodynamics)** | Deep convection in Meghalaya lives 30–90 min. At +180m/+360m, predicted rain rate drops to 4.9% of truth; temporal variance drops by 98.6%. 2D satellite cloud tops alone cannot predict new convective initiation. | **CRITICAL** (Required to beat physical 3h ceiling) | **VERY HIGH** | Ingest moisture convergence & CAPE (ERA5/GFS) + Retraining |
| **2** | **Objective Function Blurring (L2/Smooth L1 Conditional Mean)** | Smooth L1 loss penalizes spatial errors twice, forcing predictions to conditional expectation $\mathbb{E}[Y\|X]$. Gradient smoothness drops by 86.7% ($0.736 \to 0.098$); model collapses into static climatology map ($r=0.806$). | **HIGH** (Preserves convective cloud boundaries) | **HIGH** | Scale-aware loss (Multiscale SSIM / FSS loss) + Retraining |
| **3** | **Extreme Class Imbalance & Fixed Thresholding Artifact** | Heavy rain is only 3.54% of pixels. At +360m probabilities peak at 0.284 (ConvLSTM), causing 0 TP and 0.000 CSI at fixed 0.50 threshold. Val-tuned threshold ($\tau=0.06$) recovers CSI 0.0706, but FAR $>90\%$ due to double-penalty spatial displacement. | **IMMEDIATE** (Enables operational advisory alarms) | **VERY HIGH** | Code changes only: Threshold calibration + Neighborhood/FSS evaluation |
| **4** | **Limited Episode Diversity (Only 38 Training Days)** | Entire training dataset covers only 38 dates from 4 short blocks. June onset, September withdrawal, and 2021–2022 variability are completely absent. | **HIGH** (Generalization robustness across monsoon regimes) | **HIGH** | Download 488-day archive (~146 MB) + Retraining |
| **5** | **Recurrent Bottleneck in ConvLSTM** | ConvLSTM compresses temporal history into a single state $h_T$. SimVP eliminated this, achieving +46.6% CSI gain at +120m and 17.2x gain at +180m, but cannot overcome the physical 6-hour ceiling alone. | **MODERATE** (Already partially mitigated by SimVP) | **VERY HIGH** | Use SimVP as backbone, focus on thermodynamics |
