# Phase 5 Scientific Corrections & Methodological Audit Report
## Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction
**Institution**: Royal Global University, B.Tech Honours (2026–2027)  
**Study Domain**: Meghalaya, Northeast India  
**Date**: October 2026  
**Status**: **OFFICIAL AUDIT & METHODOLOGICAL CORRECTIONS**

---

## 1. Executive Summary

This document establishes the scientific corrections and methodological disclosures required for academic submission of the Honours Major Project. During Phase 4, extensive empirical evaluations were conducted on landslide susceptibility and dynamic rainfall triggering. While the numerical calculations and code execution were accurate, rigorous scientific auditing revealed several critical methodological nuances, sample-selection effects, and descriptive inaccuracies that must be transparently documented and corrected in the final dissertation.

---

## 2. Correction 1: Asymmetric Terrain Feature Extraction & Extreme-Value Bias

### 2.1 The Phenomenon
In Phase 4, two sharply contrasting sets of performance metrics were recorded for landslide susceptibility modeling under 3-fold leave-one-block-out spatial cross-validation:
- **Nominal Single-Point Extraction (0m)**: $\text{Spatial ROC-AUC} \approx 0.45 - 0.51$ (indistinguishable from random guessing).
- **Buffered Neighborhood Extraction (450m / 5x5 window)**: $\text{Spatial ROC-AUC} \approx 0.96 - 0.99$ (near-perfect separation).

### 2.2 Root Cause Analysis
Inspection of `scripts/run_phase4_landslide_experiments.py` (`extract_features` function, lines 504–532) reveals an asymmetric feature extraction protocol:
```python
if label == 1 and use_buffer_for_positives and buffer_half_size > 0:
    # Positives: Local extreme over 5x5 window (25 pixels at 90m resolution)
    s_mean = float(np.mean(terrain["slope_mean"][r_s:r_e, c_s:c_e]))
    s_max  = float(np.max(terrain["slope_max"][r_s:r_e, c_s:c_e]))
    rel    = float(np.max(terrain["relief"][r_s:r_e, c_s:c_e]))
    s_std  = float(np.max(terrain["slope_std"][r_s:r_e, c_s:c_e]))
    tri    = float(np.max(terrain["tri"][r_s:r_e, c_s:c_e]))
    curv   = float(np.min(terrain["curvature_mean"][r_s:r_e, c_s:c_e]))
    tpi    = float(np.min(terrain["tpi"][r_s:r_e, c_s:c_e]))
else:
    # Background pseudo-absences: Single central pixel point value
    s_mean = float(terrain["slope_mean"][r_idx, c_idx])
    s_max  = float(terrain["slope_max"][r_idx, c_idx])
    rel    = float(terrain["relief"][r_idx, c_idx])
    s_std  = float(terrain["slope_std"][r_idx, c_idx])
    tri    = float(terrain["tri"][r_idx, c_idx])
    curv   = float(terrain["curvature_mean"][r_idx, c_idx])
    tpi    = float(terrain["tpi"][r_idx, c_idx])
```

### 2.3 Mathematical & Methodological Implication
1. **Extreme-Value Sampling Shift**: For any continuous random field $X(\mathbf{s})$, the maximum over $N=25$ spatial neighbors $\max_{i=1..25} X_i$ is statistically shifted toward the right tail compared to a single realization $X$. When positive instances receive the neighborhood extreme while negative background instances receive a single-pixel draw, a machine learning classifier readily exploits this distribution shift.
2. **Geomorphic Motivation vs Inadvertent Separation**: The physical motivation for neighborhood buffering is sound: nominal geocodes in media reports frequently pinpoint the roadway or village in the valley floor, whereas the actual failure scarp is on the adjacent cliff face. However, applying the extreme operator *exclusively* to positives while evaluating negatives at single points creates an asymmetric attribution bias.
3. **Symmetric Evaluation Performance**: When symmetric spatial feature extraction is applied (e.g. statewide 90m raster inference or symmetric point-to-point cross-validation), authentic out-of-fold spatial discrimination is:
   $$\text{Mean Spatial ROC-AUC} \approx 0.5913$$
   - West Garo Hills: $0.4958$
   - Central Khasi Hills: $0.5904$
   - East Jaintia Hills: $0.6878$

### 2.4 Mandatory Scientific Conclusion
- All observed scores ($\text{ROC-AUC} = 0.506$ nominal, $0.962$ buffered, $0.591$ symmetric spatial) are preserved in the experiment archives.
- **The buffered $\text{ROC-AUC} \ge 0.96$ must NOT be reported as "validated real-world landslide forecast accuracy."**
- It must be explicitly disclosed as a **neighborhood extreme attribution experiment** that illustrates geocoding uncertainty rather than operational predictive skill.

---

## 3. Correction 2: Reconciling Dynamic Hazard Class Proportions

### 3.1 The Inconsistency
In the Phase 4 summary narrative, the dynamic hazard section stated:
> *"Dynamic model correctly identifies elevated hazard (0.8% cells reaching Level 2/3 during peak convective bursts on July 7, 2025, 14:00 UTC)."*

However, the empirical record in [`outputs/experiments/PHASE4_DYNAMIC_HAZARD_VALIDATION.csv`](file:///d:/Downloads/honours/outputs/experiments/PHASE4_DYNAMIC_HAZARD_VALIDATION.csv) (row 8, Step 316) lists:
- `low_hazard_cells_pct`: **99.2%** (Level 0: Low)
- `moderate_hazard_cells_pct`: **0.8%** (Level 1: Moderate)
- `high_hazard_cells_pct`: **0.0%** (Level 2: High)
- `very_high_hazard_cells_pct`: **0.0%** (Level 3: Very High)

### 3.2 Correction
- **0.8% of statewide cells reached Moderate (Level 1) hazard.**
- **0.0% of cells reached High (Level 2) or Very High (Level 3) hazard.**
- The text claiming Level 2/3 activation was a typographic misclassification of Moderate cells. In the final dissertation and frontend, this is corrected to:
  > *"During the peak monsoon storm on July 7, 2025 (14:00 UTC), 0.8% of the Meghalaya domain transitioned from Low to Moderate (Level 1) hazard, while High and Very High tiers were not triggered due to the conservative regional antecedent rainfall threshold."*

---

## 4. Correction 3: Reformulating Next-24-Hour Prediction Feasibility

### 4.1 Prior Overstatement
Prior drafts stated that supervised next-24-hour landslide prediction is "mathematically impossible."

### 4.2 Scientifically Rigorous Reformulation
Supervised next-24-hour prediction is not mathematically impossible in principle; rather, it is **empirically blocked and scientifically invalid given the constraints of authentic available data**:
1. **Temporal Epoch Disconnection**: Verified historical landslide events in NASA GLC span **2007 to 2019**. Authentic high-resolution satellite precipitation cubes (IMERG V07 / MERGIR) in our workspace span **2023, 2024, and 2025**. There is zero temporal overlap between the predictor cubes and event records.
2. **Timestamp Coarseness**: 100% of NASA GLC events have day-level nominal timestamps (`00:00:00 UTC`), precluding alignment with sub-hourly convective triggers.
3. **Extreme Spatiotemporal Sparsity**: Evaluating a 24-hour model over 13 years across 1,000 spatial cells yields:
   $$\text{Prevalence} = \frac{38}{1,000 \times 365 \times 13} \approx 8.01 \times 10^{-6}$$
   Standard supervised recurrent networks (GRU/LSTM) collapse trivially to majority-class zero ($0.000$ recall).
4. **Presence-Only Bias**: In tropical rainforests, unrecorded cells cannot be treated as confirmed non-landslide negatives.

**Conclusion**: Retaining the two-tier architecture (static geomorphic LSI + empirical antecedent rainfall thresholding) is the scientifically defensible path.

---

## 5. Correction 4: Characterization of the Dynamic Rainfall Hazard Layer

### 5.1 Prior Overstatement
Prior documentation referred to the dynamic layer as a "calibrated 24-hour failure probability" or a "NASA LHASA 2.0 implementation."

### 5.2 Corrected Characterization
- The layer is a **Custom Antecedent Rainfall Index (ARI) Dynamic Hazard Decision Matrix**, conceptually inspired by Kirschbaum & Stanley (2018) and Stanley et al. (2021).
- It couples static LSI with a 3-day antecedent rainfall index:
  $$A_t = 0.6 \times P_{24h}(t) + 0.4 \times P_{72h}(t)$$
- It outputs **discrete categorical alert tiers (0: Low, 1: Moderate, 2: High, 3: Very High)**.
- It is **NOT** a calibrated probability of slope failure and **NOT** an operational disaster early warning system.

---

## 6. Correction 5: Boundary Polygon Provenance & Event Integrity

### 6.1 Boundary Polygon
- File: `data/raw/boundary/meghalaya_boundary.geojson`
- Verified Metadata:
  - Source: **William & Mary geoBoundaries Open Administrative Boundary Database (gbOpen)** (*Runfola et al., 2020*)
  - Layer: ADM1 (State level)
  - `shapeName`: "Meghālaya"
  - `shapeISO`: "IN-ML"
  - `shapeID`: "1811400B28404458933503"
  - `shapeGroup`: "IND"
- **Correction**: Do not label this layer as "Survey of India" unless a direct Survey of India digital transfer certificate is present. It is officially cited as **geoBoundaries gbOpen ADM1 (IN-ML)**.

### 6.2 Landslide Catalog Provenance
- Source: NASA Global Landslide Catalog (GLC/COOLR, Kirschbaum et al., 2015).
- 67 raw reports $\to$ 42 within geoBoundaries polygon $\to$ 38 authentic rainfall-triggered events.
- 25 cross-border events excluded (Assam valley floodplains and Bangladesh lowlands).
- 4 non-rainfall triggers excluded (quarry collapse, open-cast coal mining excavation).

---

## 7. Correction 6: Retrospective Research Framing vs Live Claims

### 7.1 Operational Claim Removal
- The platform operates on historical NASA GPM IMERG and MERGIR archives (pilot focus: July 1–7, 2025; independent test: July 12–14, 2023; August 1–3, 2024).
- The platform does not ingest live streaming satellite telemetry in real time.
- All references in UI, banners, and documentation must clearly read:
  > **"RETROSPECTIVE RESEARCH NOWCASTS — ARCHIVE PILOT (JULY 2025)"**
- All live operational claims are strictly excised.

---

## 8. Summary Audit Matrix

| Issue | Prior Formulation | Audited Finding | Submission-Ready Scientific Correction |
|---|---|---|---|
| **Landslide Spatial ROC-AUC** | Claimed 0.962–0.990 operational forecast skill | 5x5 extreme-value operator applied to positives vs point values for negatives | Disclose as neighborhood geocoding sensitivity experiment; authentic symmetric spatial CV is $\approx 0.591$ |
| **Dynamic Hazard Proportions** | Prose claimed 0.8% High / Very High | CSV logs show 99.2% Low, 0.8% Moderate, 0.0% High/Very High | Correct prose: 0.8% Moderate, 0.0% High / Very High |
| **Next-24h Prediction** | "Mathematically impossible" | Data limitation: temporal disconnection, day-level timestamps, extreme sparsity | Frame as empirical data blockage, not mathematical impossibility |
| **Dynamic Rainfall Hazard** | "Calibrated failure probability / NASA LHASA 2" | Categorical heuristic matrix ($0.6 P_{24h} + 0.4 P_{72h}$) | Label as empirical ARI decision-matrix warning index (LHASA-inspired) |
| **Meghalaya Boundary** | "Survey of India administrative boundary" | GeoJSON metadata reveals geoBoundaries gbOpen ADM1 | Cite as William & Mary geoBoundaries (Runfola et al., 2020) |
| **System Scope** | Ambiguous live vs archive presentation | Retrospective research pipeline on historical NASA datasets | Enforce "RETROSPECTIVE RESEARCH ARCHIVE" labeling throughout |

```
========================================================================================
PHASE 5 SCIENTIFIC CORRECTIONS APPROVED FOR DISSERTATION INCLUSION
========================================================================================
```
