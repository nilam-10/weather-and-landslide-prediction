# Landslide Susceptibility & Dynamic Hazard Pipeline Audit

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Study Region:** Meghalaya, Northeast India  
**Audit Directory:** `outputs/experiments/landslide_model_comparison/LANDSLIDE_PIPELINE_AUDIT.md`  
**Date:** October 30, 2026  

---

## 1. Executive Summary & Verification of Baseline

This audit independently inspects the landslide susceptibility and hazard modeling pipeline implemented in `D:\Downloads\honours`.

### Key Forensic Findings:
1. **Verified Inventory:** The project uses **38 verified rainfall-induced landslide occurrences** inside Meghalaya, filtered from a regional catalog of 67 events via `data/raw/boundary/meghalaya_boundary.geojson`. Non-rainfall triggers (such as open-cast rat-hole coal mining collapses) were strictly excluded.
2. **Authoritative XGBoost Baseline:** Under strictly symmetric, unbuffered 3-fold spatial block cross-validation, the verified baseline performance of the production XGBoost model is:
   - **Mean Spatial ROC-AUC:** **$0.5913 \pm 0.041$**
   - **PR-AUC:** **$0.6284$** (or $0.2285$ depending on prevalence scaling)
   - **Balanced Accuracy:** **$0.5742$**
   - **Brier Score:** **$0.1685$**
3. **Flawed Asymmetric Sampling Disclosed:** Earlier experiment logs recorded an artificially high ROC-AUC of $\sim 0.9623$. This was caused by a documented methodological flaw: extracting the maximum slope over a $5 \times 5$ window ($450\text{ m}$) for landslide points, while using single-pixel extraction for background points. When both classes were evaluated under an identical, unbuffered extraction protocol, the score settled at the true nominal baseline of **$0.5913$**.
4. **Dynamic Hazard Formula Verified:** The source code (`scripts/compute_dynamic_landslide_hazard.py`, line 133) strictly implements the NASA LHASA 2.0 Antecedent Rainfall Index (ARI):
   $$\mathbf{ARI} = 0.6 \cdot P_{24\text{h}} + 0.4 \cdot P_{72\text{h}}$$
5. **Operational Scope:** The pipeline implements **spatial geomorphological susceptibility** and **empirical antecedent rainfall hazard tiers (0–3)**. It does **not** predict deterministic future landslide occurrence times or calibrated failure probabilities within a 24-hour window.

---

## 2. Answers to the 11 System Audit Questions

### 1. Where the landslide inventory is stored
- Primary processed CSV: [`outputs/landslide/meghalaya_landslide_events.csv`](file:///d:/Downloads/honours/outputs/landslide/meghalaya_landslide_events.csv) (67 regional entries).
- Boundary polygon: [`data/raw/boundary/meghalaya_boundary.geojson`](file:///d:/Downloads/honours/data/raw/boundary/meghalaya_boundary.geojson).
- Original source extract: Third-party mirror of NASA Global Landslide Catalog (GLC/COOLR), originally compiled by NASA Goddard Space Flight Center (Kirschbaum et al., 2015).

### 2. How the 38 verified rainfall-induced events are represented
Each event is represented as a point observation with attributes:
- `latitude` and `longitude`: Geocoded point coordinates.
- `event_date`: Date of occurrence (ranging from 2007 to 2019). Sub-hourly timestamps (`event_time`) are largely blank/unrecorded.
- `landslide_trigger`: Categorized into `"rain"`, `"downpour"`, or `"continuous_rain"`.
- `location_accuracy`: Documented spatial uncertainty categorized into `"exact"`, `"1km"`, `"5km"`, `"10km"`, and `"50km"`. Only 8 events have $\le 1\text{ km}$ accuracy; 19 events carry $5\text{ km}$ uncertainty, and 11 events carry coarse $10 - 50\text{ km}$ uncertainty.

### 3. How non-landslide / background samples were generated
- In `scripts/train_landslide_susceptibility.py` (`sample_pseudo_absence_background`), **$N=190$ pseudo-absences** were sampled uniformly across the state ($5:1$ negative-to-positive ratio, total $N=228$).
- **Exclusion Zone:** A 3 km circular buffer ($33 \times 90\text{ m}$ pixels) was dilated around every positive landslide event using `scipy.ndimage.binary_dilation`. No background point is permitted within this exclusion zone.
- **Validity Filter:** Only pixels where `meghalaya_mask == 1` and `valid_mask == 1` were eligible.

### 4. Which terrain, rainfall, geographic, and environmental features are used
- **Static Geomorphometric Features (8 variables extracted from SRTM 90m DEM):**
  1. `elevation_mean`: Local mean elevation above sea level (meters).
  2. `slope_mean`: Mean slope gradient across the 90m cell (degrees).
  3. `slope_max`: Maximum slope within the 90m grid cell (degrees).
  4. `relief`: Elevation difference ($\text{max} - \text{min}$) within the cell.
  5. `slope_std`: Standard deviation of slope, reflecting surface roughness.
  6. `aspect_northness`: $\cos(\text{aspect})$, continuous index from $-1$ (south) to $+1$ (north).
  7. `aspect_eastness`: $\sin(\text{aspect})$, continuous index from $-1$ (west) to $+1$ (east).
  8. `curvature_mean`: Surface profile/planform curvature (positive = convex crest, negative = concave hollow).
- **Dynamic Rainfall Features:**
  - In dynamic hazard analysis, IMERG calibrated surface precipitation is accumulated over 24-hour ($P_{24\text{h}}$) and 72-hour ($P_{72\text{h}}$) rolling windows to compute the Antecedent Rainfall Index (ARI).
  - Rainfall is **not** used as a direct static predictor in the susceptibility training matrix due to the temporal mismatch between 2007–2019 landslides and 2023–2025 satellite grids.

### 5. How missing values are handled
- The SRTM DEM grid was preprocessed with a land/boundary valid mask (`valid_mask == 1`). Invalid pixels (water bodies, out-of-state nodata) were eliminated before pseudo-absence sampling.
- No synthetic imputation is required because all sampled points map directly to valid floating-point terrain rasters.

### 6. Which features are used by the existing XGBoost model
The existing production model (`outputs/models/landslide_susceptibility/xgboost_spatial_cv.joblib`) ingests all **8 terrain morphometry features** listed in Question 4.

### 7. How training, validation, and spatial block cross-validation work
- To prevent spatial autocorrelation leakage, the state of Meghalaya is partitioned into **3 geographically disjoint blocks**:
  - **West Garo Hills Block ($\text{lon} < 91.0^\circ\text{E}$):** 11 landslides, 54 background samples.
  - **Central Khasi Hills Block ($91.0^\circ\text{E} \le \text{lon} < 92.0^\circ\text{E}$):** 17 landslides, 82 background samples.
  - **East Jaintia Hills Block ($\text{lon} \ge 92.0^\circ\text{E}$):** 10 landslides, 54 background samples.
- In each fold, two geographic regions train the model, and the model is evaluated strictly on the remaining unseen geographic district. Feature standardization (`StandardScaler`) is fitted strictly on the training partition.

### 8. Whether the current model uses class weights, resampling, or hyperparameter tuning
- **Baseline XGBoost:** Used fixed default parameters (`n_estimators=40, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0`).
- **Class Weights:** None (`scale_pos_weight=1.0`).
- **Resampling:** No synthetic oversampling (SMOTE) or undersampling was applied in the baseline.
- **Tuning:** No automated grid search was used in the baseline script.

### 9. How the model generates susceptibility scores and maps
- For any feature vector $\mathbf{x}$, the model outputs $\hat{p} = \text{XGBoost}(\mathbf{x}) \in [0, 1]$.
- Statewide mapping flattens the valid $1,380 \times 3,720$ raster of Meghalaya into an $N \times 8$ matrix, executes batched inference, and writes the output back to NetCDF format (`data/processed/terrain/meghalaya_landslide_susceptibility_90m.nc4`).
- Tiers are classified into 4 bins:
  - **Low:** $\text{LSI} < 0.3$
  - **Moderate:** $0.3 \le \text{LSI} < 0.6$
  - **High:** $0.6 \le \text{LSI} < 0.8$
  - **Very High:** $\text{LSI} \ge 0.8$

### 10. How the frontend obtains and displays landslide results
- Static Susceptibility: Queried via `GET /layers/landslide/susceptibility?model={model}`. The backend returns an $8\times$ downsampled grid ($173 \times 465$) rendered on a Leaflet HTML5 canvas.
- Dynamic Rainfall Hazard: Queried via `GET /layers/landslide/dynamic?step={step}`. Displays 4 categorical tiers (0=Low, 1=Moderate, 2=High, 3=Very High) coupled with IMERG antecedent rainfall.
- Historical Events: Queried via `GET /layers/landslide/events`. Renders 38 white circle markers on Leaflet with popups showing date, trigger, accuracy, and district.
- Cell Inspector: Map clicks trigger `GET /cell_inspect?lat={lat}&lon={lon}&step={step}` displaying LSI score, susceptibility tier, antecedent rainfall (mm), and LHASA hazard level.

### 11. Whether existing model artifacts, reports, or predictions can be reused
- Yes. The baseline dataset files, NetCDF terrain features, and verified metrics from Phase 4 are fully intact and serve as the authoritative benchmark.

---

## 3. Four Predictive Capabilities Audit

| Predictive Capability | Status | Implementation Details | Scientific Evaluation |
| :--- | :---: | :--- | :--- |
| **1. Spatial Susceptibility** | **IMPLEMENTED** | XGBoost, Random Forest, and Logistic Regression on 90m SRTM terrain morphometry. | Honest spatial cross-validation yields ROC-AUC $0.5913$. Maps static geological slope-failure predisposition. |
| **2. Current / Recent Rainfall Hazard** | **IMPLEMENTED** | NASA LHASA 2.0 dual decision matrix combining static LSI with rolling IMERG antecedent rainfall ($0.6 P_{24\text{h}} + 0.4 P_{72\text{h}}$). | Valid heuristic situational awareness tier (0–3). Useful for advisory monitoring during monsoon downpours. |
| **3. Future Landslide Events** | **NOT IMPLEMENTED** | Requires real-time coupled slope hydrology and sub-hourly forecast inputs. | Not feasible with current data; 2007–2019 inventory lacks exact time-of-day trigger records. |
| **4. Calibrated Occurrence Probability** | **NOT IMPLEMENTED** | Requires logistic calibration against a dense spatio-temporal catalog of verified presences and verified non-events across time. | Presenting ARI or LSI as calibrated future probabilities would be a scientific fallacy. |
