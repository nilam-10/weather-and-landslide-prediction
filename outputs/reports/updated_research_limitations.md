# Updated Research Limitations & Scientific Integrity Report

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Institutional Context**: RBU Honours Major Project (2026–27)  
**Study Domain**: Meghalaya, Northeast India ($89.5^\circ\text{--}93.5^\circ\text{E}, 24.5^\circ\text{--}27.0^\circ\text{N}$)  
**Date**: October 2026 (Updated Milestone Audit)  

---

## 1. Meteorological Nowcasting Limitations

### 1.1 Satellite Proxy-Label Retrieval Caveat
- **Retrieval Interdependence**: The weather nowcasting models predict future NASA GPM IMERG V07B precipitation exceedance ($P \ge 5.0\text{ mm/hr}$) from sequential NASA GPM MERGIR geostationary infrared brightness temperatures ($T_b$) and past IMERG rainfall.
- **Scientific Implication**: Because NOAA/NCEP/CPC MERGIR geostationary IR is itself one of the multi-satellite inputs used inside the NASA IMERG retrieval algorithm, evaluation against IMERG constitutes an evaluation against a **satellite proxy retrieval**, not an independent ground-truth rain gauge or Doppler weather radar network.
- **Operational Reality**: IMERG Final Run incorporates monthly Global Precipitation Climatology Centre (GPCC) gauge calibration, introducing a 3–4 month release latency. Prospective operational deployment requires ingesting IMERG Early Run (with ~4-hour latency) or real-time local automatic weather station (AWS) telemetry.

### 1.2 The 6-Hour (+360m) Physical Predictability Barrier
- **Empirical Findings**: Across held-out test evaluations (July 6–7, 2025, N=94,000 pixels), Critical Success Index (CSI) skill exhibits steep non-linear attenuation:
  - $+30\text{ min}$: $\text{CSI} = 0.4150\text{--}0.4225$ (+5.6% vs Persistence)
  - $+60\text{ min}$: $\text{CSI} = 0.2960\text{--}0.3045$ (+17.5% vs Persistence)
  - $+120\text{ min}$: $\text{CSI} = 0.1359\text{--}0.1415$ (+23.9% vs Persistence)
  - $+180\text{ min}$: $\text{CSI} = 0.0540\text{--}0.0585$ (+94.2% vs Persistence)
  - $+360\text{ min}$: $\text{CSI} = 0.0210$ (Lagged Persistence: $0.0082$, Optical Flow: $0.0145$)
- **Thermodynamic Law of Convective Dissipation**: In the rugged orographic setting of the Shillong Plateau, convective storm cells typically initiate, mature, and dissipate within 45 to 90 minutes. Beyond 2 to 3 hours, precipitation is governed by boundary layer moist convection and thermodynamic growth mechanisms that cannot be extrapolated from past satellite brightness temperatures.
- **Scientific Conclusion**: Observational satellite advection nowcasting cannot solve 6-hour convective forecasting. Validated 6-hour predictions strictly require coupling deep learning nowcasters with dynamical Numerical Weather Prediction (WRF / NCMRWF 3 km) models.

---

## 2. Landslide Susceptibility & Hazard Modeling Limitations

### 2.1 The Asymmetric Sampling Bias Artifact
- **Discovered Methodological Flaw**: Prior reported spatial cross-validation results cited near-perfect ROC-AUC of $0.9666\text{--}0.9911$. The research audit revealed that positive landslide events had features extracted using a $450\text{ m}$ circular window, assigning the maximum slope in that window (mean $38.4^\circ$), whereas pseudo-absences were sampled strictly at a single nominal pixel (mean $14.2^\circ$).
- **Symmetric Unbiased Evaluation**: When evaluated using strictly symmetric sampling (single nominal coordinate for both positives and pseudo-absences):
  - Logistic Regression ROC-AUC drops from $0.9911 \to \mathbf{0.5379}$ (PR-AUC: $0.2554$).
  - XGBoost Classifier ROC-AUC drops from $0.9666 \to \mathbf{0.5913}$ (PR-AUC: $0.2285$).
- **Scientific Root Cause**: Landslide coordinates sourced from news-media catalogs (NASA Global Landslide Catalog) have reported location uncertainties ranging from $1\text{ km}$ to $50\text{ km}$. News articles typically report the nearest town, village, or highway name (e.g. "Landslide near Cherrapunji"), placing the coordinate at the administrative post office or town center (often located on flat plateaus or valley bottoms), rather than the steep failure scarp.
- **Requirement for Fine-Scale Modeling**: Symmetrically extracting 90 m terrain features at noisy coordinates bounds classification skill near random chance ($0.54\text{--}0.61$). Reliable high-resolution susceptibility mapping requires authoritative scarp polygon inventories (such as Geological Survey of India Bhukosh maps) obtained through formal institutional data-sharing agreements.

### 2.2 Extreme Data Sparsity & Supervised 24-Hour Forecasting (`BLOCKED_BY_DATA`)
- **Event Scarcity**: Across the entire state of Meghalaya (22,429 km²), only **38 rainfall-triggered landslide events** are recorded over a 17-year period (2007–2024), representing an average of $2.2\text{ events/year}$.
- **Spatiotemporal Class Ratio**:
  $$\text{Prevalence} = \frac{38\text{ events}}{17\text{ years} \times 365\text{ days} \times 1,000\text{ cells/day}} \approx 6.1 \times 10^{-6} \quad (< 1 \text{ in } 160,000)$$
- **Scientific Honesty Mandate**: Training a supervised deep recurrent neural network (such as GRU/LSTM) on 38 positive events scattered across 17 years would result in severe statistical overfitting or trivial zero-prediction.
- **Appropriate Modeling Paradigm**: The system implements an **exploratory static susceptibility baseline** coupled with an empirical antecedent rainfall trigger ($A = 0.6 P_{24\text{h}} + 0.4 P_{72\text{h}}$, inspired by NASA LHASA principles). It is explicitly designated as a **decision matrix**, NOT a calibrated next-day landslide probability.

---

## 3. Multimodal Integration & Software Architecture Limitations

### 3.1 Non-Multiplication of Weather and Terrain Probabilities
- Following strict scientific principles, the platform rejects the mathematically invalid operation of multiplying weather nowcast probability by static terrain susceptibility:
  $$P(\text{Joint Disaster}) \neq P(\text{Convection}) \times P(\text{Susceptibility})$$
- Convection exceedance and static slope susceptibility represent completely different physical quantities with distinct temporal units, spatial scales, and conditional assumptions. Multiplying them creates an uncalibrated, uninterpretable index that gives a false impression of predictive precision.
- The platform presents independent layers side-by-side in the Web GIS interface, with a boolean spatial co-location flag ($\text{Hazard} \ge 2 \land P_{\text{nowcast}} \ge 0.5$) clearly labelled as **not** an operational disaster warning.

### 3.2 Platform Operational Scope
- The system is a **retrospective research platform** evaluated on historical monsoon windows (July 1–7, 2024 and July 1–7, 2025).
- It is not connected to real-time Doppler radar feeds or operational telemetric slope sensors and must not be used as an operational civil defense warning system.
