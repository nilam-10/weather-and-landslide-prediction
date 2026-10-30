# Limitations and Future Work: Weather Nowcasting & Hazard Integration

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Target:** B.Tech Honours Major Project Thesis  
**Status:** **AUTHORITATIVE SCIENTIFIC LIMITATIONS DISCLOSURE**

---

## 1. Physical and Meteorological Limitations

### 1.1 The 6-Hour (+360-Minute) Predictability Ceiling
While the compound spatial loss objective (Exp D) successfully prevented complete convective extinction at the 6-hour forecast horizon (+360 minutes)—restoring maximum predicted rain rates to $7.47\text{ mm/hr}$ and achieving non-zero skill ($\text{CSI} = 0.0125 \pm 0.0129$, $\text{FSS} = 0.0526 \pm 0.0495$ on the confirmatory test benchmark)—**absolute forecasting skill at 6 hours remains weak**.

#### Scientific Explanation:
1. **Intrinsic Limit of Satellite Extrapolation:** Spatiotemporal deep learning models operating purely on historical satellite infrared brightness temperature ($T_b$) and IMERG precipitation estimate future trajectories via learned advection and morphological deformation. Beyond 3 to 4 hours, convective storm life cycles are dominated by **initiation, thermodynamic instability, boundary-layer convergence, and cold-pool dissipation**, rather than kinematic extrapolation of existing radar/satellite features.
2. **Absence of Thermodynamic Profiles:** The model has no access to atmospheric sounding data (e.g., Convective Available Potential Energy [CAPE], Convective Inhibition [CIN], vertical wind shear, or precipitable water).
3. **Threshold of Useful Skill:** In spatial verification literature (Roberts & Lean, 2008), an FSS score must exceed $\text{FSS}_{\text{useful}} = 0.5 + f_0/2 \approx 0.53$ to be considered practically useful for operational decision-making. At +360m, an FSS of $0.05$ is a statistically detectable mathematical signal over baseline collapse, but **it is not an operationally reliable weather warning product**.

---

### 1.2 Absence of Explicit Orographic Elevation Inputs
The Meghalaya Plateau features one of the steepest orographic boundaries in the world, with the southern Cherrapunji-Mawsynram escarpment rising abruptly from the Sylhet plains of Bangladesh ($< 20\text{ m}$ elevation to $> 1,500\text{ m}$ within $20\text{ km}$). Deep convection in this domain is heavily locked to stationary terrain features.
- In this study, the input tensor was restricted to 3 dynamic channels ($T_b$, $P$, mask) to isolate the mathematical impact of the loss functions.
- **Limitation:** Without static Shuttle Radar Topography Mission (SRTM) elevation ($DEM$), orographic slope, and aspect channels in the convolutional encoder, the neural network cannot anchor orographically forced stationary updrafts, allowing convective cores to drift unrealistically at extended forecast horizons (+180m to +360m).

---

## 2. Evaluation Methodology and Sample Limitations

### 2.1 Single-Day Validation Episode ($N=33$ Sequences)
The primary validation split used for model selection consists of a single active monsoon day (July 5, 2025, 33 sequences spanning 16.5 hours of rainfall):
- **Impact on Checkpoint Selection:** Validated as completely sound. Total validation loss provided a clear, convex convergence curve that selected optimal checkpoints without overfitting.
- **Impact on Statistical Inference:** With a 3-hour block size ($L=6$), a sample of 33 sequences contains only $\approx 5.5$ blocks. This small effective sample size induces discrete jumps in empirical resampling distributions. Consequently, primary statistical inferential power in this project relies on the larger 129-sequence confirmatory test benchmark.

---

### 2.2 Historical Benchmark Status of Confirmatory Test Set
The 3-day test cohort (July 12–14, 2023, $N=129$ sequences, 72.0 total observation hours) was utilized as a benchmark in earlier exploratory phases (Phase 3 baseline audit and Step 2 spatial verification).
- **Integrity Compliance:** In Step 5, all six model checkpoints were frozen prior to test evaluation and were **never** accessed during loss computation, backpropagation, learning rate scheduling, or checkpoint selection.
- **Limitation:** While execution was strictly non-adaptive and causally locked, this dataset represents a **locked historical benchmark episode**, rather than a completely unexposed, blind prospective test suite. True operational validation requires real-time prospective testing across future monsoon seasons.

---

### 2.3 Heuristic Nature of Storm Core Displacement Metric
The reported $\sim 110\text{ km}$ average convective core displacement at +180m was derived from a single global maximum (`argmax`) coordinate matching algorithm:
- **Limitation:** In complex monsoon environments, multiple convective cells often co-exist across the $260 \times 400\text{ km}$ study domain. If the model places peak rainfall in a western cell while the observation's peak occurs in an eastern cell, the distance is recorded as $> 150\text{ km}$, even if the model accurately placed a secondary convective core over the eastern cell.
- **Thesis Guidance:** This finding must be reported as a heuristic single-peak centroid distance diagnostic on matched frames, **not** as an object-based multi-cell tracking algorithm (such as MODE or CRA).

---

## 3. System Boundary & Downstream Landslide Integration

### 3.1 Separation of Weather Nowcasting from Landslide Hazard
A central goal of the overall Major Project synopsis is coupling precipitation nowcasting with landslide susceptibility modeling. It is critical to state clearly that **Step 5 focused exclusively on the atmospheric nowcasting sub-system**:

```
[Satellite / Precipitation Inputs]
              │
              ▼
┌───────────────────────────────┐
│   SimVP Nowcasting Engine     │ ◄── EVALUATED & VALIDATED IN STEP 5
│   (Compound Spatial Loss)     │     (+30m to +360m rain rates & probabilities)
└──────────────┬────────────────┘
               │ (Future Rain Rates)
      ═════════╪══════════════════════════════════════════════════════════
               │ SYSTEM INTEGRATION BOUNDARY (NOT EVALUATED IN STEP 5)
               ▼
┌───────────────────────────────┐
│  Phase 4 Landslide Susceptibility│ ◄── STATIC SPATIAL MODEL (ROC-AUC = 0.5913)
│   (Random Forest / Terrain)   │     (Evaluated on 38 GSI inventory points)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Dynamic Antecedent Hazard Tier│ ◄── HEURISTIC 2D DECISION MATRIX
│ (Low, Moderate, High, Severe) │     (Rule-based lookup table)
└───────────────────────────────┘
```

#### Mandatory Thesis Disclosures:
1. **No Coupled Landslide Verification:** The improved precipitation fields from Exp D have **not yet been piped into the downstream Phase 4 Random Forest model**.
2. **No Claim of Improved Landslide Lead Time:** The candidate must **not** claim that Exp D improved landslide prediction accuracy, warning lead time, or false alarm rates. Such claims are unverified until end-to-end coupled validation is conducted.
3. **Preserving Conceptual Boundaries:** Maintain the strict distinction between:
   - *Precipitation Nowcast ($P$ in $\text{mm/hr}$):* Continuous meteorological rate.
   - *Convective Probability ($p_{\text{conv}}$):* Exceedance probability of $5.0\text{ mm/hr}$.
   - *Landslide Susceptibility ($S$):* Static terrain-based likelihood ($0$ to $1$).
   - *Antecedent Hazard Tier:* Discrete operational category (Low, Moderate, High, Severe) assigned via a deterministic 2D lookup matrix.

---

## 4. Prioritized Future Research Directions

1. **Topographic Encoder Conditioning (High Priority):**
   Incorporate 30m SRTM DEM elevation, slope gradient, and aspect as static input channels into the SimVP convolutional encoder to anchor orographic convection along the southern Meghalaya escarpment.
2. **Coupled End-to-End Hydrometeorological Validation (High Priority):**
   Pipe the verified Exp D rainfall forecasts into the dynamic landslide hazard engine and evaluate retrospective warning metrics against documented landslide occurrence dates.
3. **Object-Based Verification (MODE / CRA) (Medium Priority):**
   Implement the Method for Object-Based Diagnostic Evaluation (MODE) to decompose forecast errors into explicit object attributes: cell centroid displacement, area bias, orientation angle, and intensity distribution.
4. **Numerical Weather Prediction (NWP) Blending (Medium Priority):**
   Develop a blended nowcasting system that transitions smoothly from satellite spatiotemporal extrapolation at $+30\text{m}$ to $+120\text{m}$ toward convection-permitting WRF/NCUM numerical weather prediction fields at $+180\text{m}$ to $+360\text{m}$.
