# Phase 4 Feasibility Audit: Next-24-Hour Supervised Landslide Prediction
## RBU Honours Major Project (2026–2027)

**Scientific Determination**: **SUPERVISED NEXT-24-HOUR PROBABILITY FORECASTING IS FUNDAMENTALLY BLOCKED BY INSUFFICIENT AUTHENTIC OBSERVATIONS.**

---

### 1. Data Inventory Audit & Failure Timeline Mismatch

1. **Temporal Epoch Disconnection**:
   - The verified NASA Global Landslide Catalog (GLC/COOLR) for Meghalaya contains **38 rainfall-triggered landslide events**.
   - These events span historical years **2007 to 2019** (e.g., 2015: 9 events; 2017: 7 events; 2014: 6 events).
   - The authentic high-resolution half-hourly NASA IMERG/MERGIR satellite cubes acquired and causally verified in our workspace cover active Southwest Monsoon periods in **2023, 2024, and 2025**.
   - **Zero historical landslide events occurred during the 2023–2025 satellite observation episodes.**

2. **Absence of Hourly Failure Initiation Timestamps**:
   - 100% of the 38 verified records in the NASA GLC catalog originate from media, police reports, and district press releases.
   - Event times are recorded with day-level nominal timestamps (e.g., `0:00:00 UTC`).
   - Landslides in steep terrain are triggered by intense sub-hourly convective bursts (e.g., 30–60 minute rainfall intensities exceeding 25 mm/hr) or multi-day pore-water accumulation. Without exact failure initiation timestamps, sub-daily temporal sequence models (such as ConvLSTM or GRU) cannot causally align rainfall bursts with failure moments.

---

### 2. Extreme Class Imbalance & The "Verified Absence" Fallacy

1. **Extreme Spatiotemporal Imbalance**:
   - Evaluating a 24-hour forecast model across the $25 \times 40$ ($0.1^\circ$) grid over a 13-year period involves:
     $$\text{Total Cell-Days} = 1,000 \text{ cells} \times 365 \text{ days} \times 13 \text{ years} \approx 4,745,000 \text{ cell-days}$$
   - With only 38 documented events statewide, positive prevalence is:
     $$\text{Prevalence} = \frac{38}{4,745,000} \approx 8.01 \times 10^{-6}$$
   - Any supervised neural network (such as a GRU or LSTM sequence classifier) collapses trivially to the all-zero majority class (yielding 99.999% accuracy while exhibiting 0.000 recall).

2. **The "Unreported != Landslide-Free" Fallacy**:
   - Meghalaya features dense subtropical jungle, unpopulated tribal reserve forests, and steep gorges where small debris flows and rockfalls occur without human fatalities or press reporting.
   - Treating all unrecorded grid cells on all days as "verified non-landslide negatives" introduces severe label noise.
   - In ecological and geotechnical modeling, this is universally recognized as **presence-only / presence-background** data, which mathematically supports **relative susceptibility mapping**, but **violates the fundamental assumptions of supervised temporal classification**.

---

### 3. Geocoding Uncertainty & Spatial Smearing

- 76.3% of the historical events have spatial uncertainty of **5 km to 50 km** (median uncertainty = 5 km).
- At 90 m resolution, a 5 km uncertainty circle contains:
  $$\text{Pixels} \approx \frac{\pi \times (5000)^2}{90 \times 90} \approx 9,700 \text{ terrain pixels}$$
- Attempting to train a supervised machine learning model by assigning a precise binary label $Y=1$ to an arbitrary nominal 90m pixel causes gradient conflict where steep escarpments and flat river floodplains receive the exact same failure label.

---

### 4. Defensible Architecture Decision

In accordance with strict scientific integrity:
1. **We DO NOT train a GRU or temporal neural network on fabricated timestamps or synthetic labels.**
2. **We RETAIN AND VALIDATE the two-tier physically defensible architecture**:
   - **Tier 1 (Static Susceptibility)**: Machine-learning Landslide Susceptibility Index (LSI) trained on spatial block cross-validation with geocoding uncertainty buffering.
   - **Tier 2 (Dynamic Rainfall Triggering)**: Empirical decision-matrix hazard levels (Low, Moderate, High, Very High) coupled with multi-day antecedent rainfall (NASA LHASA 2.0 formulation).
