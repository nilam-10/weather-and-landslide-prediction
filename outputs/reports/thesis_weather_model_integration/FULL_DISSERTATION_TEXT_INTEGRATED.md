# MULTIMODAL AI PLATFORM FOR CONVECTIVE NOWCASTING AND LANDSLIDE RISK PREDICTION
## B.Tech (Honours) Major Project Dissertation

**Institution:** Department of Computer Science & Engineering, Shri Ramdeobaba College of Engineering and Management, Nagpur  
**Programme:** Bachelor of Technology (Honours) in Computer Science & Engineering / AI & ML  
**Academic Session:** 2026–2027  
**Project Group Members:**
- Nilam Bhojwani (Roll No. AIML-B-33)
- Moli Agrawal (Roll No. CSE-B-35)
- Grisha Lala (Roll No. CSE-A-53)
- Shloak Nioding (Roll No. CSE-C-34)

**Project Guide:** Dr. Amit Pimpalkar, Assistant Professor, Department of Computer Science & Engineering  
**Document Status:** Final Audited Dissertation Copy (Integrated with Step 8 Research Findings)  
**Location:** `outputs/reports/thesis_weather_model_integration/FULL_DISSERTATION_TEXT_INTEGRATED.md`

---

## ABSTRACT

The North Eastern Region (NER) of India, and specifically the Meghalaya Plateau ($25.0^\circ\text{N} - 26.2^\circ\text{N}, 89.8^\circ\text{E} - 92.8^\circ\text{E}$), is globally renowned for extreme orographic precipitation and severe hydrometeorological hazards. The abrupt topographic ascent of the Meghalaya escarpment directly intercepting the southwest Indian summer monsoon generates high-intensity convective cloud clusters that frequently trigger devastating shallow landslides, debris flows, and flash floods. Existing disaster management workflows rely on disconnected, coarse-resolution numerical weather models and static heuristic susceptibility maps that fail to provide actionable, location-specific warnings at convective timescales (0 to 6 hours).

This dissertation presents the design, implementation, and rigorous scientific evaluation of an end-to-end **Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction**. The system integrates multi-source earth observation data—including high-resolution precipitation fields from NASA GPM IMERG, thermal infrared brightness temperatures from geostationary satellites, high-resolution ALOS PALSAR digital elevation models (DEM), and Geological Survey of India (GSI) landslide inventories—into a unified spatiotemporal AI engine and an operational web GIS dashboard.

A primary scientific challenge addressed in this work is the severe **intensity attenuation** inherent to spatiotemporal deep learning models (such as ConvLSTM and standard SimVP architectures) trained with conventional mean squared error or Smooth-$L_1$ regression objectives. Under conditional mean optimization, models smooth out localized convective storm cells at lead times beyond 60 minutes, driving convective Critical Success Index (CSI) and Fractions Skill Score (FSS) to zero. To overcome this limitation without destabilizing domain-wide bulk error, we developed and replicated a **Compound Spatial Loss Objective (Exp D)** combining intensity-weighted $L_1$ regression, multiscale differentiable Soft Fractions Skill Score (Soft-FSS), and categorical occurrence binary cross-entropy.

Across a rigorous three-seed replication protocol ($N=3$ seeds: 42, 123, 999) on a 38-day training corpus ($1,764$ sequences), the compound objective dramatically restores intermediate-horizon convective nowcasting skill:
1. At $+180\text{ minutes}$ (3 hours ahead, $\ge 5.0\text{ mm/hr}$), Pointwise CSI improves by **$6.07\times$** on validation ($0.0183 \pm 0.0107 \to 0.1111 \pm 0.0184$) and by **$10.8\times$** on a locked historical benchmark ($0.0120 \pm 0.0132 \to 0.1300 \pm 0.0231$). Circular FSS ($30\text{ km}$) improves by **$5.43\times$** on validation ($0.0740 \pm 0.0535 \to 0.4016 \pm 0.0523$) and by **$8.2\times$** on the historical benchmark ($0.0527 \pm 0.0556 \to 0.4333 \pm 0.0424$).
2. At $+120\text{ minutes}$, Circular FSS ($30\text{ km}$) increases from $0.3232 \pm 0.1158$ to $\mathbf{0.5560 \pm 0.0307}$ ($+72.0\%$).
3. Stationary temporal block bootstrapping ($L=12$ frames / 6.0 hours, $B=1,000$ resamples) confirms that intermediate-horizon gains are statistically significant at $p < 0.001$.
4. Crucially, bulk domain-averaged Quantitative Precipitation Forecast (QPF) error is preserved: at $+180\text{m}$, RMSE difference between Exp D and the baseline is statistically non-significant ($-0.0008\text{ mm/hr}$, $p = 0.417$, 95% CI: $[-0.0616, +0.0483]\text{ mm/hr}$), demonstrating that heavy-cell retention was achieved without inflating false-alarm domain error.
5. Forensic spatial audit reveals that at short horizons (+30m to +60m), peak convective cells are localized with median displacement errors of $14.9\text{ km}$ to $27.2\text{ km}$. At +180m, mean displacement is $109.9\text{ km}$ on validation and $156.6\text{ km}$ on test; detailed analysis shows that in 100% of large-displacement cases ($> 100\text{ km}$), multiple simultaneous convective clusters were active across the domain.
6. At $+360\text{ minutes}$ (6 hours ahead), the compound objective prevents structural extinction ($\text{FSS} = 0.0526 \pm 0.0495$, max rain rate $7.47\text{ mm/hr}$ vs baseline total collapse to $0.0000$ and $1.75\text{ mm/hr}$); however, absolute skill remains low ($\text{FSS} \approx 0.05$), marking the physical predictability ceiling of satellite-only spatiotemporal modeling.

For landslide hazard assessment, a spatial cross-validated Random Forest susceptibility model trained on morphometric and geo-environmental factors achieves an AUC-ROC of $0.842 \pm 0.031$, coupled with an empirical dynamic hazard tier system driven by antecedent rainfall. The entire pipeline is deployed via a high-performance FastAPI backend ($< 150\text{ ms}$ inference latency) and an interactive React/Leaflet geospatial dashboard. All findings are presented with full disclosure of methodological boundaries, providing an auditable, reproducible contribution to AI-based natural hazard mitigation.

---

## CHAPTER 1: INTRODUCTION AND PROBLEM CONTEXT

### 1.1 Geographic Setting: The Meghalaya Plateau
The state of Meghalaya in Northeast India occupies an elevated plateau bounded by the Assam Valley to the north and the Bengal floodplains to the south. Rising precipitously from near sea level to elevations exceeding $1,500\text{ meters}$ within a horizontal span of less than $20\text{ km}$, the southern escarpment creates one of the most intense orographic precipitation triggers on Earth. During the Indian Summer Monsoon (June–September), warm, moisture-laden southerly airflows from the Bay of Bengal are forced upward over the hills, condensing into massive mesoscale convective systems (MCS) and stationary thunderstorm clusters. Cherrapunji (Sohra) and Mawsynram receive world-record annual rainfall exceeding $11,000\text{ mm}$.

### 1.2 Hydrometeorological Cascades and Slope Instability
Extreme rainfall in this terrain does not act as an isolated meteorological event; it initiates catastrophic cascading geohazards:
1. **Pore-Water Pressure Elevation:** Rapid infiltration through weathered regolith saturates soil mantles overlying impervious gneissic and sandstone bedrock, reducing effective normal shear strength and triggering translational and rotational slides along steep highway corridors (e.g., NH-06 connecting Meghalaya to Assam and Mizoram).
2. **Flash Flooding and Debris Flows:** Rapid surface runoff down steep river valleys causes channel scour, structural infrastructure destruction, and loss of life.
3. **The Warning Gap:** Current operational weather forecasts issued by regional meteorological centers rely on coarse-resolution global or regional numerical weather prediction (NWP) models (e.g., IMD WRF at 3 km to 12 km grid spacing). While NWP resolves synoptic monsoon depressions well at 24 to 72 hours, it exhibits significant spatial-temporal phase errors in resolving the precise location, initiation, and hourly evolution of convective storm cores over complex terrain at lead times of 0 to 6 hours.

### 1.3 Project Motivation and Objectives
To address this gap, this B.Tech Honours Major Project establishes a multi-tiered, multimodal AI platform that unifies short-term convective precipitation nowcasting with geospatial landslide hazard intelligence.

The formal objectives approved in the project synopsis are:
1. **0–6-Hour Convective Nowcasting:** Develop, train, and validate spatiotemporal deep learning models to predict multi-lead precipitation fields (+30, +60, +120, +180, and +360 minutes) at $0.10^\circ \approx 10.55\text{ km}$ spatial resolution over Meghalaya.
2. **Landslide Susceptibility and Dynamic Hazard Tiers:** Develop and spatially cross-validate machine learning models to map baseline landslide susceptibility from terrain morphometry, integrating multi-day antecedent rainfall to classify dynamic hazard alert levels.
3. **Benchmarking Published Literature Methodologies:** Adapt and benchmark methodologies from recent literature on NER landslide and nowcasting modeling (e.g., Rani et al. 2026, Mihu et al. 2026, Badavath & Sahoo 2025, Kassoumeh et al. 2025, Jash et al. 2025).
4. **Research into Model Architectures and Spatial Losses:** Conduct rigorous empirical research to solve convective intensity attenuation in spatiotemporal nowcasting models.
5. **Operational Full-Stack Web Platform:** Deploy models into an interactive, containerized web GIS platform featuring REST APIs and interactive map visualization with sub-second response times.

---

## CHAPTER 2: LITERATURE REVIEW AND RESEARCH GAPS

### 2.1 Benchmark Papers Analysis
In accordance with Section 4 of the approved project synopsis, we surveyed and benchmarked five representative methodologies from recent literature:

1. **Rani et al. (2026) — *Transformer-Based Rainfall-Sequence Landslide Probability Modeling*:**
   - *Core Contribution:* Employs temporal attention mechanisms over antecedent daily rainfall sequences to capture cumulative moisture saturation.
   - *Adaptation in this Project:* Implemented rolling antecedent precipitation aggregations ($A_1, A_3, A_7, A_{14}, A_{30}$) to capture soil moisture memory, establishing multi-day precipitation triggering thresholds.
2. **Mihu et al. (2026) — *Landslide Susceptibility and SHAP Explainability in the Eastern Himalayas*:**
   - *Core Contribution:* Evaluates gradient boosted decision trees (XGBoost and LightGBM) on the Dibang Valley of Arunachal Pradesh, utilizing TreeSHAP for geo-environmental feature attribution.
   - *Adaptation in this Project:* Benchmarked Random Forest, XGBoost, and LightGBM using spatial block cross-validation over Meghalaya, generating global and local SHAP summary curves to verify that slope gradient and drainage proximity dominate failure mechanics.
3. **Badavath & Sahoo (2025) — *Machine Learning Landslide Susceptibility Mapping in Meghalaya*:**
   - *Core Contribution:* Leverages the Geological Survey of India (GSI) landslide inventory across Meghalaya, combining multi-criteria conditioning factors (slope, aspect, curvature, lithology, distance to faults/roads).
   - *Adaptation in this Project:* Adopted the GSI Meghalaya historical inventory as our ground-truth spatial polygon dataset, computing 12 terrain morphometric and distance covariates derived from ALOS PALSAR 30 m DEM.
4. **Kassoumeh et al. (2025) — *Multimodal Radar-Satellite Fusion for Heavy Rainfall Nowcasting*:**
   - *Core Contribution:* Demonstrates that fusing thermal infrared cloud-top temperatures with surface precipitation estimates improves early storm detection before radar reflectivity strengthens.
   - *Adaptation in this Project:* Engineered multimodal input tensors combining geostationary thermal infrared brightness temperature ($T_b$, $10.8\ \mu\text{m}$), IMERG surface precipitation rate ($P$), and observational validity masks ($M$) into 4-frame historical input sequences ($T_{\text{in}}=4$).
5. **Jash et al. (2025) — *Thunderstorm Evolution Prediction Using Spatiotemporal Deep Learning*:**
   - *Core Contribution:* Explores deep neural networks for multi-lead radar nowcasting, highlighting the rapid loss of high-reflectivity convective cores at lead times exceeding 2 hours.
   - *Adaptation in this Project:* Formed the scientific baseline for our loss function investigation, showing that conventional MSE objectives wash out convective storm cores.

### 2.2 The Classical Nowcasting Bottleneck: Conditional Mean Blur
When spatiotemporal deep learning models are trained using pointwise regression losses—such as Mean Squared Error (MSE, $L_2$), Mean Absolute Error (MAE, $L_1$), or Huber / Smooth-$L_1$ loss—the model naturally converges toward the conditional expectation:
$$\hat{\mathbf{Y}} = \mathbb{E}[\mathbf{Y} \mid \mathbf{X}_{t-3:t}]$$
Because the exact future trajectory of convective storm initiation and dissipation is stochastic and subject to rapid chaotic turbulence, multiple plausible futures exist. Under an $L_2$ objective, the mathematical minimizer of expected loss is the arithmetic mean across all plausible states. Consequently, as the forecast lead time extends from $+30\text{m}$ to $+180\text{m}$ and $+360\text{m}$, the network produces an increasingly smoothed, blurred spatial field. While this strategy minimizes domain-wide RMSE by avoiding high-intensity "double penalty" misplacements, it completely extinguishes localized heavy rainfall peaks ($\ge 5.0\text{ mm/hr}$), rendering nowcasts useless for convective hazard warning.

---

## CHAPTER 3: PLATFORM ARCHITECTURE AND SYSTEM ENGINEERING

The platform is engineered as an enterprise-grade, microservice-oriented GIS platform comprising data ingestion, AI inference, and presentation tiers:

```
+---------------------------------------------------------------------------------------------------------+
|                                    DATA ACQUISITION & INGESTION PIPELINE                                |
+---------------------------------------------------------------------------------------------------------+
| - NASA GPM IMERG Half-Hourly Precipitation (0.1 deg)                                                    |
| - Satellite Thermal Infrared Brightness Temperature (10.8 um)                                           |
| - ALOS PALSAR 30m Digital Elevation Model (Terrain Morphometry)                                         |
| - GSI Historical Landslide Inventory (Georeferenced Polygon Catalog)                                    |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                  SPATIOTEMPORAL AI INFERENCE ENGINE                                     |
+---------------------------------------------------------------------------------------------------------+
| [Production Baseline: ConvLSTM]                 [Research Advance: SimVPWeather]                        |
| - 161K parameters, recurrent cell gating        - 337K parameters, non-recurrent GTC translator          |
| - Frozen Canonical Baseline (Seed 42)           - Compound Loss Optimization (Exp D)                    |
| - +30m to +360m Lead Times                      - Multi-Seed Replicated (Seeds 42, 123, 999)             |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                LANDSLIDE SUSCEPTIBILITY & HAZARD TIER ENGINE                            |
+---------------------------------------------------------------------------------------------------------+
| - Spatial Cross-Validated Random Forest Classifier (AUC-ROC = 0.842 +/- 0.031)                         |
| - Terrain Conditioning Factors (Slope, Aspect, STI, TWI, Curvature, Fault Proximity)                   |
| - Dynamic Antecedent Moisture Index (Rolling 24h, 72h, 7d Rainfall Coupling)                            |
| - 4-Tier Hazard Alert Matrix (Low, Moderate, High, Severe)                                              |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                 APPLICATION & GEOSPATIAL DELIVERY TIER                                  |
+---------------------------------------------------------------------------------------------------------+
| - FastAPI REST Backend: Asynchronous PyTorch inference (< 150 ms latency)                              |
| - Interactive React / Leaflet Web GIS Dashboard: Dual-layer synoptic mapping, live probe inspector     |
| - Explainable AI Microservice: Real-time SHAP terrain attribution and Integrated Gradients heatmaps     |
+---------------------------------------------------------------------------------------------------------+
```

### 3.1 Data Acquisition and Domain Standardization
- **Study Domain:** The spatial domain encompasses the entire state of Meghalaya and adjacent drainage catchments ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$).
- **Spatial Grid:** Resampled onto a regular $25 \times 40$ latitude-longitude grid with cell spacing of $0.10^\circ \approx 10.55\text{ km}$ ($11.08\text{ km}$ latitudinal, $10.03\text{ km}$ longitudinal).
- **Temporal Cadence:** 30-minute intervals aligned with IMERG half-hourly observations.
- **Input Channels ($C=3$):**
  1. *Channel 0 ($T_b$):* Thermal infrared brightness temperature ($10.8\ \mu\text{m}$), normalized to $[0, 1]$ via min-max scaling ($190\text{ K} - 310\text{ K}$).
  2. *Channel 1 ($P$):* Surface precipitation rate ($\text{mm/hr}$), log-transformed as $\log_{1p}(P) = \log(1 + P)$ to stabilize heavy-tailed convective distributions.
  3. *Channel 2 ($M$):* Binary observation validity mask ($1$ for valid sensor data, $0$ for invalid/missing).
- **Sequence Dimension:** Inputs comprise 4 past frames ($t-90\text{m}, t-60\text{m}, t-30\text{m}, t$; shape $B \times 4 \times 3 \times 25 \times 40$), predicting 5 future leads ($+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}, +360\text{m}$; leads $1, 2, 4, 6, 12$).

### 3.2 Canonical Production Model: Multi-Horizon ConvLSTM
The operational production baseline deployed in the live system is a multi-horizon **ConvLSTM** network comprising $161,546$ parameters. It employs a two-layer ConvLSTM recurrent cell with $3 \times 3$ convolutional kernels, an embedding dimension of 32 channels, and multi-head 2D convolutional output projection layers. The model outputs occurrence classification logits and continuous rain rates through parallel heads.

---

## CHAPTER 4: WEATHER NOWCASTING RESEARCH: SPATIOTEMPORAL MODELING AND COMPOUND SPATIAL LOSS OPTIMIZATION

### 4.1 Spatiotemporal Architecture: SimVPWeather
To eliminate the sequential compounding of recurrent cell hidden states across long forecast horizons, we adapted the **Simple Video Prediction (SimVP)** architecture for meteorological nowcasting (`SimVPWeather`, $337,445$ parameters). SimVP decouples spatial feature extraction from temporal sequence translation:
1. **Spatial Encoder:** A 2-block convolutional neural network operating independently on each input frame ($B \cdot T_{\text{in}} \times 3 \times 25 \times 40$), mapping input features to latent representations ($B \cdot T_{\text{in}} \times 32 \times 25 \times 40$).
2. **Spatiotemporal Translator:** A non-recurrent stack of 3 Gated Temporal Convolution (GTC) blocks with depthwise separable convolutions that translate the concatenated historical sequence ($B \times 128 \times 25 \times 40$) directly into multi-lead future latent states ($B \times 160 \times 25 \times 40$).
3. **Spatial Decoder:** Multi-head convolutional projection layers predicting:
   - *Head 1 (Occurrence Logits):* Binary presence/absence logits for each lead time.
   - *Head 2 (QPF Rain Rates):* Continuous precipitation rates ($\text{mm/hr}$) enforced non-negative via Softplus activation.

### 4.2 The Compound Spatial Loss Objective (Exp D)
To actively combat convective intensity attenuation without destabilizing domain-wide bulk error, we formulated a compound objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{\mathbf{Y}}_{\text{cat}}, \mathbf{Y}_{\text{cat}}) + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}(\hat{\mathbf{Y}}_{\text{qpf}}, \mathbf{Y}_{\text{qpf}}) + \lambda_{\text{fss}}\mathcal{L}_{\text{Soft-FSS}}(\hat{\mathbf{Y}}_{\text{qpf}}, \mathbf{Y}_{\text{qpf}})$$

1. **Categorical Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$):** Enforces sharp classification boundaries for precipitation presence across all leads.
2. **Intensity-Weighted $L_1$ Regression ($\mathcal{L}_{\text{reg}}$):** Rather than treating dry pixels and extreme storm cells identically, the regression loss weights errors proportionally to observed rainfall intensity:
   $$\mathcal{L}_{\text{reg}} = \frac{1}{\sum M_{i,j}} \sum_{i,j} M_{i,j} \cdot w(Y_{i,j}) \cdot \left| \hat{Y}_{i,j} - Y_{i,j} \right|$$
   where the intensity weight function is:
   $$w(Y) = 1.0 + \alpha \cdot \log_{1p}\left(\max\left(0, \frac{Y}{\tau}\right)\right)$$
   with convective threshold $\tau = 5.0\text{ mm/hr}$ and scaling factor $\alpha = 0.5$. This guarantees that a $5\text{ mm/hr}$ error on a $30\text{ mm/hr}$ convective core receives significantly higher penalty than an identical error in light stratiform drizzle.
3. **Multiscale Differentiable Soft Fractions Skill Score ($\mathcal{L}_{\text{Soft-FSS}}$):** Pointwise losses penalize spatial displacement twice ("double penalty"). To train the network to place storm cells in approximately the correct geographic region, we formulated a differentiable surrogate of the Fractions Skill Score:
   $$\hat{I}_{i,j} = \sigma\left(\frac{\hat{Y}_{i,j} - \tau}{T}\right), \quad I_{i,j} = \mathbb{I}(Y_{i,j} \ge \tau)$$
   where $\sigma(\cdot)$ is the sigmoid function with temperature $T=0.5$. Fractional neighborhood coverage fields $\hat{F}_{k}$ and $F_{k}$ are generated via spatial 2D circular convolution with isotropic Euclidean kernel $K_k$. The multiscale loss is defined as:
   $$\mathcal{L}_{\text{Soft-FSS}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \left[ 1.0 - \text{FSS}_k(\hat{\mathbf{Y}}, \mathbf{Y}) \right]$$
   across neighborhood radii $\mathcal{K} = \{10\text{ km}, 20\text{ km}, 30\text{ km}, 50\text{ km}\}$ ($k \in \{1, 2, 3, 5\}$ grid cells).
4. **Empirical Hyperparameters:** $\lambda_{\text{reg}} = 1.0, \lambda_{\text{fss}} = 0.5$, calibrated to prevent gradient dominance over the classification head.

---

## CHAPTER 5: EXPERIMENTAL PROTOCOL AND QUANTITATIVE RESULTS

### 5.1 Dataset Partitions and Integrity Rules
- **Training Cohort:** 38 active monsoon days across 2023–2025, generating **1,764 continuous sequences** (July 1–10, 2023; July 1–14, 2024; August 5–14, 2024; July 1–4, 2025).
- **Validation Split:** Single heavy convective episode on **July 5, 2025** (33 sequences). Used strictly for epoch checkpoint selection (lowest validation loss).
- **Confirmatory Historical Benchmark:** Extreme monsoon depression event on **July 12–14, 2023** (129 sequences, 72.0 continuous hours). This split was locked and never accessed during training or loss optimization.

### 5.2 Three-Seed Replication Results (Validation Split)
A balanced $2 \times 3$ factorial experiment was conducted comparing Baseline Exp A (BCE + Smooth $L_1$) against Compound Loss Exp D across three independent random seeds (42, 123, 999). Metrics are evaluated at the convective threshold $\ge 5.0\text{ mm/hr}$ and reported as mean $\pm$ standard deviation across seeds:

| Horizon | Metric | Baseline (Exp A) | Combined (Exp D) | Absolute Gain ($\Delta$) | Relative Gain |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+30m** | Pointwise CSI | $0.3491 \pm 0.0263$ | $\mathbf{0.3819 \pm 0.0160}$ | $+0.0328$ | $+9.4\%$ |
| | Circular FSS ($30\text{ km}$) | $0.7583 \pm 0.0464$ | $\mathbf{0.7858 \pm 0.0121}$ | $+0.0275$ | $+3.6\%$ |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.1150 \pm 0.0119}$ | $1.1531 \pm 0.0485$ | $+0.0381$ | $+3.4\%$ |
| **+60m** | Pointwise CSI | $0.2094 \pm 0.0100$ | $\mathbf{0.2662 \pm 0.0227}$ | $+0.0568$ | $+27.1\%$ |
| | Circular FSS ($30\text{ km}$) | $0.5860 \pm 0.0348$ | $\mathbf{0.6756 \pm 0.0316}$ | $+0.0896$ | $+15.3\%$ |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.4404 \pm 0.0116}$ | $1.4927 \pm 0.0673$ | $+0.0523$ | $+3.6\%$ |
| **+120m** | **Pointwise CSI** | $0.0796 \pm 0.0300$ | $\mathbf{0.1685 \pm 0.0149}$ | $\mathbf{+0.0889}$ | **$+111.7\%$ ($2.12\times$)** |
| | **Circular FSS ($30\text{ km}$)** | $0.3232 \pm 0.1158$ | $\mathbf{0.5560 \pm 0.0307}$ | $\mathbf{+0.2328}$ | **$+72.0\%$** |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.7972 \pm 0.0385}$ | $1.8276 \pm 0.0257$ | $+0.0304$ | $+1.7\%$ |
| **+180m** | **Pointwise CSI** | $0.0183 \pm 0.0107$ | $\mathbf{0.1111 \pm 0.0184}$ | $\mathbf{+0.0928}$ | **$+507.1\%$ ($6.07\times$)** |
| | **Circular FSS ($30\text{ km}$)** | $0.0740 \pm 0.0535$ | $\mathbf{0.4016 \pm 0.0523}$ | $\mathbf{+0.3276}$ | **$+442.7\%$ ($5.43\times$)** |
| | **QPF RMSE ($\text{mm/hr}$)** | $2.0396 \pm 0.0450$ | $\mathbf{2.0145 \pm 0.0255}$ | $\mathbf{-0.0251}$ | **$-1.2\%$ (Better)** |
| **+360m** | Pointwise CSI | $0.0000 \pm 0.0000$ | $\mathbf{0.0037 \pm 0.0037}$ | $+0.0037$ | Non-zero restored |
| | Circular FSS ($30\text{ km}$) | $0.0000 \pm 0.0000$ | $\mathbf{0.0193 \pm 0.0172}$ | $+0.0193$ | Non-zero restored |
| | QPF RMSE ($\text{mm/hr}$) | $2.3842 \pm 0.0101$ | $\mathbf{2.3520 \pm 0.0144}$ | $-0.0322$ | **$-1.4\%$ (Better)** |

### 5.3 Confirmatory Historical Benchmark Results (July 12–14, 2023, $N=129$)
To verify generalization on an independent multi-day synoptic episode, frozen checkpoints were evaluated without retraining across 72 continuous observation hours:

| Horizon | Metric | Baseline (Exp A Mean) | Combined (Exp D Mean) | Mean Difference ($\bar{\Delta}$) | 95% Bootstrap CI ($L=12$) | Bounded $p$-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+120m** | Pointwise CSI | $0.1233 \pm 0.0413$ | $\mathbf{0.2133 \pm 0.0285}$ | $\mathbf{+0.0901}$ | $[+0.0633, +0.1139]$ | $\mathbf{p < 0.001}$ |
| | Circular FSS ($30\text{ km}$) | $0.4113 \pm 0.1009$ | $\mathbf{0.6000 \pm 0.0344}$ | $\mathbf{+0.1887}$ | $[+0.1340, +0.2526]$ | $\mathbf{p < 0.001}$ |
| | QPF RMSE ($\text{mm/hr}$) | $1.3757 \pm 0.0146$ | $\mathbf{1.3698 \pm 0.0155}$ | $-0.0059$ | $[-0.0398, +0.0289]$ | $p = 0.355$ |
| **+180m** | **Pointwise CSI** | $0.0120 \pm 0.0132$ | $\mathbf{0.1300 \pm 0.0231}$ | $\mathbf{+0.1180}$ | $[+0.0652, +0.1654]$ | $\mathbf{p < 0.001}$ |
| | **Circular FSS ($30\text{ km}$)** | $0.0527 \pm 0.0556$ | $\mathbf{0.4333 \pm 0.0424}$ | $\mathbf{+0.3807}$ | $[+0.2241, +0.4789]$ | $\mathbf{p < 0.001}$ |
| | **QPF RMSE ($\text{mm/hr}$)** | $1.5224 \pm 0.0122$ | $\mathbf{1.5216 \pm 0.0165}$ | $-0.0008$ | $[-0.0616, +0.0483]$ | $p = 0.417$ |
| **+360m** | Pointwise CSI | $0.0000 \pm 0.0000$ | $\mathbf{0.0125 \pm 0.0129}$ | $+0.0125$ | $[+0.0038, +0.0223]$ | $\mathbf{p = 0.002}$ |
| | Circular FSS ($30\text{ km}$) | $0.0000 \pm 0.0000$ | $\mathbf{0.0526 \pm 0.0495}$ | $+0.0526$ | $[+0.0126, +0.0984]$ | $\mathbf{p < 0.001}$ |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.6727 \pm 0.0056}$ | $1.6771 \pm 0.0475$ | $+0.0044$ | $[-0.0315, +0.0446]$ | $p = 0.584$ |

### 5.4 Multi-Scale Fractions Skill Score (10 km, 20 km, 30 km, 50 km)
Spatial verification over physical circular neighborhood radii ($k \in \{1, 2, 3, 5\}$ cells) at the $\ge 5.0\text{ mm/hr}$ convective threshold demonstrates consistent scale-selective improvements across both evaluation cohorts:

| Forecast Lead | Neighborhood Radius | Window Size | Exp A (Baseline) | Exp D (Combined Loss) | Absolute Gain ($\Delta$) | Theoretical Ref ($0.5 + f_0/2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+30m** | 10 km ($k=1$) | 5 cells | $0.6648 \pm 0.0394$ | $\mathbf{0.6968 \pm 0.0144}$ | $+0.0320$ | $\sim 0.52$ |
| | 20 km ($k=2$) | 13 cells | $0.7198 \pm 0.0434$ | $\mathbf{0.7497 \pm 0.0136}$ | $+0.0299$ | $\sim 0.52$ |
| | 30 km ($k=3$) | 29 cells | $0.7583 \pm 0.0462$ | $\mathbf{0.7858 \pm 0.0121}$ | $+0.0275$ | $\sim 0.52$ |
| | 50 km ($k=5$) | 81 cells | $0.8064 \pm 0.0510$ | $\mathbf{0.8323 \pm 0.0112}$ | $+0.0259$ | $\sim 0.52$ |
| **+60m** | 10 km ($k=1$) | 5 cells | $0.4669 \pm 0.0223$ | $\mathbf{0.5457 \pm 0.0303}$ | $+0.0788$ | $\sim 0.52$ |
| | 20 km ($k=2$) | 13 cells | $0.5303 \pm 0.0290$ | $\mathbf{0.6137 \pm 0.0320}$ | $+0.0834$ | $\sim 0.52$ |
| | 30 km ($k=3$) | 29 cells | $0.5860 \pm 0.0346$ | $\mathbf{0.6756 \pm 0.0317}$ | $+0.0896$ | $\sim 0.52$ |
| | 50 km ($k=5$) | 81 cells | $0.6592 \pm 0.0361$ | $\mathbf{0.7600 \pm 0.0290}$ | $+0.1008$ | $\sim 0.52$ |
| **+120m** | 10 km ($k=1$) | 5 cells | $0.2156 \pm 0.0772$ | $\mathbf{0.3989 \pm 0.0261}$ | $\mathbf{+0.1833}$ | $\sim 0.52$ |
| | 20 km ($k=2$) | 13 cells | $0.2665 \pm 0.0962$ | $\mathbf{0.4749 \pm 0.0292}$ | $\mathbf{+0.2084}$ | $\sim 0.52$ |
| | 30 km ($k=3$) | 29 cells | $0.3232 \pm 0.1156$ | $\mathbf{0.5560 \pm 0.0307}$ | $\mathbf{+0.2328}$ | $\sim 0.52$ |
| | 50 km ($k=5$) | 81 cells | $0.3953 \pm 0.1326$ | $\mathbf{0.6763 \pm 0.0308}$ | $\mathbf{+0.2810}$ | $\sim 0.52$ |
| **+180m** | 10 km ($k=1$) | 5 cells | $0.0515 \pm 0.0324$ | $\mathbf{0.2813 \pm 0.0393}$ | $\mathbf{+0.2298}$ | $\sim 0.52$ |
| | 20 km ($k=2$) | 13 cells | $0.0619 \pm 0.0417$ | $\mathbf{0.3387 \pm 0.0457}$ | $\mathbf{+0.2768}$ | $\sim 0.52$ |
| | 30 km ($k=3$) | 29 cells | $0.0740 \pm 0.0533$ | $\mathbf{0.4016 \pm 0.0522}$ | $\mathbf{+0.3276}$ | $\sim 0.52$ |
| | 50 km ($k=5$) | 81 cells | $0.0889 \pm 0.0655$ | $\mathbf{0.4902 \pm 0.0595}$ | $\mathbf{+0.4013}$ | $\sim 0.52$ |

---

## CHAPTER 6: FORENSIC INVESTIGATION OF CONVECTIVE CORE DISPLACEMENT

An essential component of spatial model verification is understanding storm positioning errors. The evaluation scripts reported an average peak convective storm core displacement of approximately **$109.9\text{ km}$** on validation and **$156.6\text{ km}$** on test at $+180\text{m}$. A forensic audit was conducted to analyze its physical and mathematical origin:

### 6.1 Algorithmic Formulation
The displacement diagnostic is calculated across matched frames using physical coordinate spacing:
$$d = \sqrt{\left((\hat{y}_{\text{max}} - y^*_{\text{max}}) \cdot 11.08\text{ km}\right)^2 + \left((\hat{x}_{\text{max}} - x^*_{\text{max}}) \cdot 10.03\text{ km}\right)^2}$$
- *Condition for Matched Frames:* A frame is eligible if $\max(\mathbf{Y}_{\text{obs}}) \ge 5.0\text{ mm/hr}$, and matched if $\max(\hat{\mathbf{Y}}_{\text{fcst}}) \ge 5.0\text{ mm/hr}$.
- *Single-Peak Nature:* The metric evaluates strictly the single global domain maximum coordinate (`argmax`). It is **not** an object-based multi-cell tracking algorithm (such as MODE or CRA).

### 6.2 Matched vs. Unmatched Event Breakdown Across Cohorts

| Split Cohort | Lead Time | Total Frames | Eligible Storms ($\ge 5\text{ mm/hr}$) | Exp A Matched | Exp D Matched | Exp A Miss Rate | Exp D Miss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | +30m | 33 | 26 | 24 | 24 | 7.7% | 7.7% |
| ($N=33$) | +60m | 33 | 27 | 24 | 24 | 11.1% | 11.1% |
| | +120m | 33 | 29 | 8 | **23** | 72.4% | **20.7%** |
| | **+180m** | 33 | 31 | **1** | **22** | **96.8%** | **29.0%** |
| | +360m | 33 | 33 | **0** | **10** | **100.0%** | **69.7%** |
| **Test Benchmark**| +30m | 129 | 117 | 107 | 112 | 8.5% | 4.3% |
| ($N=129$) | +60m | 129 | 117 | 94 | 102 | 19.7% | 12.8% |
| | +120m | 129 | 117 | 50 | **78** | 57.3% | **33.3%** |
| | **+180m** | 129 | 117 | **6** | **62** | **94.9%** | **47.0%** |
| | +360m | 129 | 116 | **0** | **35** | **100.0%** | **69.8%** |

### 6.3 Progression of Median Displacements Across Forecast Horizons
- **+30m:** On the test benchmark ($N=112$ matched frames), the median displacement is **$14.9\text{ km}$** ($1.3$ grid cells), with 25% of cases exhibiting zero offset ($0.0\text{ km}$).
- **+60m:** On the test benchmark ($N=102$ matched frames), the median displacement is **$27.2\text{ km}$** ($2.5$ grid cells).
- **+120m:** Median displacement increases to **$83.1\text{ km}$**.
- **+180m:** Median displacement is **$157.1\text{ km}$** on test ($73.3\text{ km}$ on validation).

### 6.3 Multi-Peak Ambiguity in Large Displacement Cases
An audit of all frames where the displacement offset exceeded $100\text{ km}$ revealed that **100% of these cases (9 of 9 on validation)** contained multiple distinct convective cores active simultaneously across the domain (between 4 and 22 observed cores and 1 to 9 forecast cores). When multiple storm clusters exist simultaneously across Meghalaya (e.g., western Garo Hills vs. eastern Khasi Hills, separated by $\approx 200\text{ km}$), slight fluctuations in predicted intensity between the clusters cause the single domain-wide `argmax` pixel to switch clusters, resulting in apparent jumps of $150\text{--}245\text{ km}$ even though both clusters are actively forecasted.

### 6.4 The Conditioning Bias in Baseline Exp A
Baseline Exp A predicted rainfall $\ge 5.0\text{ mm/hr}$ on only **1 sequence** out of 31 on validation (displacement = $237.2\text{ km}$) and only **6 sequences** on test ($94.9\%$ misses). Exp A does not achieve superior placement; it simply suffers severe intensity extinction. Exp D successfully forecasted convective cells across 22 validation frames and 62 test frames.

---

## CHAPTER 7: LANDSLIDE SUSCEPTIBILITY AND DYNAMIC HAZARD TIERS

### 7.1 Morphometric Conditioning Factors
From ALOS PALSAR 30 m DEM and geological maps, 12 spatial conditioning layers were derived across Meghalaya:
1. Slope Gradient ($^\circ$): Primary driving force for shear stress.
2. Aspect ($^\circ$): Controls solar exposure and prevailing monsoon windward orientation.
3. Terrain Ruggedness Index (TRI) and Topographic Wetness Index (TWI).
4. Stream Power Index (SPI) and Sediment Transport Index (STI).
5. Plan and Profile Curvature: Identify flow divergence and convergence zones.
6. Distance to Faults and Shear Zones (km): Captures tectonic fracturing.
7. Distance to Major Roads and Drainage Networks (km).

### 7.2 Spatial Block Cross-Validation and Model Performance
To eliminate spatial autocorrelation leakage, machine learning classifiers were evaluated using 5-fold spatial block cross-validation against the GSI Meghalaya historical inventory:
- **Random Forest:** AUC-ROC = $\mathbf{0.842 \pm 0.031}$, PR-AUC = $0.781 \pm 0.029$.
- **XGBoost:** AUC-ROC = $0.835 \pm 0.034$.
- **LightGBM:** AUC-ROC = $0.829 \pm 0.036$.

### 7.3 Dynamic Hazard Tier Matrix
Dynamic hazard warning levels are computed by combining static susceptibility ($S$) with antecedent rainfall ($A_{\text{eff}} = P_{24\text{h}} + 0.5 \cdot P_{72\text{h}} + 0.2 \cdot P_{7\text{d}}$):
- **Level 1 (Low / Green):** $S < 0.40$ or $A_{\text{eff}} < 25\text{ mm}$.
- **Level 2 (Moderate / Yellow):** $0.40 \le S < 0.70$ and $25\text{ mm} \le A_{\text{eff}} < 75\text{ mm}$.
- **Level 3 (High / Orange):** $S \ge 0.70$ and $50\text{ mm} \le A_{\text{eff}} < 125\text{ mm}$, or $0.40 \le S < 0.70$ and $A_{\text{eff}} \ge 75\text{ mm}$.
- **Level 4 (Severe / Red):** $S \ge 0.70$ and $A_{\text{eff}} \ge 125\text{ mm}$.

### 7.4 Explicit Downstream Coupling Boundary
The thesis strictly maintains that while SimVP + Exp D provides superior 2-hour and 3-hour rainfall fields, the downstream coupling between SimVP nowcasts and the Random Forest landslide classifier was **not re-benchmarked** during Step 5. No claims of improved landslide polygon classification accuracy or increased warning lead times are made.

---

## CHAPTER 8: EXPLAINABLE AI AND FEATURE ATTRIBUTION

To provide transparent decision support for emergency managers, the platform incorporates explainable AI (XAI) microservices:
1. **TreeSHAP for Landslide Susceptibility:** Confirms that slope gradient ($38.4\%$ mean absolute SHAP attribution), distance to drainage channels ($21.2\%$), and Topographic Wetness Index ($16.8\%$) dominate slope failure susceptibility across the Meghalaya escarpment.
2. **Integrated Gradients for Spatiotemporal Nowcasting:** Attributions computed over input channels reveal that thermal infrared brightness temperatures ($T_b$) dominate lead times beyond 60 minutes by capturing upper-tropospheric cloud expansion, while surface IMERG precipitation dominates immediate advection (+30m).

---

## CHAPTER 9: LIMITATIONS, OPEN GAPS, AND FUTURE WORK

To uphold the highest standards of scientific integrity, the following limitations are formally acknowledged:
1. **The 6-Hour Predictability Ceiling:** At $+360\text{m}$, satellite-based spatiotemporal deep learning achieves an FSS of $\approx 0.05$. Purely data-driven sequence modeling cannot resolve chaotic convective thunderstorm evolution 6 hours ahead without numerical weather prediction (NWP) physics coupling.
2. **Validation Cohort Limitation:** Checkpoints were selected using a single severe monsoon day (July 5, 2025, 33 sequences). Expanding validation to multi-month seasons is recommended for future operational deployments.
3. **Historical Benchmark Status:** The July 12–14, 2023 benchmark, while locked during training, was evaluated in earlier project phases and chronologically precedes the 2024–2025 training samples.
4. **Orographic Static Inputs:** While the landslide engine ingests high-resolution DEM covariates, the weather nowcasting model operates on satellite channels ($T_b, P, M$) without explicit DEM terrain elevation channels. Future work will investigate orographic embedding layers.
5. **Coupled System Validation:** End-to-end prospective evaluation linking deep learning weather nowcasts directly into dynamic landslide polygon hazard activation remains a vital direction for field deployment.

---

## CHAPTER 10: CONCLUSION

This B.Tech Honours Major Project successfully designed, engineered, and scientifically audited a comprehensive multimodal AI platform for convective nowcasting and landslide hazard prediction over Meghalaya, Northeast India. Through rigorous three-seed replication and stationary temporal block bootstrapping, we demonstrated that a compound spatial loss objective (Intensity-Weighted L1 + Multiscale Differentiable Soft-FSS + BCE) substantially mitigates convective intensity attenuation in spatiotemporal deep learning models. At lead times of 2 to 3 hours, the model achieves a $6\times$ to $10\times$ improvement in convective Critical Success Index and an $8\times$ improvement in neighborhood Fractions Skill Score, with no statistically significant increase in domain-wide bulk error. Integrated with an operational GIS dashboard and explainable AI services, this research establishes an auditable, reproducible baseline for AI-driven hydrometeorological disaster mitigation in complex orographic terrain.

---

## REFERENCES

1. Badavath, S., & Sahoo, S. (2025). Landslide susceptibility mapping in complex orographic terrain of Meghalaya using machine learning and spatial conditioning factors. *Natural Hazards*, 126(2), 1421–1448.
2. Gao, Z., Tan, C., Wu, L., & Li, S. Z. (2022). SimVP: Simpler yet better video prediction. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3170–3180.
3. Jash, D., Ghosh, S., & Sen, A. (2025). Thunderstorm evolution nowcasting using Doppler weather radar and spatiotemporal deep learning in eastern India. *Atmospheric Research*, 302, 107291.
4. Kassoumeh, R., Wang, H., & Liu, Y. (2025). Multimodal fusion of geostationary satellite infrared and radar observations for convective precipitation nowcasting. *IEEE Transactions on Geoscience and Remote Sensing*, 63, 1–14.
5. Mihu, T., Sharma, R., & Das, P. (2026). Machine learning-based landslide susceptibility assessment and SHAP explainability in the Dibang Valley, Eastern Himalayas. *Georisk: Assessment and Management of Risk for Engineered Systems and Geohazards*, 20(1), 89–107.
6. Mittermaier, M. P. (2014). A comparison of Fractions Skill Score and spatial displacement metrics for high-resolution precipitation forecasts. *Monthly Weather Review*, 142(7), 2530–2546.
7. Rani, K., Kumar, A., & Singh, R. (2026). Transformer-based spatiotemporal modeling of rainfall sequences for regional landslide hazard early warning. *Computers & Geosciences*, 185, 105542.
8. Roberts, N. M., & Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from high-resolution WRF simulations. *Monthly Weather Review*, 136(1), 78–97.
9. Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *Advances in Neural Information Processing Systems (NeurIPS)*, 28, 802–810.
10. Skok, G. (2015). Analysis of Fractions Skill Score properties and sensitivity to domain size and precipitation coverage. *Atmospheric Research*, 164, 218–226.
