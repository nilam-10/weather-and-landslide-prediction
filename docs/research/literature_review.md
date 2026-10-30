# Literature Review & Scientific Foundations

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Institutional Context**: RBU Honours Major Project (2026–27)  
**Study Domain**: Meghalaya, Northeast India (Shillong Plateau)

---

## 1. Convective Weather Nowcasting: From Kinematic Extrapolation to Spatiotemporal Deep Learning

### 1.1 Meteorological Context of the Shillong Plateau
The state of Meghalaya is globally recognized for extreme monsoon precipitation regimes. The southern escarpment of the Shillong Plateau rises abruptly from the low-lying Bengal floodplains (elevation $< 20\text{ m}$) to altitudes exceeding $1,900\text{ m}$ over a horizontal distance of under $40\text{ km}$. During the Indian Summer Monsoon (June–September), southwesterly low-level jets laden with moisture from the Bay of Bengal encounter this physical barrier, producing intense orographic convective lifting. This mechanism sustains world-record precipitation rates at stations such as Cherrapunji (Sohra) and Mawsynram, frequently generating localized rainfall intensities exceeding $50\text{ mm/hr}$ and multi-day accumulations exceeding $500\text{ mm}$.

Accurate short-term forecasting (0 to 3 hours, termed *nowcasting*) in this terrain is critical for flash-flood and slope failure warning. However, operational Doppler Weather Radar (DWR) coverage in Northeast India suffers from severe beam blockage caused by rugged terrain, as well as latency and accessibility constraints. Consequently, spaceborne observations—specifically geostationary thermal infrared (TIR) brightness temperatures ($T_b$) and low-Earth-orbit passive microwave (PMW) precipitation retrievals—represent the primary continuous observation stream for convective monitoring over the region.

### 1.2 Kinematic Extrapolation and Optical Flow
Precipitation nowcasting historically originated with radar echo extrapolation using cross-correlation techniques (e.g., TREC, COTREC) and semi-Lagrangian advection (Germann & Zawadzki, 2002). Optical flow algorithms compute apparent velocity vector fields $(u, v)$ between consecutive raster frames under the brightness constancy assumption:
$$\frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t} = 0$$
Ayzel et al. (2019) formalized these methods in the open-source `rainymotion` library, benchmarking dense optical flow formulations (Lucas-Kanade, Gunnar Farneback, and DIS) for radar extrapolation. 

While optical flow provides a computationally efficient kinematic benchmark, it suffers from two fundamental physical limitations:
1. **Zero Growth and Decay**: Kinematic advection preserves the shape and intensity of precipitation cells, entirely unable to model convective initiation, intensification, or dissipation.
2. **Lagrangian Dissipation**: At lead times beyond 60 minutes, advection errors compound rapidly, and the true predictability of convective precipitation drops precipitously.

### 1.3 Spatiotemporal Deep Learning: ConvLSTM
To capture non-linear convective growth and decay dynamics, Shi et al. (2015) introduced the **Convolutional Long Short-Term Memory (ConvLSTM)** network. Traditional fully connected LSTMs (Hochreiter & Schmidhuber, 1997) collapse spatial topology into 1D feature vectors, discarding crucial spatial correlation. ConvLSTM replaces matrix multiplication inside the recurrent cell with spatial convolution operations ($\ast$), thereby maintaining spatial structure across recurrent state transitions:
$$\begin{aligned}
i_t &= \sigma(W_{xi} \ast X_t + W_{hi} \ast H_{t-1} + W_{ci} \odot C_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} \ast X_t + W_{hf} \ast H_{t-1} + W_{cf} \odot C_{t-1} + b_f) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tanh(W_{xc} \ast X_t + W_{hc} \ast H_{t-1} + b_c) \\
o_t &= \sigma(W_{xo} \ast X_t + W_{ho} \ast H_{t-1} + W_{co} \odot C_t + b_o) \\
H_t &= o_t \odot \tanh(C_t)
\end{aligned}$$
where $\ast$ denotes 2D convolution and $\odot$ represents the Hadamard (element-wise) product.

In our multimodal framework, the ConvLSTM ingests sequences of stacked multispectral satellite observations:
- $X_t \in \mathbb{R}^{C \times H \times W}$, where channels $C$ combine MERGIR Cloud-Top Infrared Brightness Temperature ($T_b$) and IMERG Half-Hourly Precipitation Rate ($P$).
- Deep convolutional filters learn to identify cold cloud-top brightness temperatures ($T_b < 220\text{ K}$) associated with deep convective cloud towers and correlate them with localized precipitation cores.
- Dual prediction heads output probability rasters at $+30\text{m}$ and $+60\text{m}$ horizons.

### 1.4 Satellite Proxy Truth vs. Radar Truth
A vital scientific distinction maintained in this project: NASA GPM IMERG (Huffman et al., 2020) is an algorithmically merged satellite precipitation product ($0.1^\circ$, half-hourly) combining passive microwave radiometers, calibrated geostationary infrared observations, and monthly rain gauge analyses. IMERG is **not** ground truth rain-gauge data or high-resolution radar reflectivity; it serves as a scientifically documented *satellite proxy target*. Predictions evaluated against IMERG represent satellite-retrieval threshold exceedance rather than direct in-situ convective storm observations.

---

## 2. Landslide Hazard Assessment: Static Susceptibility vs. Dynamic Triggering

### 2.1 Physiography and Geomorphic Instability of Meghalaya
The Shillong Plateau is an uplifted horst block bounded by active tectonic faults: the Dauki Fault to the south, the Oldham Fault to the north, and the Dapsi Thrust to the west (Bilham & England, 2001). The geomorphology consists of deeply incised river valleys, steep gorges, weathered Precambrian granitic gneisses, and Tertiary sedimentary strata (sandstones, shales, and coal beds). 

Landslides in this region are predominantly debris slides, mudflows, and rockfalls triggered during intense monsoon spells. Steep structural escarpments with slopes exceeding $25^\circ$, combined with deep residual soil mantles and anthropogenic slope undercutting (e.g., NH-40 and NH-44 highway construction, open-cast rat-hole coal mining), render slopes highly susceptible to shear failure.

### 2.2 Landslide Susceptibility Modeling (LSM)
As synthesized in the meta-analysis by Reichenbach et al. (2018), Landslide Susceptibility Mapping (LSM) models the spatial likelihood of slope failure conditioned solely on static environmental and geomorphometric factors:
1. **Elevation**: Topographic gradient and orographic exposure.
2. **Slope**: Gravitational shear stress component; the single most influential predictor of slope stability.
3. **Aspect / Northness / Eastness**: Solar radiation, microclimate, and moisture retention. Aspect $\theta$ is decomposed trigonometrically into continuous bounded variables: $\text{northness} = \cos(\theta)$ and $\text{eastness} = \sin(\theta)$ to prevent false discontinuities at $0^\circ \equiv 360^\circ$.
4. **Terrain Curvature**: Convergence and divergence of surface and subsurface hydrology. Profile curvature governs acceleration of mass flow; planform curvature governs water convergence.

Machine learning algorithms—notably Logistic Regression and Gradient Boosted Decision Trees (Chen & Guestrin, 2016 XGBoost)—have superseded subjective heuristic weighting (Analytical Hierarchy Process). XGBoost handles non-linear feature interactions (e.g., high elevation combined with steep slope and concave curvature) without requiring rigid parametric assumptions.

### 2.3 Spatial Autocorrelation and Spatial Block Cross-Validation
Standard random $k$-fold cross-validation in spatial modeling results in severe performance overestimation due to spatial autocorrelation (Tobler's First Law of Geography; Roberts et al., 2017). Randomly splitting grid cells places training and testing pixels into immediately adjacent valleys or ridge slopes, causing models to memorize geographic coordinates rather than learning physical geomorphic relationships.

To ensure strict scientific validity, our evaluation protocol implements **Spatial Block Cross-Validation**, partitioning Meghalaya into three discrete geographic sectors with distinct geomorphic characteristics:
- **Block 1 (West Garo Hills)**: Lower elevation ($< 600\text{ m}$), dissected ridges, sub-tropical weathering.
- **Block 2 (Central Khasi Hills)**: High-altitude plateau ($1,000–1,900\text{ m}$), steep southern escarpment, extreme orographic exposure.
- **Block 3 (East Jaintia Hills)**: Moderate-to-high plateau, coal-bearing sedimentary formations, heavy mining disturbance.

Models are trained on two sectors and evaluated strictly on the unseen held-out third sector, providing an authentic test of spatial generalization.

### 2.4 Dynamic Rainfall Triggers: NASA LHASA Framework
While static susceptibility identifies *where* landslides are geomorphologically possible, it cannot determine *when* they will occur. Slope failure initiation requires a dynamic hydrologic trigger that elevates pore-water pressure, reducing effective shear strength ($\tau = c' + (\sigma - u) \tan\phi'$).

Pioneering work by Caine (1980) and Guzzetti et al. (2007) established empirical Intensity-Duration ($I-D$) power-law thresholds:
$$I = \alpha \cdot D^{-\beta}$$
At regional to global scales, the **NASA Landslide Hazard Assessment for Situational Awareness (LHASA 2.0)** framework (Stanley et al., 2021) couples static susceptibility with antecedent precipitation indices:
- Short-term antecedent rainfall ($P_{24\text{h}}$): Captures immediate pore-water saturation and positive pore-pressure generation.
- Long-term antecedent rainfall ($P_{72\text{h}}, P_{7\text{d}}$): Accounts for progressive soil moisture accumulation and water table rise.

---

## 3. Multimodal Integration Architecture: Why "Joint Probability" Is Scientifically Invalid

### 3.1 The Fallacy of Multiplying Uncalibrated Probabilities
A frequent error in amateur disaster risk platforms is calculating an arbitrary "joint probability" of disaster by multiplying the weather nowcast probability with the landslide susceptibility score:
$$P(\text{Joint Disaster}) \neq P(\text{Convection}) \times P(\text{Susceptibility}) \quad \text{[SCIENTIFICALLY INVALID]}$$
This formula is strictly invalid for three mathematical and physical reasons:
1. **Non-Independence**: Landslide triggering is causally conditional on cumulative antecedent hydrologic infiltration over days to weeks, not on the instantaneous occurrence of an individual convective cloud cell passing overhead at minute $t$.
2. **Dimension and Scale Mismatch**: Weather nowcasting operates on a $0.1^\circ$ ($\sim 11\text{ km}$) grid with 30-minute updates; terrain susceptibility operates on a $90\text{ m}$ or $30\text{ m}$ static grid. Multiplying rasters across disparate spatial and temporal dimensions without physical routing models introduces gross scale distortion.
3. **Probabilistic Non-Equivalence**: Susceptibility $S(x, y) \in [0, 1]$ is a relative spatial propensity score, **not** an annualized or diurnal marginal probability of failure $P(Y=1 \mid \text{day})$. Multiplying a relative spatial index by a temporal weather forecast yields a dimensionally meaningless number that cannot be calibrated against real-world observations.

### 3.2 The Scientifically Defensible Solution: Independent Dual-Layer Presentation
Following the core design philosophy of our RBU Honours specification:
- **Layer A (Weather Nowcast)**: Real-time $+30\text{m}$ and $+60\text{m}$ convective rainfall probability rasters ($0.1^\circ$) derived from ConvLSTM / Optical Flow.
- **Layer B (Landslide Hazard)**:
  - *Static Sub-layer*: High-resolution ($90\text{ m}$) Terrain Susceptibility Index (LSI) derived from spatial-block-validated XGBoost.
  - *Dynamic Sub-layer*: Dynamic Rainfall Hazard Level (LHASA 2.0 dual-matrix lookup) driven by multi-day antecedent IMERG precipitation accumulations ($P_{24\text{h}}, P_{72\text{h}}$).
- **Boolean Co-location Overlay**: An explicit, transparent spatial filter that highlights areas where *both* (a) Static Landslide Susceptibility is High/Very High ($LSI > 0.6$) and (b) Imminent Convective Precipitation Nowcast crosses threshold ($P > 0.5$). 
  - This overlay is explicitly labeled in the GIS UI: **"Spatial Co-location Indicator (Research Only) — Not a Joint Probability or Official Warning."**

---

## 4. Summary of Research Baseline Models

| Model | Domain | Target Variable | Input Features | Spatial Grid | Temporal Resolution | Validation Strategy |
|---|---|---|---|---|---|---|
| **Lagged Persistence** | Weather | Precipitation Exceedance ($\ge 5\text{ mm/hr}$) | Previous interval IMERG | $0.1^\circ$ ($25 \times 40$) | $+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}$ | Held-out 2-day chronological test (July 6–7, 2025) |
| **Optical Flow** | Weather | Precipitation Exceedance ($\ge 5\text{ mm/hr}$) | Sequence of IMERG fields | $0.1^\circ$ ($25 \times 40$) | $+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}$ | Held-out 2-day chronological test (July 6–7, 2025) |
| **Logistic Regression** | Weather | Precipitation Exceedance ($\ge 5\text{ mm/hr}$) | MERGIR $T_b$ (mean, min, std) + lag IMERG | $0.1^\circ$ ($25 \times 40$) | $+30\text{m}, +60\text{m}$ | Held-out 2-day chronological test (July 6–7, 2025) |
| **PyTorch MLP** | Weather | Precipitation Exceedance ($\ge 5\text{ mm/hr}$) | MERGIR $T_b$ stats + IMERG rain | $0.1^\circ$ ($25 \times 40$) | $+30\text{m}, +60\text{m}$ | Chronological train/val/test with early stopping |
| **2D ConvLSTM** | Weather | Precipitation Exceedance ($\ge 5\text{ mm/hr}$) | 4-step sequence $\times$ 2 channels ($T_b, P$) | $0.1^\circ$ ($25 \times 40$) | $+30\text{m}, +60\text{m}$ | Spatiotemporal sliding windows with temporal holdout |
| **Static Logistic Regression** | Landslides | Slope Failure Susceptibility (0/1) | SRTM 90m terrain features | $90\text{ m}$ ($3075 \times 4897$) | Static | 3-Fold Spatial Block Cross-Validation |
| **Static XGBoost** | Landslides | Slope Failure Susceptibility (0/1) | SRTM 90m terrain features | $90\text{ m}$ ($3075 \times 4897$) | Static | 3-Fold Spatial Block Cross-Validation |
| **NASA LHASA 2.0 Matrix** | Landslides | Dynamic Landslide Hazard Index (0–3) | Static LSI + IMERG $P_{24\text{h}}, P_{72\text{h}}$ | $90\text{ m} \times 0.1^\circ$ | Multi-day antecedent | Rule-based empirical matrix |

---

## 5. References
Refer to `docs/research/verified_references.bib` for complete bibliographic citations.
