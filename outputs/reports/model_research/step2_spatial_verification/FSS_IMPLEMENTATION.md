# Fractions Skill Score (FSS) & Spatial Neighborhood Verification Engine: Implementation & Mathematical Formulation

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Module:** `src/evaluation/spatial_metrics.py`  
**Unit Tests:** `tests/test_spatial_metrics.py` (8/8 automated test suites passing)

---

## 1. Scientific Motivation: The Double-Penalty Problem in Complex Orography

Traditional point-by-point verification metrics (such as Critical Success Index / Threat Score, Probability of Detection, and False Alarm Ratio computed on a strict $0.1^\circ \approx 10\text{ km}$ grid) impose an severe double penalty on high-resolution convective forecasts:
1. A **miss penalty** (false negative) at the exact grid cell where convection actually occurred.
2. A **false alarm penalty** (false positive) at the grid cell where the model predicted convection.

In topographically extreme terrain like the southern slopes of Meghalaya (Cherrapunji–Mawsynram scarp rising abruptly from the Sylhet plains of Bangladesh), orographic convective cells frequently trigger with localized spatial displacements of $5\text{--}30\text{ km}$. A model that accurately predicts intense convective rainfall $15\text{ km}$ east of the observed centroid receives a **pointwise CSI of 0.0000**—statistically indistinguishable from a model that predicted complete dryness or placed rainfall $500\text{ km}$ away.

To resolve this limitation, we implemented a rigorous spatial verification framework based on:
1. **Fractions Skill Score (FSS)** following Roberts and Lean (2008) and Mittermaier (2014).
2. **Fuzzy Neighborhood Contingency Metrics** following Ebert (2008).

---

## 2. Mathematical Formulation of Fractions Skill Score (FSS)

### 2.1 Binary Event Thresholding
Let $O(x, y)$ be the observed precipitation rate field ($\text{mm/hr}$) and $M(x, y)$ be the model forecast field ($\text{mm/hr}$) over the spatial grid of size $H \times W = 25 \times 40$.
For a chosen precipitation threshold $q \in \{0.1, 1.0, 5.0\}\text{ mm/hr}$:

$$
I_O(x, y) = \begin{cases} 1 & \text{if } O(x, y) \ge q \\ 0 & \text{otherwise} \end{cases}
$$

$$
I_M(x, y) = \begin{cases} 1 & \text{if } M(x, y) \ge q \\ 0 & \text{otherwise} \end{cases}
$$

### 2.2 Neighborhood Fractional Fields
For a neighborhood radius of $k$ grid cells (corresponding to a square window of size $n = 2k + 1$ by $n = 2k + 1$ cells), the fraction of grid cells exceeding the threshold within the neighborhood surrounding $(x, y)$ is:

$$
O_k(x, y) = \frac{1}{\sum_{(u, v) \in \mathcal{N}_k(x, y)} W(u, v)} \sum_{(u, v) \in \mathcal{N}_k(x, y)} I_O(u, v) \cdot W(u, v)
$$

$$
M_k(x, y) = \frac{1}{\sum_{(u, v) \in \mathcal{N}_k(x, y)} W(u, v)} \sum_{(u, v) \in \mathcal{N}_k(x, y)} I_M(u, v) \cdot W(u, v)
$$

where $\mathcal{N}_k(x, y) = \{ (u, v) : |u - x| \le k, |v - y| \le k \}$ and $W(u, v) \in \{0, 1\}$ is the observation validity mask.

### 2.3 Fractions Brier Score (FBS)
The mean squared difference between forecast and observed fractional coverages across the domain of $N_V$ valid grid cells is:

$$
FBS = \frac{1}{N_V} \sum_{(x, y) \in \mathcal{D}_{valid}} \left[ O_k(x, y) - M_k(x, y) \right]^2
$$

The worst-case Fractions Brier Score ($FBS_{worst}$) represents the largest possible error given the observed and forecast event frequencies, occurring when there is zero spatial overlap between forecast and observed fractional areas:

$$
FBS_{worst} = \frac{1}{N_V} \sum_{(x, y) \in \mathcal{D}_{valid}} \left[ O_k(x, y)^2 + M_k(x, y)^2 \right]
$$

### 2.4 Fractions Skill Score (FSS)
The Fractions Skill Score is defined as:

$$
FSS = 1 - \frac{FBS}{FBS_{worst}} = 1 - \frac{\sum \left[ O_k(x, y) - M_k(x, y) \right]^2}{\sum \left[ O_k(x, y)^2 + M_k(x, y)^2 \right]}
$$

Properties:
- $FSS = 1.0$: Perfect spatial agreement at scale $k$.
- $FSS = 0.0$: Complete spatial mismatch (no overlap).
- Useful forecast skill threshold (Roberts & Lean 2008):
  $$FSS_{useful} = 0.5 + \frac{f_O}{2}$$
  where $f_O$ is the base rate (domain-wide observed fraction). For rare convective events ($f_O \ll 0.1$), $FSS_{useful} \approx 0.50$.
- Random forecast baseline:
  $$FSS_{random} = f_O$$

---

## 3. Accumulated Cohort FSS vs. Arithmetic Frame-Mean FSS

When evaluating across a sequence of $T$ validation/test time steps, two aggregation methods exist:

1. **Accumulated FSS (Roberts & Lean 2008; Mittermaier 2014) [Canonical Benchmark]**:
   $$FSS_{accumulated} = 1 - \frac{\sum_{t=1}^T \sum_{(x, y)} \left[ O_{k, t}(x, y) - M_{k, t}(x, y) \right]^2}{\sum_{t=1}^T \sum_{(x, y)} \left[ O_{k, t}(x, y)^2 + M_{k, t}(x, y)^2 \right]}$$
   *Advantage*: Robust against zero-event or low-event individual time frames. Highly stable and mathematically consistent with total Brier score decomposition.

2. **Frame-Mean FSS**:
   $$FSS_{mean} = \frac{1}{T_{active}} \sum_{t \in \mathcal{T}_{active}} FSS_t$$
   computed only over time steps where at least one observation or forecast event is active.

Both metrics are computed and exported by `SpatialVerificationEngine.evaluate_sequence_dataset()`.

---

## 4. Grid Geometry and Scale Mapping for Meghalaya

The domain spans:
- Latitude: $24.55^\circ\text{N}$ to $26.95^\circ\text{N}$ ($25$ cells, spacing $\Delta \phi = 0.100^\circ$)
- Longitude: $89.55^\circ\text{E}$ to $93.45^\circ\text{E}$ ($40$ cells, spacing $\Delta \lambda = 0.100^\circ$)

Using the WGS-84 reference ellipsoid:
- Meridional distance per $0.1^\circ$: $\Delta y = 111.132 \times 0.1 = 11.08\text{ km}$.
- Zonal distance per $0.1^\circ$ at domain center ($25.75^\circ\text{N}$):
  $$\Delta x = 111.413 \times \cos(25.75^\circ) \times 0.1 = 10.03\text{ km}$$
- Mean effective cell dimension:
  $$\Delta s = \frac{11.08 + 10.03}{2} = 10.55\text{ km}$$

### Neighborhood Radius to Window Size Mapping
| Nominal Physical Radius | Grid Radius $k$ (cells) | Window Dimension ($n \times n$) | Effective Scale ($L = (2k+1) \times 10.55\text{ km}$) |
| :--- | :--- | :--- | :--- |
| **$10\text{ km}$** | $k = 1$ | $3 \times 3$ cells | $31.6\text{ km}$ |
| **$20\text{ km}$** | $k = 2$ | $5 \times 5$ cells | $52.8\text{ km}$ |
| **$30\text{ km}$** | $k = 3$ | $7 \times 7$ cells | $73.9\text{ km}$ |
| **$50\text{ km}$** | $k = 5$ | $11 \times 11$ cells | $116.1\text{ km}$ |

---

## 5. Boundary Condition, Mask, and Zero-Denominator Safeguards

1. **Boundary / Edge Handling**:
   Convolutions are computed with boundary normalization:
   $$O_k(x, y) = \frac{(I_O \cdot W) * K(k)}{W * K(k)}$$
   where $K(k)$ is the spatial neighborhood kernel and $*$ denotes 2D convolution. Grid cells where fewer neighbor cells exist (near the border) are divided strictly by the number of valid neighbors within the kernel, avoiding edge attenuation artifacts.

2. **Observation Validity Mask ($W$)**:
   Missing satellite data or out-of-boundary pixels ($W(x, y) = 0$) are excluded from both numerator and denominator. If a grid cell has zero valid neighbors, its fraction is assigned $0.0$ and excluded from $N_V$.

3. **Zero-Denominator Safeguards**:
   - Case 1: Both observed and predicted fields have zero events ($FBS = 0$ and $FBS_{worst} = 0$). Handled as **$FSS = 1.0$** (perfect agreement on null occurrence).
   - Case 2: One field has events and the other has zero ($FBS_{worst} > 0$ and $FBS = FBS_{worst}$). Handled as **$FSS = 0.0$** (complete lack of overlap).
   - Case 3: In `compute_neighborhood_contingency`, if $TP + FP + FN = 0$, $N\text{-CSI} = 1.0$ if no events existed anywhere, or $0.0$ if defined on events.

---

## 6. Fuzzy Neighborhood Contingency (Ebert 2008)

To complement FSS with traditional contingency table semantics, morphological binary dilation with kernel $K(k)$ is evaluated:
- **Neighborhood Hit (TP)**: Observed event matched by predicted event within radius $k$, OR predicted event matched by observed event within radius $k$.
  $$TP_{neigh} = \sum \left[ (I_O \land \mathcal{D}_k(I_M)) \lor (I_M \land \mathcal{D}_k(I_O)) \right]$$
- **Neighborhood False Alarm (FP)**: Predicted event with no observed event within radius $k$.
  $$FP_{neigh} = \sum \left[ I_M \land \neg \mathcal{D}_k(I_O) \right]$$
- **Neighborhood Miss (FN)**: Observed event with no predicted event within radius $k$.
  $$FN_{neigh} = \sum \left[ I_O \land \neg \mathcal{D}_k(I_M) \right]$$
- **Neighborhood CSI ($N\text{-CSI}$)**:
  $$N\text{-CSI} = \frac{TP_{neigh}}{TP_{neigh} + FP_{neigh} + FN_{neigh}}$$
- **Neighborhood POD ($N\text{-POD}$)**:
  $$N\text{-POD} = \frac{TP_{neigh}}{TP_{neigh} + FN_{neigh}}$$
- **Neighborhood FAR ($N\text{-FAR}$)**:
  $$N\text{-FAR} = \frac{FP_{neigh}}{TP_{neigh} + FP_{neigh}}$$
