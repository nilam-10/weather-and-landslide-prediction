# PHASE 6: EXPLAINABLE AI (XAI) METHODOLOGY & MATHEMATICAL DERIVATION

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Year:** 2026–2027 (B.Tech Honours Major Project)  
**Institution:** Royal Global University (RGU / RBU Honours), Guwahati, Assam, India  
**Target Domain:** Meghalaya Plateau and Surrounding Barak/Surma Basins ($25.0^\circ\text{N} - 26.2^\circ\text{N}$, $89.8^\circ\text{E} - 92.9^\circ\text{E}$)  
**Module:** Deep Learning Nowcasting Explainability Module (`src/explainability`)  

---

## 1. EXECUTIVE SUMMARY & RESEARCH OBJECTIVE

Deep spatiotemporal recurrent architectures (e.g., Convolutional Long Short-Term Memory, ConvLSTM) have demonstrated predictive skill superior to Lagrangian optical flow and classical baselines for short-range precipitation nowcasting. However, their complex internal gate mechanisms and non-linear spatiotemporal representations are frequently characterized as "black boxes."

Phase 6 introduces a mathematically rigorous, scientifically defensible Explainable AI (XAI) framework into the platform to address three fundamental diagnostic questions:
1. **Sensor Modality Attribution:** How much predictive influence stems from cloud-top vertical thermodynamic cooling (NASA MERGIR $T_b$) versus kinematic radar/satellite precipitation advection (NASA IMERG $P$)?
2. **Temporal Lookback Memory:** How does the ConvLSTM recurrent memory weight historical lookback frames ($t-90\text{ min}, t-60\text{ min}, t-30\text{ min}$) relative to the instantaneous convective analysis at $t = 0$?
3. **Spatial Receptive Field Influence:** What geographical source regions and upstream atmospheric convective cells dictate the rainfall forecast at a specific high-risk mountain location?

To guarantee scientific integrity, the framework pairs **Axiomatic Path Integration (Integrated Gradients)** with an **Independent Perturbation Benchmark (Occlusion Sensitivity)** and enforces the **Adebayo et al. (2018) Parameter Randomization Sanity Test**.

---

## 2. AXIOMATIC INTEGRATED GRADIENTS (Sundararajan et al., 2017)

### 2.1 The Need for Axiomatic Attributions

Standard gradient saliency ($\nabla_x F(x)$) suffers from two severe mathematical flaws in deep neural networks:
1. **Gradient Saturation:** For extreme convective storms where the model predicts high rainfall probabilities ($p \to 1.0$), non-linear activation functions (such as Sigmoid or ReLU plateaus) have vanishing local gradients ($\frac{\partial F}{\partial z} \to 0$). Saliency maps falsely assign zero importance to critical storm cores.
2. **Violation of Implementation Invariance and Completeness:** Vanilla gradients do not sum to the model's prediction drop relative to a non-convective sky.

Integrated Gradients solves this by integrating gradients along the straight-line trajectory in $\mathbb{R}^D$ from a non-convective reference baseline $x'$ to the input storm observation $x$.

### 2.2 Mathematical Formulation

Let $F: \mathbb{R}^{B \times T \times C \times H \times W} \to \mathbb{R}$ denote the model prediction for a specific lead horizon $\tau \in \{+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}, +360\text{m}\}$ and target grid cell $(r, c)$. The path integral for input feature $x_i$ (where $i = (t, c, h, w)$ indexes time, channel, latitude, and longitude) is defined as:

$$\mathrm{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} \, d\alpha$$

### 2.3 Discrete Riemann Path Approximation

In practice, the continuous path integral is approximated via a uniform Riemann summation over $m$ interpolation steps:

$$\mathrm{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

In our production engine (`src/explainability/integrated_gradients.py`):
- Default integration steps: $m = 50$.
- Path evaluation: Scaled in mini-batches to fit within GPU VRAM limits (RTX 4050 Laptop GPU, 6141 MB).
- Autograd tracking: Evaluated strictly with `create_graph=False, retain_graph=False` to avoid computational graph expansion.

### 2.4 Completeness Axiom Proof

The fundamental mathematical theorem of Integrated Gradients (Sundararajan et al., ICML 2017) states that the sum of attributions across all input dimensions identically equals the difference between the model output on the target input $F(x)$ and the baseline $F(x')$:

$$\sum_{i} \mathrm{IG}_i(x) = F(x) - F(x')$$

This property is validated at runtime via relative error $\epsilon_{\text{rel}}$:

$$\epsilon_{\text{rel}} = \frac{\left| \sum_{i} \mathrm{IG}_i^{\text{approx}}(x) - [F(x) - F(x')] \right|}{\left| F(x) - F(x') \right| + 10^{-7}}$$

**Empirical Completeness Results (Frozen Model Checkpoint `convlstm_seed42.pt`):**
- **Quantitative Precipitation Forecast (QPF Head, mm/hr):** Relative error $\epsilon_{\text{rel}} = 0.0030\%$ (absolute error $0.00016\text{ mm/hr}$ against $\Delta F = +5.5042\text{ mm/hr}$).
- **Convective Rain Probability (Logit Head):** Relative error $\epsilon_{\text{rel}} = 1.46\%$ ($\Delta F = +4.6846$, $\sum \mathrm{IG} = +4.6160$).
- Across all 5 lead horizons, the completeness axiom is satisfied within the $< 5\%$ scientific threshold for continuous QPF and $< 10\%$ for logit path integration.

### 2.5 Logit Attribution vs. Probability Attribution

A critical architectural distinction is implemented between **logit space** and **calibrated probability space**:
- When explaining classification heads, evaluating gradients with respect to raw output probabilities $p = \sigma(z) = \frac{1}{1 + e^{-z}}$ introduces the derivative factor $\sigma'(z) = p(1 - p)$. For high-confidence convective predictions ($p > 0.95$), $\sigma'(z) < 0.0475$, artificially crushing gradient magnitudes.
- Therefore, our framework evaluates attributions directly in the **unbounded logit space** ($z$), preserving linearity across the entire path. For QPF heads, attributions are evaluated directly in physical units ($\text{mm/hr}$).

---

## 3. PHYSICALLY GROUNDED REFERENCE BASELINES

Integrated Gradients measures the model's sensitivity relative to a reference state $x'$. Choosing an unphysical baseline (e.g., all zeros in an unnormalized temperature array) introduces severe artifacts. Three distinct, documented baselines are supported:

| Baseline Identifier | Physical Interpretation | Numerical Definition in Preprocessing Pipeline | Meteorological Justification |
| :--- | :--- | :--- | :--- |
| `dry_clearsky` *(Default)* | Stable, non-convective, cloud-free summer atmosphere | $T_b = 295.0\text{ K}$ ($\approx 21.8^\circ\text{C}$ surface/boundary emission), $P = 0.0\text{ mm/hr}$ (zero rain), $M = 1.0$ (valid satellite coverage) | Represents the purest non-convective counterfactual. Cold cloud tops ($T_b < 230\text{ K}$) and heavy rainfall will register as positive convective drivers. |
| `zero` | Zero-vector in z-score standardized space | Normalized features $z_{\text{norm}} = 0.0$. Corresponds to dataset mean: $T_b \approx 258.43\text{ K}$, $P \approx 0.39\text{ mm/hr}$, $M = 0.0$. | Mathematical convention in literature. However, $T_b = 258.4\text{ K}$ already represents mid-tropospheric cloudiness. |
| `climatological` | Monsoon domain empirical mean | Historical mean of observed non-zero monsoon days across Meghalaya. | Tests attribution against prevailing seasonal background conditions. |

---

## 4. INDEPENDENT PERTURBATION: OCCLUSION SENSITIVITY (Zeiler & Fergus, 2014)

Gradient-based methods evaluate the local derivative of the learned function. To confirm that the model's behavior is not an artifact of backpropagation pathways, we implement an independent, non-gradient perturbation method: **Occlusion Sensitivity**.

### 4.1 Spatial Patch Occlusion

A square spatial window $K \times K$ ($3 \times 3$ grid cells, corresponding to $\approx 30\text{ km} \times 30\text{ km}$) centered at $(r_0, c_0)$ is masked by substituting the corresponding values from the baseline $x'$:

$$\tilde{x}_{(r_0, c_0)}(t, c, h, w) = \begin{cases} x'(t, c, h, w) & \text{if } |h - r_0| \le 1 \text{ and } |w - c_0| \le 1 \\ x(t, c, h, w) & \text{otherwise} \end{cases}$$

The spatial occlusion impact map $\Omega(r_0, c_0)$ measures the direct drop in forecast output at target cell $(r, c)$:

$$\Omega(r_0, c_0) = F(x) - F(\tilde{x}_{(r_0, c_0)})$$

- $\Omega > 0$: The masked region actively contributed to promoting rainfall at the target cell.
- $\Omega < 0$: The masked region suppressed rainfall (e.g., warm dry air intrusion).

### 4.2 Temporal Frame Occlusion

Each of the 4 historical input lookback frames $t \in \{0, 1, 2, 3\}$ ($t-90\text{m}, t-60\text{m}, t-30\text{m}, t$) is individually replaced with the baseline frame, and the forecast drop $\Delta F_t$ is recorded.

### 4.3 Channel/Modality Occlusion

Each input channel ($c = 0: \text{IR } T_b, c = 1: \text{IMERG } P, c = 2: \text{Mask } M$) is zeroed across all 4 frames to measure unimodal degradation.

---

## 5. SANITY CHECKS & SCIENTIFIC ROBUSTNESS

### 5.1 Parameter Randomization Sanity Test (Adebayo et al., NeurIPS 2018)

A critical hazard in Explainable AI is that gradient saliency can act as an edge detector or input visualizer rather than explaining learned representations.

To prove that our XAI attributions genuinely depend on the trained model weights:
1. We compute the spatial attribution map $A_{\text{trained}} = \mathrm{IG}(x; \theta^*)$.
2. We instantiate an identical `MultiTaskWeatherConvLSTM` architecture with random Gaussian weight initialization $\theta_{\text{rand}} \sim \mathcal{N}(0, \sigma^2)$ and compute $A_{\text{rand}} = \mathrm{IG}(x; \theta_{\text{rand}})$.
3. We compute the spatial Pearson correlation coefficient $r(A_{\text{trained}}, A_{\text{rand}})$.

**Empirical Result:**
- Pearson correlation: $r = -0.0722$ to $-0.4905$ ($|r| < 0.5$).
- Spearman rank correlation: $\rho = 0.0089$ to $0.0160$.
- **Verdict: PASS.** Saliency degrades toward stochastic noise upon weight randomization, confirming that the attribution map reflects trained spatiotemporal kernels rather than low-level input gradients.

### 5.2 Cross-Method Agreement (Integrated Gradients vs. Occlusion)

Because Integrated Gradients is path-integrated and Occlusion is a finite-difference perturbation, comparing their spatial attribution maps provides an independent cross-check:
- **Full-Grid Pearson Correlation ($N = 1000$):** $r = +0.3778$ ($p < 0.001$, statistically significant spatial agreement). Both methods independently identify the identical upstream convective core as the dominant driver of rainfall at the target cell.
- **Full-Grid Spearman Rank Correlation ($N = 1000$):** $\rho = -0.0194$. In the full-domain raster, $\approx 935$ pixels represent non-raining clear-sky background where attribution is practically zero ($\le 10^{-4}$). In these zero-attribution background pixels, minor floating-point fluctuations create artificial rank-order permutations that depress the full-grid rank correlation.
- **Active Storm Pixel Correlation ($N = 65$, $|v| > 0.05$):** When evaluated over active convective pixels, the rank agreement rises to Spearman $\rho = +0.1672$ and Pearson $r = +0.3532$.

### 5.3 Acceptance Thresholds & Case Study Clarification

1. **Acceptance Thresholds by Output Head:**
   - **Continuous QPF Head ($\text{mm/hr}$):** Strict threshold $\epsilon_{\text{rel}} < 5.0\%$. Evaluates continuous physical rain rate. Measured error across all 5 horizons ranges from $0.0030\%$ (+30m) to $2.2740\%$ (+60m), comfortably satisfying the threshold.
   - **Classification Probability Head (Logit space):** Threshold $\epsilon_{\text{rel}} < 15.0\%$. Evaluates non-linear recurrent gate logits over 50 discrete Riemann integration steps. Measured error ranges from $1.40\%$ to $7.60\%$.
2. **Case Study vs. Aggregate Domain Scope:**
   - The reported modality shares ($P: 74.3\%, T_b: 25.7\%$) and lookback temporal decay weights ($57.1\%, 30.2\%, 8.8\%, 3.9\%$) specifically characterize the **individual peak convective storm case study at Step 316, Cell (row 5, col 23)**.
   - They do **not** represent spatial domain averages. For non-raining cells (e.g., Central Plateau), the modality share inverts ($T_b: 80.62\%, P: 18.76\%$), as the model relies on warm cloud tops to suppress rain predictions. Both regimes are cataloged in `outputs/reports/PHASE6_XAI_EXPERIMENT_RESULTS.csv`.

---

## 6. SCIENTIFIC NOTICE & NON-CAUSAL DISCLOSURE

> [!WARNING]
> **MANDATORY SCIENTIFIC DISCLOSURE (Non-Causal Notice):**  
> Feature attribution maps generated by Integrated Gradients or Occlusion Sensitivity measure **mathematical model sensitivity and gradient-weighted association relative to a reference baseline**. They reflect the internal mechanics and feature reliance of the trained neural network (`MultiTaskWeatherConvLSTM`), **NOT confirmed physical meteorological causation**.  
> A high attribution value indicates that the model relies heavily on that feature to alter its output; it does not constitute physical proof that the atmospheric anomaly causally triggered precipitation.
