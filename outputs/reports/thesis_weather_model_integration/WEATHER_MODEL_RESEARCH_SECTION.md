# Weather Nowcasting Research: Spatiotemporal Modeling & Compound Spatial Loss Optimization

**Document:** B.Tech Honours Major Project — Weather Research Chapter  
**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Study Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Target Submission:** Bachelor of Technology (Honours) Major Project Thesis (2026–2027)  
**Primary Evidence References:** Step 4 Pilot, Step 5 Three-Seed Replication, and Step 6 Integrity Audit  
**Status:** **AUTHORITATIVE THESIS RESEARCH SECTION**

---

## 1. Introduction and Problem Context

Accurate precipitation nowcasting across complex orographic terrain is essential for timely hydrometeorological hazard mitigation. In the Meghalaya Plateau of Northeast India—one of the wettest regions on Earth, characterized by abrupt topographic escarpments rising from near sea level to over $1,500\text{ m}$ within $20\text{ km}$—convective storm systems produce intense localized rainfall capable of triggering rapid slope failures and flash flooding.

### 1.1 The Classical Nowcasting Trade-off
Traditional spatiotemporal deep learning models (such as ConvLSTM and standard SimVP architectures) trained with mean squared error (MSE) or Smooth-$L_1$ regression objectives suffer from severe **intensity attenuation** as the forecast horizon extends from $+30$ to $+360$ minutes. Under conditional mean optimization, the neural network minimizes loss by predicting a blurred average across multiple plausible future storm trajectories. While this strategy optimizes domain-wide bulk metrics (such as Root Mean Squared Error, RMSE), it flattens localized convective cores, eroding heavy rainfall peaks to below actionable thresholds and causing convective Critical Success Index (CSI) and Fractions Skill Score (FSS) to collapse to zero beyond 2 to 3 hours.

### 1.2 Research Objective
This chapter documents a systematic research program to mitigate convective intensity attenuation in spatiotemporal nowcasting models. Specifically, we investigate whether a **Compound Spatial Loss Objective (Exp D)**—integrating intensity-weighted $L_1$ regression and multiscale differentiable Soft Fractions Skill Score (Soft-FSS) alongside categorical binary cross-entropy—restores convective heavy-rainfall prediction across multiple forecast horizons (+30m, +60m, +120m, +180m, +360m) without destabilizing bulk precipitation error.

---

## 2. Spatiotemporal Model Architecture: SimVPWeather

Rather than relying solely on recurrent gates, we employ **SimVP (Simple Video Prediction)** adapted for meteorological nowcasting (`SimVPWeather`). SimVP completely decouples spatial feature extraction from temporal evolution, allowing efficient convolution-based sequence modeling.

```
Input Frames (T_in = 4, C = 3)
[T_b (Infrared), P (IMERG), Mask]
               │
               ▼
┌───────────────────────────────┐
│     Spatial Encoder (CNN)     │  4 Conv2d layers with LeakyReLU
│  Downsamples (25, 40) → (6, 10)│  Hidden Channels = 32
└──────────────┬────────────────┘
               │ Spatial Latent Tensors (B, 4, 32, 6, 10)
               ▼
┌───────────────────────────────┐
│  Spatiotemporal Translator    │  3 Gated Temporal Convolution (GTC)
│      (GTC Blocks)             │  Learns nonlinear cloud advection
└──────────────┬────────────────┘
               │ Predicted Latents (B, 5, 32, 6, 10)
               ▼
┌───────────────────────────────┐
│     Spatial Decoder (CNN)     │  4 ConvTranspose2d layers
│  Upsamples (6, 10) → (25, 40) │  Restores native grid resolution
└──────────────┬────────────────┘
               ├──► Head 1: Categorical Occurrence Logits (5 leads)
               └──► Head 2: Continuous Rain Rate via Softplus (5 leads, mm/hr)
```

### 2.1 Architectural Specifications
- **Input Dimensions:** 4 historical frames sampled every 30 minutes ($T_{\text{in}} = 4$, spanning $t - 90\text{m}$ to $t$).
- **Input Channels:** 3 physical channels per frame:
  1. INSAT-3D/3DR Thermal Infrared 1 Brightness Temperature ($T_b$, normalized: $\mu = 268.442\text{ K}, \sigma = 20.619\text{ K}$).
  2. GPM IMERG Precipitation Rate ($P$, normalized: $\mu = 0.709\text{ mm/hr}, \sigma = 2.110\text{ mm/hr}$).
  3. Observation Validity Mask ($M \in \{0, 1\}$).
- **Spatial Resolution:** $25 \times 40$ regular grid cells covering $89.55^\circ\text{E} - 93.45^\circ\text{E}$ and $24.55^\circ\text{N} - 26.95^\circ\text{N}$ at $0.10^\circ \approx 10.55\text{ km}$ spacing.
- **Translator Architecture:** 3 stacked Gated Temporal Convolution (GTC) blocks with hidden channel dimension $C_{\text{hidden}} = 32$.
- **Dual-Task Decoders:**
  - *Occurrence Head:* Outputs 5 independent logits corresponding to forecast leads $+30, +60, +120, +180,$ and $+360$ minutes.
  - *Quantitative Precipitation Forecast (QPF) Head:* Outputs non-negative continuous rain rates ($\text{mm/hr}$) enforced via `Softplus` activation.
- **Parameter Footprint:** 337,138 trainable parameters (substantially more expressive than the 161,034-parameter canonical ConvLSTM, while remaining computationally lightweight for local GPU training).

---

## 3. Mathematical Formulation of the Compound Objective

The baseline model (Exp A) was optimized using standard multi-task loss:
$$\mathcal{L}_{\text{Baseline}} = \mathcal{L}_{\text{BCE}}(\hat{\mathbf{p}}, \mathbf{y}_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\hat{\mathbf{q}}, \mathbf{y}_{\text{cont}})$$

To directly penalize convective attenuation and spatial displacement errors, we formulated the **Compound Nowcasting Objective (Exp D)**:
$$\mathcal{L}_{\text{Compound}} = w_{\text{bce}} \mathcal{L}_{\text{BCE}} + w_{\text{qpf}} \mathcal{L}_{\text{Weighted-L1}} + w_{\text{fss}} \mathcal{L}_{\text{Soft-FSS}}$$
where $w_{\text{bce}} = 1.0, w_{\text{qpf}} = 0.5,$ and $w_{\text{fss}} = 0.5$.

### 3.1 Intensity-Weighted L1 Loss ($\mathcal{L}_{\text{Weighted-L1}}$)
Standard $L_1$ and $L_2$ losses treat errors equally regardless of observed rainfall intensity. Because non-raining or light-rain pixels vastly outnumber convective storm pixels, gradients are dominated by near-zero background values. We introduce an exponential intensity weight that amplifies penalties for heavy rainfall:

$$\mathcal{L}_{\text{Weighted-L1}} = \frac{1}{\sum_{i} m_i} \sum_{i} m_i \cdot w_i \cdot |\hat{q}_i - y_i|$$
$$w_i = 1.0 + \alpha \cdot \log\left(1.0 + \max(0, y_i - \tau)\right)$$

where $\tau = 5.0\text{ mm/hr}$ is the convective precipitation threshold, $\alpha = 0.5$ is the scaling factor, and $m_i \in \{0, 1\}$ is the spatial validity mask. For $y_i < 5.0\text{ mm/hr}$, $w_i = 1.0$ (standard $L_1$); for $y_i \ge 5.0\text{ mm/hr}$, the penalty grows logarithmically with true rainfall rate, preventing the model from zeroing out peak convective cores.

### 3.2 Multiscale Differentiable Soft-FSS Loss ($\mathcal{L}_{\text{Soft-FSS}}$)
Pointwise losses severely penalize small spatial displacements ("double penalty" problem). The Fractions Skill Score (Roberts & Lean, 2008) measures spatial neighborhood agreement. To make FSS directly optimizable via gradient descent, we implement **Differentiable Soft-FSS** using a temperature-scaled sigmoid relaxation:

1. **Continuous Event Probability Field:**
   $$\tilde{\mathbf{B}}_{\text{fcst}} = \sigma\left(\frac{\hat{\mathbf{q}} - \tau}{T}\right), \quad \mathbf{B}_{\text{obs}} = \mathbb{I}(y \ge \tau)$$
   where $\tau = 5.0\text{ mm/hr}$ and $T = 0.5$ is the temperature parameter ensuring smooth gradient flow.

2. **Spatial Fraction Smoothing:**
   For a spatial neighborhood of radius $k$ cells, smoothed fraction fields are generated via 2D depthwise convolution with a uniform kernel $\mathbf{K}_k$:
   $$\mathbf{P}_{\text{fcst}}^{(k)} = \mathbf{K}_k * (\tilde{\mathbf{B}}_{\text{fcst}} \odot \mathbf{M}) \oslash (\mathbf{K}_k * \mathbf{M})$$
   $$\mathbf{P}_{\text{obs}}^{(k)} = \mathbf{K}_k * (\mathbf{B}_{\text{obs}} \odot \mathbf{M}) \oslash (\mathbf{K}_k * \mathbf{M})$$

3. **Multiscale Soft Fractions Skill Score:**
   $$\text{FBS}^{(k)} = \frac{1}{N_{\text{valid}}} \sum (\mathbf{P}_{\text{fcst}}^{(k)} - \mathbf{P}_{\text{obs}}^{(k)})^2$$
   $$\text{FBS}_{\text{worst}}^{(k)} = \frac{1}{N_{\text{valid}}} \sum \left[(\mathbf{P}_{\text{fcst}}^{(k)})^2 + (\mathbf{P}_{\text{obs}}^{(k)})^2\right]$$
   $$\text{SoftFSS}^{(k)} = 1.0 - \frac{\text{FBS}^{(k)}}{\text{FBS}_{\text{worst}}^{(k)} + \epsilon}$$
   $$\mathcal{L}_{\text{Soft-FSS}} = 1.0 - \frac{1}{|K|} \sum_{k \in K} \text{SoftFSS}^{(k)}$$

We optimize across multiple spatial scales simultaneously: $K = \{1, 2, 3\}$ grid cells, corresponding to neighborhood radii of approximately $10\text{ km}$, $20\text{ km}$, and $30\text{ km}$.

---

## 4. Experimental Design and Data Partitions

To prevent data snooping and information leakage, all sequence generation adheres to strictly causal boundaries.

### 4.1 Dataset Partitions

| Partition | Calendar Dates | Total Days | Sequences ($N$) | Temporal Stride | Methodological Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Training Split** | • July 1–10, 2023 (10 d)<br>• July 1–14, 2024 (14 d)<br>• Aug 5–14, 2024 (10 d)<br>• July 1–4, 2025 (4 d) | **38 days** | **1,764** | 30 min | Parameter optimization via AdamW ($lr=10^{-3}, \text{weight\_decay}=10^{-4}$). |
| **Validation Split** | • July 5, 2025 (1 d) | **1 day** | **33** | 30 min | Checkpoint selection (lowest validation loss epoch). |
| **Confirmatory Benchmark** | • July 12–14, 2023 (3 d) | **3 days** | **129** | 30 min | Out-of-sample evaluation on locked historical benchmark. |

### 4.2 Training Protocol
- **Optimization:** AdamW with cosine annealing learning rate schedule ($T_{\text{max}} = 15, \eta_{\text{min}} = 10^{-5}$).
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (~6 GB VRAM) using mixed precision (`torch.amp.autocast`).
- **Checkpoint Selection Rule:** Models were evaluated on the July 5, 2025 validation set at the end of each epoch. The checkpoint achieving the **lowest total validation loss** was frozen. The test set was **never** accessed during training or model selection.

---

## 5. Quantitative Verification Results

### 5.1 Three-Seed Replication Summary (Validation Split, July 5, 2025)
To ensure reproducibility, a balanced $2 \times 3$ factorial replication was executed across three independent seeds ($42, 123, 999$). All values represent mean $\pm$ standard deviation across seeds ($N=3$).

| Horizon | Metric | Baseline (Exp A) | Combined (Exp D) | Absolute Gain ($\Delta$) | Relative Gain |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+30m** | Pointwise CSI ($\ge 5\text{ mm/hr}$) | $0.3491 \pm 0.0263$ | $\mathbf{0.3819 \pm 0.0160}$ | $+0.0328$ | $+9.4\%$ |
| | Circular FSS ($30\text{ km}$, $\ge 5\text{ mm/hr}$) | $0.7583 \pm 0.0464$ | $\mathbf{0.7858 \pm 0.0121}$ | $+0.0275$ | $+3.6\%$ |
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

---

### 5.2 Out-of-Sample Benchmark Validation (July 12–14, 2023, $N=129$)
To verify that gains are not an artifact of July 5, 2025, checkpoints were evaluated without modification on the 3-day confirmatory benchmark (72.0 total observation hours).

| Horizon | Metric | Exp A (Benchmark Mean) | Exp D (Benchmark Mean) | Mean Difference ($\bar{\Delta}$) | 95% Bootstrap CI | Bounded $p$-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+120m** | Pointwise CSI | $0.1233 \pm 0.0413$ | $\mathbf{0.2133 \pm 0.0285}$ | $\mathbf{+0.0901}$ | $[+0.0633, +0.1139]$ | $\mathbf{p < 0.001}$ |
| | Circular FSS ($30\text{ km}$) | $0.4113 \pm 0.1009$ | $\mathbf{0.6000 \pm 0.0344}$ | $\mathbf{+0.1887}$ | $[+0.1340, +0.2526]$ | $\mathbf{p < 0.001}$ |
| **+180m** | Pointwise CSI | $0.0120 \pm 0.0132$ | $\mathbf{0.1300 \pm 0.0231}$ | $\mathbf{+0.1180}$ | $[+0.0652, +0.1654]$ | $\mathbf{p < 0.001}$ |
| | Circular FSS ($30\text{ km}$) | $0.0527 \pm 0.0556$ | $\mathbf{0.4333 \pm 0.0424}$ | $\mathbf{+0.3807}$ | $[+0.2241, +0.4789]$ | $\mathbf{p < 0.001}$ |
| | QPF RMSE ($\text{mm/hr}$) | $1.5224 \pm 0.0122$ | $\mathbf{1.5216 \pm 0.0165}$ | $-0.0008$ | $[-0.0617, +0.0545]$ | $p = 0.449$ |
| **+360m** | Pointwise CSI | $0.0000 \pm 0.0000$ | $\mathbf{0.0125 \pm 0.0129}$ | $+0.0125$ | $[+0.0038, +0.0223]$ | $p = 0.0020$ |
| | Circular FSS ($30\text{ km}$) | $0.0000 \pm 0.0000$ | $\mathbf{0.0526 \pm 0.0495}$ | $+0.0526$ | $[+0.0126, +0.0984]$ | $\mathbf{p < 0.001}$ |

---

## 6. Intensity Diagnostics and Spatial Displacement Analysis

### 6.1 Intensity Retention vs. Conditional Mean Blur
At the +180-minute lead time, true observed precipitation peaks reached **$48.58\text{ mm/hr}$** with $1,544$ storm pixel instances exceeding $5.0\text{ mm/hr}$.
- **Baseline Exp A:** Predicted an average maximum intensity of only **$7.06\text{ mm/hr}$** and a 99th percentile of **$3.05\text{ mm/hr}$**. Convective event frequency collapsed to 60 cells (frequency bias = $0.038$, a 96% under-prediction).
- **Compound Loss Exp D:** Preserved an average maximum intensity of **$13.15\text{ mm/hr}$** and a 99th percentile of **$6.17\text{ mm/hr}$**. Convective cell count recovered to 666 instances (frequency bias = $0.431$, an $11.3\times$ recovery).

### 6.2 Forensic Examination of the ~110 km Displacement Finding
The experimental evaluation script reported a mean convective storm core displacement of approximately **$109.9\text{ km}$** for Exp D at +180m. We conducted a forensic audit of this diagnostic to clarify its meteorological meaning:

1. **Diagnostic Formulation:**
   For each evaluation frame $t$, the script identified the single maximum pixel in the valid domain:
   $$(\hat{y}, \hat{x}) = \arg\max \hat{\mathbf{q}}_t, \quad (y^*, x^*) = \arg\max \mathbf{y}_t$$
   Displacement was computed using physical grid spacing:
   $$d = \sqrt{\left((\hat{y} - y^*) \cdot 11.08\text{ km}\right)^2 + \left((\hat{x} - x^*) \cdot 10.03\text{ km}\right)^2}$$

2. **Progression of Median Displacements Across Lead Times:**
   - **+30m:** Median displacement = **$14.9\text{ km}$** on Test ($47.2\text{ km}$ on Val). Matched events = 112 / 117.
   - **+60m:** Median displacement = **$24.3\text{ km}$** on Test ($43.0\text{ km}$ on Val). Matched events = 102 / 117.
   - **+120m:** Median displacement = **$83.0\text{ km}$** on Test ($68.7\text{ km}$ on Val). Matched events = 77 / 117.
   - **+180m:** Median displacement = **$157.1\text{ km}$** on Test ($73.3\text{ km}$ on Val). Matched events = 62 / 117.

3. **Methodological Interpretation:**
   - At $+30\text{m}$ and $+60\text{m}$, storm cores are placed within $15\text{--}25\text{ km}$ of observed positions, directly matching the high Fractions Skill Score ($\text{FSS} > 0.70$ at $30\text{ km}$).
   - At $+180\text{m}$, advection and storm evolution over 3 hours produce larger centroid offsets ($70\text{--}150\text{ km}$). This explains why Pointwise CSI is modest ($\sim 0.13$), while Circular FSS at $30\text{ km}$ is substantial ($0.4333$): the model successfully predicts convective storm existence and intensity in the general region, but localized cell positioning drifts by several grid cells.
   - **Multi-Peak Ambiguity in Large Offsets:** Detailed analysis of frames with argmax displacements $> 100\text{ km}$ revealed that 100% of these cases (9 of 9 on validation) contained multiple simultaneous local convective cores across the plateau (4 to 22 observed peaks, 1 to 9 forecast peaks). Because the metric evaluates only the single global domain maximum, an offset of $\sim 150\text{--}240\text{ km}$ occurs when the observation's primary peak is in the eastern Khasi Hills while the model's primary peak is in the western Garo Hills, even if both convective clusters are actively predicted.
   - **Limitation:** This metric evaluates only the single global domain maximum (`argmax`) on frames where both fields exceed $5.0\text{ mm/hr}$. It is a heuristic peak-offset diagnostic, **not** an object-based multi-cell tracking algorithm.

---

## 7. Conclusions and Scientific Defensibility

1. **Convective Attenuation Successfully Mitigated:** The compound spatial loss objective (Intensity-Weighted L1 + Multiscale Differentiable Soft-FSS + BCE) consistently restores intermediate-range (+120m and +180m) convective nowcasting skill over standard regression baselines across three independent seeds.
2. **Preservation of Bulk Regression Performance:** Unlike aggressive threshold weighting schemes that destabilize background predictions, Exp D exhibited no statistically significant difference in bulk QPF RMSE compared to baseline Exp A ($1.5216\text{ mm/hr}$ vs $1.5224\text{ mm/hr}$ on test, $p = 0.417$, 95% CI: $[-0.0616, +0.0483]\text{ mm/hr}$ under $L=12$ block bootstrap). This indicates that convective heavy-cell retention was achieved without detectable degradation in domain-wide bulk error at the evaluated sample size.
3. **Statistical Validity:** Significance with respect to temporal meteorological sampling was confirmed via stationary block bootstrapping ($p < 0.001$), while across-seed consistency was verified across $N=3$ random initializations.
4. **Physical Boundaries Preserved:** At the 6-hour forecast horizon (+360m), satellite-based nowcasting without atmospheric thermodynamic soundings or NWP coupling remains subject to physical predictability limits ($\text{FSS} \approx 0.05 \ll 0.50$). Operational readiness is not claimed at 6 hours.
