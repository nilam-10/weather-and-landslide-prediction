# ARCHITECTURE EXPERIMENT PLAN: RANKED CANDIDATES & CONTROLLED ABLATIONS

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Hardware Profile:** NVIDIA GeForce RTX 4050 Laptop GPU (~6 GB VRAM)  
**Study Domain:** Meghalaya Plateau ($25 \times 40$ spatial grid at $0.1^\circ$ resolution)  
**Author:** Senior Research Engineer in Remote Sensing & Spatiotemporal Deep Learning  
**Date:** October 29, 2026  

---

## 1. EVALUATION & RANKING OF CANDIDATE ARCHITECTURES

Each candidate was evaluated against our specific operational constraints:
* **Small spatial dimension:** $25 \times 40$ grid cells (spatial downsampling must not collapse dimensions to $< 3 \times 5$).
* **Limited dataset:** 38 active monsoon days (1,764 sliding sequences); models with $> 2\text{M}$ parameters suffer severe overfitting.
* **3 Input Channels:** Infrared $T_b$, IMERG rain rate, Sensor validity mask.
* **Target:** 5 forecast horizons (+30m, +60m, +120m, +180m, +360m) with dual heads (Rain Occurrence probability + Continuous QPF rain rate).
* **Hardware:** 6 GB VRAM laptop GPU.

| Rank | Candidate Model | Scientific Relevance | Data Suitability ($25\times 40$) | Memory & VRAM Fit | Overfitting Risk | Expected Skill Gain at +120m to +360m | Verdict & Action |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Candidate 1: Canonical ConvLSTM** | Baseline Standard | Optimal ($161\text{ K}$ params) | Minimal ($< 1.0\text{ GB}$) | Very Low | Baseline reference | **MANDATORY BASELINE (Experiment A)** |
| **2** | **Candidate 3: SimVP (Gao et al., 2022)** | High (CVPR 2022 video SOTA) | High (Non-recurrent 2D/temporal blocks) | Excellent ($< 1.2\text{ GB}$) | Low | High (Avoids recursive error accumulation) | **RECOMMENDED (Experiment B)** |
| **3** | **Candidate 2: Compact U-Net–ConvLSTM (Rahimi et al., 2024)** | Highest (Directly designed for GPM IMERG) | High (Multi-scale skips preserve rain cores) | Excellent ($< 1.5\text{ GB}$) | Low to Moderate | High (Multi-scale feature propagation) | **RECOMMENDED (Experiment C)** |
| **4** | **Candidate 6: SmaAt-UNet (Trebing et al., 2021)** | Moderate (Radar nowcasting) | Moderate (Channel stacking lacks causal flow) | Good ($< 1.8\text{ GB}$) | Moderate | Moderate | Reserve for future ablation |
| **5** | **Candidate 5: TrajGRU (Shi et al., 2017)** | Moderate (Deformable flow) | Low (Bilinear flow interpolation on $25\times 40$ is noisy) | Moderate (2–3 GB) | High | Low to Moderate | Deferred (Added complexity unjustified) |
| **6** | **Candidate 7: TUPANN-Inspired Physical Advection (Catão et al., 2026)** | High (Physics-aligned) | Moderate (Coarse grid degrades optical warping) | Moderate (2.5 GB) | Moderate | High in theory, high code risk | Deferred (Requires custom warping kernels) |
| **7** | **Candidate 4: PredRNN-v2 (Wang et al., 2022)** | High (TPAMI 2022) | Poor (12M–24M parameters will severely overfit on 1,764 seqs) | High (4–5.5 GB) | **Extreme** | Poor due to parameter explosion | **REJECTED (Severe Overfitting Risk)** |

---

## 2. THE THREE APPROVED CONTROLLED EXPERIMENTS

To maintain strict scientific isolation and preserve the frozen canonical system, exactly three controlled experiments are approved:

### Experiment A: Verified Canonical ConvLSTM Baseline
* **Architecture:** 2-Layer ConvLSTM (`hidden_channels=(32, 32)`, $k=3$) with dual 5-lead categorical and QPF heads.
* **Trainable Parameters:** 161,034.
* **Objective:** Establish the exact, reproducible reference benchmark across seeds 42, 123, 999 using identical data loaders.

### Experiment B: SimVP (Simpler Yet Better Video Prediction)
* **Architecture:** Completely non-recurrent spatiotemporal architecture adapted for 5-lead dual-head nowcasting:
  1. **Spatial Encoder:** 3-layer 2D CNN with residual connections projecting $(B \cdot 4, 3, 25, 40) \to (B \cdot 4, 32, 25, 40)$.
  2. **Spatiotemporal Translator:** 4-layer Temporal Conv / Gated Conv blocks operating along the sequence dimension without sequential recurrence, preventing vanishing gradients over long lookaheads.
  3. **Spatial Decoder & Dual Heads:** Multi-lead projection heads reconstructing 5 forecast leads for rain probability and continuous QPF.
* **Trainable Parameters:** $\approx 312,000$ (compact, parameter-efficient).
* **Scientific Hypothesis:** Non-recurrent temporal translation avoids recursive hidden-state error compounding, yielding higher retention of storm features at +120m, +180m, and +360m.

### Experiment C: Compact Multi-Scale U-Net–ConvLSTM (Rahimi et al., 2024 adaptation)
* **Architecture:** Multi-scale U-Net encoder-decoder embedding ConvLSTM cells at multiple resolution scales:
  1. **Encoder Level 1 ($25 \times 40$):** Conv2d $(3 \to 24) \to$ ConvLSTMCell $(24 \to 24)$.
  2. **Encoder Level 2 ($12 \times 20$):** Average Pooling $(2 \times 2) \to$ Conv2d $(24 \to 48) \to$ ConvLSTMCell $(48 \to 48)$.
  3. **Bottleneck ($6 \times 10$):** Average Pooling $(2 \times 2) \to$ ConvLSTMCell $(48 \to 64)$.
  4. **Decoder with Skip Connections:** Bilinear Upsampling + Skip concatenations restoring spatial resolution to $25 \times 40$.
  5. **Dual Multi-Lead Heads:** Parallel categorical and QPF projection heads.
* **Trainable Parameters:** $\approx 485,000$.
* **Scientific Hypothesis:** Downsampling creates a wider spatial receptive field capable of capturing synoptic-scale moisture advection, while skip connections preserve fine convective precipitation cores.

---

## 3. CONTROLLED TRAINING IMPROVEMENTS & ABLATIONS

To ensure fair comparison, training improvements will be investigated systematically:

1. **Loss Function Ablation:**
   - Baseline: Masked BCEWithLogits + $0.5 \times$ Smooth L1.
   - Candidate: Focal Loss ($\gamma = 2.0, \alpha = 0.75$) to counter the $3.2\%$ positive event imbalance, combined with Huber loss.
2. **Probability Calibration & Post-Processing:**
   - Temperature scaling / Platt scaling fitted strictly on the July 5, 2025 validation set.
   - Dual-threshold reporting: Fixed $0.50$ threshold vs. Validation-Tuned threshold for each lead time.
3. **Multi-Seed Evaluation:**
   - Models evaluated across Seeds 42, 123, and 999 to establish mean and standard error of the mean (SEM).

---

## 4. IMPLEMENTATION SAFETY PROTOCOL

To ensure zero interference with the frozen major project:
1. All new models will be placed in a dedicated module: [`src/models/experimental/`](file:///d:/Downloads/honours/src/models/experimental/).
2. All new checkpoints will be stored in: `outputs/experiments/research_models/checkpoints/`.
3. The canonical checkpoint [`outputs/models/phase3/convlstm_seed42.pt`](file:///d:/Downloads/honours/outputs/models/phase3/convlstm_seed42.pt) is strictly **READ-ONLY** and will never be overwritten.
4. The production FastAPI backend and Leaflet frontend remain completely untouched.
5. Unit tests validating shape invariants, forward passes, and parameter counts will be written in `tests/test_experimental_models.py`.
