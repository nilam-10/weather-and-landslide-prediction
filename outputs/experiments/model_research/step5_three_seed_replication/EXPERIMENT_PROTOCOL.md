# Step 5 Experiment Protocol: Three-Seed Replication Specification

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (~6 GB VRAM)  
**Execution Environment:** Python 3.12.5, PyTorch 2.6.0+cu124, CUDA 12.4  
**Target Output Directory:** `outputs/experiments/model_research/step5_three_seed_replication/`

---

## 1. Experimental Matrix

We evaluate a balanced $2 \times 3$ factorial design:
- **Factor 1: Training Loss Objective** (2 levels)
  - `Exp A (Baseline)`: $\mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\hat{q}, y_{\text{cont}})$
  - `Exp D (Combined)`: $\mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{Weighted-L1}}(\hat{q}, y_{\text{cont}}) + 0.5 \cdot \mathcal{L}_{\text{Soft-FSS}}(\hat{q}, y_{\text{cont}})$
- **Factor 2: Random Initialization Seed** (3 levels)
  - `Seed 42`
  - `Seed 123`
  - `Seed 999`

| Run ID | Model Name | Configuration | Random Seed | Checkpoint Output File |
| :---: | :--- | :--- | :---: | :--- |
| **Run 1** | `SimVP_ExpA_Seed42` | Baseline Control | 42 | `CHECKPOINTS/ExpA_Seed42_best.pt` |
| **Run 2** | `SimVP_ExpD_Seed42` | Combined Objective | 42 | `CHECKPOINTS/ExpD_Seed42_best.pt` |
| **Run 3** | `SimVP_ExpA_Seed123` | Baseline Control | 123 | `CHECKPOINTS/ExpA_Seed123_best.pt` |
| **Run 4** | `SimVP_ExpD_Seed123` | Combined Objective | 123 | `CHECKPOINTS/ExpD_Seed123_best.pt` |
| **Run 5** | `SimVP_ExpA_Seed999` | Baseline Control | 999 | `CHECKPOINTS/ExpA_Seed999_best.pt` |
| **Run 6** | `SimVP_ExpD_Seed999` | Combined Objective | 999 | `CHECKPOINTS/ExpD_Seed999_best.pt` |

---

## 2. Training Protocol & Hardware Configuration

- **Dataset**:
  - Training: 38 active monsoon days ($N = 1,764$ causal sequences, $T_{\text{in}} = 4, T_{\text{lead}} = 12$).
  - Validation: July 5, 2025 ($N = 33$ sequences).
  - Confirmatory Test: July 12–14, 2023 ($N = 129$ sequences).
- **Architecture**: `SimVPWeather`
  - In channels: 3 per frame ($T_b$, $P$, mask).
  - Hidden spatial channels: 32.
  - Spatiotemporal Translator: 3 Gated Temporal Convolutional blocks.
  - Decoder: Separate occurrence logit head and continuous rain rate head ($\ge 0$ via `Softplus`).
  - Total parameters: 337,138.
- **Optimization**:
  - Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`.
  - LR Schedule: `CosineAnnealingLR(T_max=15, eta_min=1e-5)`.
  - Batch size: 16 (consumes ~2.8 GB VRAM, fitting within 6 GB VRAM budget).
  - Gradient norm clipping: 5.0.
  - Precision: `torch.amp.autocast("cuda")` mixed precision.
  - Training budget: 15 epochs per run (~3.5 to 4.2 minutes per run; total runtime ~23 minutes).
- **Checkpoint Selection**:
  - Evaluated on July 5, 2025 validation set at the end of every epoch.
  - The model state achieving the **lowest total validation loss** is saved as the best checkpoint.
  - The test set is **never** accessed during training or checkpoint selection.

---

## 3. Evaluation & Metrics Specification

### 3.1 Quantitative Precipitation Forecast (QPF)
- **RMSE**: Root Mean Squared Error across all valid spatial grid cells ($\text{mm/hr}$).
- **MAE**: Mean Absolute Error ($\text{mm/hr}$).
- **Pearson $r$**: Linear spatial correlation coefficient.
- **Mean Bias**: $\text{mean}(\hat{q}) - \text{mean}(y_{\text{cont}})$ ($\text{mm/hr}$).

### 3.2 Heavy Convective Rain Detection ($\ge 5.0\text{ mm/hr}$)
Fixed physical threshold: an event is detected when predicted rain rate $\hat{q} \ge 5.0\text{ mm/hr}$.
- **CSI**: $\frac{TP}{TP + FP + FN}$ (Critical Success Index / Threat Score).
- **POD**: $\frac{TP}{TP + FN}$ (Probability of Detection / Hit Rate).
- **FAR**: $\frac{FP}{TP + FP}$ (False Alarm Ratio).
- **Precision**: $\frac{TP}{TP + FP}$.
- **Recall**: $\frac{TP}{TP + FN}$.

### 3.3 Spatial Verification (Fractions Skill Score)
Evaluated following Roberts & Lean (2008) and Mittermaier (2014) using the **Corrected Circular Euclidean Kernel** ($u^2 + v^2 \le k^2$):
- Radii: $10\text{ km}$ ($k=1$, 5 cells), $20\text{ km}$ ($k=2$, 13 cells), $30\text{ km}$ ($k=3$, 29 cells), $50\text{ km}$ ($k=5$, 81 cells).
- Accumulated FSS: $1.0 - \frac{\sum_t FBS_t}{\sum_t FBS_{\text{worst}, t}}$.
- Neighborhood Contingency: Fuzzy $N\text{-CSI}, N\text{-POD}, N\text{-FAR}$ (Ebert 2008).

### 3.4 Intensity Diagnostics
- Ground truth and forecast maxima ($\max(y), \max(\hat{q})$).
- 95th and 99th percentiles of predicted precipitation rates.
- Convective event frequency: count and fraction of cells with $\hat{q} \ge 5.0, 10.0, 20.0\text{ mm/hr}$.
- Frequency bias: $B = \frac{N(\hat{q} \ge 5.0)}{N(y \ge 5.0)}$.
- Matched storm core displacement: Euclidean distance between observed convective core centroid and nearest forecast convective core within $30\text{ km}$.

---

## 4. Statistical Analysis Specification

1. **Per-Seed Reporting**: Metrics reported individually for Seeds 42, 123, 999.
2. **Ensemble Statistics**: Mean ($\mu$) and sample standard deviation ($\sigma$) across $N=3$ seeds:
   $$\mu = \frac{1}{3}\sum_{s} M_s, \quad \sigma = \sqrt{\frac{1}{2}\sum_s (M_s - \mu)^2}$$
3. **Paired Seed Difference**:
   $$\Delta_s = M_{\text{ExpD}, s} - M_{\text{ExpA}, s} \quad \text{for } s \in \{42, 123, 999\}$$
   Mean paired gain: $\bar{\Delta} = \frac{1}{3}\sum_s \Delta_s$.
4. **Temporal Block Bootstrap**:
   To account for temporal autocorrelation among consecutive weather frames, stationary block bootstrapping (block size = 6 half-hour sequences = 3 hours) will be executed over 1,000 iterations to derive 95% confidence intervals and paired permutation $p$-values.

---

## 5. Decision Thresholds for Further Development

The replication will recommend advancing Exp D into the main project pipeline only if:
1. **Consistency**: Exp D outperforms Exp A in $+120\text{m}$ and $+180\text{m}$ heavy-rain detection ($5.0\text{ mm/hr}$) across all three seeds ($\Delta_s > 0$ for every seed).
2. **Spatial Skill Threshold**: Mean circular FSS ($30\text{ km}$) at $+120\text{m}$ exceeds $0.50$ (Roberts & Lean useful skill target), and at $+180\text{m}$ exceeds $0.35$.
3. **Non-Zero 6-Hour Convective Skill**: Exp D maintains predicted maximum rainfall $\ge 10.0\text{ mm/hr}$ and circular FSS $> 0.05$ at $+360\text{m}$ across all seeds (breaking the baseline $0.0000$ collapse).
4. **Short-Term Safety**: Short-term RMSE degradation at $+30\text{m}$ must remain within $5\%$ ($< 0.04\text{ mm/hr}$).
