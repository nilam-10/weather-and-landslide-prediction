# Next Experiment Recommendation: Evidence-Based Pathway Following Spatial Verification

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Hardware Budget:** NVIDIA RTX 4050 Laptop GPU (~6 GB VRAM)  
**Evidence Source:** Step 1 Forensic Audit + Step 2 Spatial Verification (FSS & Neighborhood Contingency)

---

## 1. The Core Scientific Bottleneck Identified by Spatial Verification

The spatial verification experiment revealed two distinct regimes in our nowcasting models:

1. **Short Horizons (+30m, +60m)**:
   Models possess outstanding regional spatial skill ($FSS > 0.80$ at $30\text{ km}$ radius for heavy rain $\ge 5.0\text{ mm/hr}$). Spatial displacement is small ($< 15\text{ km}$), and both ConvLSTM and SimVP maintain actionable accuracy.
2. **Intermediate Horizon (+120m)**:
   SimVP reaches useful skill ($FSS = 0.5466$ at $30\text{ km}$), proving that non-recurrent feed-forward spatial-temporal modeling successfully mitigates recurrent temporal decay.
3. **Long Horizons (+180m, +360m)**:
   For heavy convective rainfall ($\ge 5.0\text{ mm/hr}$), deep learning models experience an absolute collapse ($FSS = 0.0000$ at $+360\text{m}$). 
   **Crucial Diagnostic**: This is NOT caused by incorrect advection direction. It is caused by **intensity attenuation (blurring)**. Because the models are trained with Mean Squared Error (MSE / $L_2$ loss), the model minimizes variance under increasing temporal uncertainty by predicting the conditional mean of the distribution. At $+360\text{m}$, the conditional mean is approximately $0.8\text{--}1.5\text{ mm/hr}$. The model therefore never generates intensities $\ge 5.0\text{ mm/hr}$, resulting in zero true positives and $FSS = 0.0000$.

---

## 2. Evaluation of Candidate Improvements

Based on the forensic audit and spatial verification, we evaluate four potential interventions:

| Intervention | Mechanism | Expected FSS Impact at +180m / +360m | VRAM Feasibility (6 GB) | Priority Rank |
| :--- | :--- | :--- | :--- | :--- |
| **A. Differentiable Soft-FSS / Focal Loss** | Replaces/supplements MSE with a multi-scale neighborhood loss that directly rewards spatial fractions and penalizes peak attenuation. | High: Prevents convective peak collapse; forces model to maintain high-intensity variance. | Excellent: Zero memory overhead during forward/backward pass. | **#1 (Recommended)** |
| **B. Frequency-Weighted Heavy-Rain Loss** | Weights grid cells by $(1 + w \cdot \text{RainRate})$ to overcome the extreme class imbalance of convective peaks ($< 3\%$ base rate). | High: Strongly penalizes underestimating $\ge 5.0\text{ mm/hr}$ storms. | Excellent: Trivial implementation, identical memory footprint. | **#2 (Complementary to A)** |
| **C. Extended Historical Input Context ($T_{in} = 6\text{ or }8$)** | Increases input sequence from 2 hours (4 frames) to 3-4 hours (6-8 frames) to resolve convective lifecycle trends. | Moderate: Provides acceleration and lifecycle context for 6-hour projection. | Moderate: Fits in 6 GB VRAM with batch size 8 or 16. | **#3 (Follow-up)** |
| **D. Massive Heavyweight Transformers (Earthformer/FourCastNet)** | Replaces spatial translator with large attention networks (>10M parameters). | Questionable: High risk of overfitting on modest dataset (1,488 train sequences); severe VRAM exhaustion. | Risky: Prone to CUDA Out-Of-Memory on 6 GB RTX 4050. | Rejected / Low |

---

## 3. The Single Best Next Experiment: Differentiable Multi-Scale Spatial Loss on SimVP

### Experiment Title:
**"Mitigating Convective Intensity Attenuation in Spatiotemporal Nowcasting via Multi-Scale Soft-FSS and Threshold-Weighted Loss"**

### Baseline Architecture:
- `SimVP_Optimal` (~337K parameters) — Established as the clear winner in Step 2 spatial verification ($FSS = 0.5466$ at $+120\text{m}$, outperforming ConvLSTM's $0.3676$).
- RTX 4050 compatible: batch size 16, mixed precision (`torch.cuda.amp.autocast`), training time ~15-20 minutes.

### Loss Formulation:
The current training loss is:
$$\mathcal{L}_{\text{current}} = \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + \lambda \mathcal{L}_{\text{MSE}}(\hat{q}, y_{\text{cont}})$$

We formulate the revised compound spatial loss:
$$\mathcal{L}_{\text{next}} = \mathcal{L}_{\text{BCE}} + \alpha \mathcal{L}_{\text{Weighted-L1}} + \beta \mathcal{L}_{\text{Soft-FSS}}$$

Where:
1. **Intensity-Weighted L1 Loss**:
   $$\mathcal{L}_{\text{Weighted-L1}} = \frac{1}{N} \sum_{(x, y)} \left( 1 + \gamma \cdot y(x, y) \right) \cdot |\hat{y}(x, y) - y(x, y)|$$
   where $\gamma = 0.5$, penalizing errors on convective storms 3-5x more heavily than light drizzle.
2. **Differentiable Multi-Scale Soft-FSS Loss**:
   Using average pooling with kernel sizes $k \in \{1, 3, 5\}$:
   $$\mathcal{L}_{\text{Soft-FSS}} = \sum_{k} \left[ 1 - FSS_{\text{soft}}(\hat{y}, y; k, q = 5.0) \right]$$
   where sigmoid relaxation converts hard binary thresholding into smooth probabilities.

### Quantifiable Success Criteria for Next Experiment:
1. **At $+120\text{m}$ (2 Hours)**:
   - FSS at $30\text{ km}$ radius for $\ge 5.0\text{ mm/hr}$ must exceed **$0.60$** (up from $0.5466$).
2. **At $+180\text{m}$ (3 Hours)**:
   - FSS at $30\text{ km}$ radius for $\ge 5.0\text{ mm/hr}$ must reach useful skill **$FSS \ge 0.40\text{--}0.50$** (up from current $0.2416$).
3. **At $+360\text{m}$ (6 Hours)**:
   - FSS for heavy rain must be non-zero ($FSS > 0.15$), breaking the current total blur collapse ($FSS = 0.0000$).
4. **Safety Verification**:
   - Zero regression on QPF MAE/RMSE at $+30\text{m}$ and $+60\text{m}$.
   - All tests run strictly in an isolated experiment branch without touching `outputs/models/phase3/convlstm_seed42.pt` or FastAPI backend routes.

---

## 4. Execution Roadmap

```
Step 1: Forensic Audit (COMPLETED)
   ├── Verified baseline metrics & failure modes
   └── Root causes: Double penalty, MSE blurring, orographic complexity

Step 2: Spatial Verification (COMPLETED)
   ├── Implemented Roberts & Lean FSS + Ebert Neighborhood Contingency
   ├── 8/8 automated pytest suite passing
   ├── 480 evaluation combinations + paired bootstrap CI (p < 0.001)
   └── Proved SimVP useful at +120m (FSS=0.55), but +360m collapses due to MSE loss

Step 3: Targeted Model Enhancement (READY FOR EXECUTION)
   ├── Implement Multi-Scale Soft-FSS + Weighted-L1 loss function
   ├── Train SimVP with revised spatial loss on RTX 4050 GPU
   ├── Compare against canonical ConvLSTM and frozen SimVP baselines
   └── Re-run spatial verification to measure FSS recovery at +180m and +360m
```
