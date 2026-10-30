# Next Experiment Recommendation: Evidence-Based Loss Function Intervention

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Auditor:** Senior Research Engineer in Spatiotemporal Deep Learning & Remote Sensing  
**Hardware Budget:** NVIDIA RTX 4050 Laptop GPU (~6 GB VRAM)  
**Evidence Source:** Step 3 Spatial Verification Audit (Bit-Level Reconciliation, Geometry Correction, Intensity Diagnostics)

---

## 1. Empirical Evidence Justifying the Next Experiment

The Step 3 spatial verification audit produced three decisive empirical facts:

1. **SimVP is the Superior Architectural Backbone**:
   Across all random seeds (42, 123, 999), SimVP consistently outperforms ConvLSTM in both pointwise CSI ($0.11\text{--}0.17$ vs. $0.08\text{--}0.10$ at $+120\text{m}$) and corrected circular FSS ($0.35\text{--}0.52$ vs. $0.28\text{--}0.35$). SimVP also achieves the lowest quantitative RMSE ($1.326\text{ mm/hr}$).
2. **Intermediate Convective Skill is Retained Under Circular Geometry**:
   Under a strict Euclidean circular neighborhood (radius 30 km, 29 cells, $3,223\text{ km}^2$), `SimVP_Optimal` achieves $FSS = \mathbf{0.5192}$ at $+120\text{m}$ for heavy rain ($\ge 5.0\text{ mm/hr}$), crossing the Roberts & Lean useful forecast threshold ($FSS \ge 0.50$).
3. **The 6-Hour Convective Failure is Caused by MSE Intensity Attenuation, NOT Parameter Capacity**:
   Our diagnostic audit (`PREDICTION_INTENSITY_DIAGNOSTICS.csv`) revealed that while ground truth convective peaks reach **$37.37\text{ mm/hr}$**, the maximum rain rate predicted by ANY deep learning model at $+360\text{m}$ is at most **$4.50\text{ mm/hr}$** (and ConvLSTM tops out at $1.50\text{ mm/hr}$). Under Mean Squared Error (MSE / $L_2$) loss, the models minimize variance by predicting the conditional mean of the distribution, completely washing out convective extremes and forcing $FSS = 0.0000$.

---

## 2. Why Architectural Scaling is the Wrong Next Step

A common temptation when long-range forecasts fail is to train larger architectures (e.g. Earthformer, FourCastNet, or Swin-Transformers). However, the evidence demonstrates that:
1. SimVP already has ample capacity (~337K parameters) to capture spatiotemporal advection.
2. Increasing parameters will **not** fix the conditional mean collapse as long as the objective function remains MSE. A 10-million parameter model trained with MSE will still predict the smooth conditional expectation at $+360\text{m}$ to minimize squared residuals.
3. Heavyweight transformers would violate the hardware constraint (6 GB VRAM on the RTX 4050 laptop GPU) and risk severe overfitting on our modest training split (1,764 sequences).

---

## 3. The Single Best Next Experiment: Multi-Scale Soft-FSS + Weighted-L1 Loss on SimVP

### Objective:
Train `SimVPWeather` using a **differentiable spatial-scale loss function** that directly optimizes neighborhood fractional alignment and penalizes intensity attenuation at long forecast horizons (+180m and +360m).

### Mathematical Formulation:
Currently, the model is trained with:
$$\mathcal{L}_{\text{current}} = \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + \lambda \mathcal{L}_{\text{MSE}}(\hat{q}, y_{\text{cont}})$$

We formulate the revised compound objective:
$$\mathcal{L}_{\text{next}} = \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + \alpha \mathcal{L}_{\text{Weighted-L1}}(\hat{q}, y_{\text{cont}}) + \beta \mathcal{L}_{\text{Soft-FSS}}(\hat{q}, y_{\text{cont}})$$

#### 1. Intensity-Weighted $L_1$ Loss:
To prevent convective peak dampening without the extreme outlier sensitivity of $L_2$:
$$\mathcal{L}_{\text{Weighted-L1}} = \frac{1}{N_V} \sum_{(x, y) \in \mathcal{D}_{\text{valid}}} \left( 1.0 + \gamma \cdot y_{\text{cont}}(x, y) \right) \cdot |\hat{q}(x, y) - y_{\text{cont}}(x, y)|$$
with $\gamma = 0.5$. A $10\text{ mm/hr}$ convective storm cell receives $6\times$ higher gradient weight than a dry pixel, actively preventing the model from collapsing to the low-intensity mean.

#### 2. Differentiable Multi-Scale Soft-FSS Loss:
Using smooth sigmoid relaxation for binary thresholding ($\hat{I}_q = \sigma(\kappa \cdot (\hat{q} - q))$ with $\kappa = 2.0$), we compute average pooling across circular kernels with radii $k \in \{1, 2, 3\}$ cells ($10, 20, 30\text{ km}$):
$$\mathcal{L}_{\text{Soft-FSS}} = \sum_{k \in \{1, 2, 3\}} \left[ 1.0 - \text{FSS}_{\text{soft}}(\hat{q}, y_{\text{cont}}; k, q = 5.0) \right]$$
This forces the backpropagation gradient to reward placing fractional convective mass within the correct regional neighborhood, even if slightly displaced.

---

## 4. Hardware Feasibility & Experimental Protocol

- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
- **Batch Size**: 16 (fits comfortably in ~2.8 GB VRAM using mixed precision `torch.cuda.amp.autocast`).
- **Training Time**: ~15 minutes for 15 epochs with early stopping on validation loss.
- **Seeds**: 42, 123, 999 (full 3-seed evaluation).
- **Safety Safeguard**:
  - Training will be conducted strictly in `outputs/experiments/model_research/checkpoints/simvp_soft_fss_seed*.pt`.
  - The canonical production checkpoint `outputs/models/phase3/convlstm_seed42.pt`, the FastAPI backend, and the Leaflet frontend will remain **completely untouched and protected**.

---

## 5. Explicit Quantifiable Success Criteria

To establish whether the experiment succeeds, the revised SimVP must satisfy:
1. **At $+120\text{m}$ (2 Hours)**:
   - Circular FSS ($30\text{ km}$) for $\ge 5.0\text{ mm/hr}$ must exceed **$0.55$** (up from current $0.5192$).
2. **At $+180\text{m}$ (3 Hours)**:
   - Circular FSS ($30\text{ km}$) for $\ge 5.0\text{ mm/hr}$ must reach useful skill **$FSS \ge 0.40$** (up from current $0.2262$).
3. **At $+360\text{m}$ (6 Hours)**:
   - Must break the zero-skill collapse: predicted maximum rain rate must exceed $5.0\text{ mm/hr}$, achieving **$FSS > 0.10$** (up from current $0.0000$).
4. **Intensity Integrity**:
   - Short-term QPF RMSE at $+30\text{m}$ and $+60\text{m}$ must not degrade by more than $5\%$ relative to the baseline ($0.734\text{ mm/hr}$ and $1.002\text{ mm/hr}$).

---

## 6. Current Status & Stop-Rule Confirmation

In accordance with user instructions:
- **Zero models have been retrained.**
- **All existing checkpoints and results have been preserved.**
- **We have paused and will wait for user review before proceeding with any training execution.**
