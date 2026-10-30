# Loss Implementation Audit: Formulation, Outputs, and Normalization

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Auditor:** Senior Research Engineer in Spatiotemporal Deep Learning & Remote Sensing  
**Module Audited:** `src/models/experimental/simvp.py` and `src/models/experimental/spatial_losses.py`  
**Test Suite:** `tests/test_spatial_losses.py` (5/5 automated unit tests passing)

---

## 1. Phase A Audit: Resolution of Seven Core Questions

### Q1: Does SimVP actually use MSE, Smooth L1, L1, or a compound loss?
**Answer: SimVP uses a compound multi-task objective combining binary classification and regression.**
Inspection of `train_and_evaluate_research_models.py` (lines 126–128) and `run_rigorous_model_improvement_program.py` (lines 318–320) confirmed:
$$\mathcal{L}_{\text{canonical}} = \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\hat{q}, y_{\text{cont}})$$
with element-wise observation masking. It does **not** use pure unweighted MSE; it uses Smooth L1 for regression paired with binary cross-entropy for occurrence.

### Q2: Does the model produce rainfall rates, rainfall probabilities, or separate outputs?
**Answer: The model produces dual, separate outputs via dedicated decoder heads.**
`SimVPWeather.forward()` returns:
1. `cat_logits`: shape `(B, 5, 25, 40)` unnormalized logits for convective rainfall occurrence ($\ge 5.0\text{ mm/hr}$). Pass through $\sigma(\cdot)$ yields probabilities in $[0, 1]$.
2. `qpf_rates`: shape `(B, 5, 25, 40)` continuous precipitation rates in physical units ($\text{mm/hr}$). The head terminates in `nn.Softplus()`, guaranteeing non-negative continuous rates $\ge 0.0$.

### Q3: What units and normalization are used by each output?
- `cat_logits`: Dimensionless log-odds $\in (-\infty, +\infty)$.
- `qpf_rates`: **Physical precipitation rate in mm/hr** $[0, +\infty)$. They are **not** standardized by $(\mu, \sigma)$ or log-transformed during loss evaluation.
- Input tensors (`x`): Infrared brightness temperature is standardized as $(T_b - 268.442) / 20.619$; precipitation channel is standardized as $(P - 0.709) / 2.110$; observation mask is binary $\{0, 1\}$.

### Q4: Is the existing rainfall target transformed or clipped?
- The ground truth continuous target $y_{\text{cont}}$ is in raw physical units ($\text{mm/hr}$), untransformed and unclipped (ranges from $0.0$ to $37.37\text{ mm/hr}$ in the test set).
- The binary occurrence target $y_{\text{bin}} = (y_{\text{cont}} \ge 5.0)$ is derived dynamically from $y_{\text{cont}}$.

### Q5: How are missing observations masked during training and evaluation?
- Each sequence batch includes a binary spatial mask $M \in \{0, 1\}$ of shape `(B, 5, 25, 40)`.
- Loss computation evaluates element-wise, multiplies by $M$, sums across all elements, and divides by $\max(\sum M, 1.0)$:
  $$\mathcal{L} = \frac{\sum (\ell(\hat{y}, y) \odot M)}{\max(\sum M, 1.0)}$$
- Missing or invalid satellite pixels contribute zero loss and zero gradient.

### Q6: Which checkpoint and configuration correspond to the reported SimVP baseline?
- `outputs/experiments/research_models/checkpoints/simvp_seed42.pt` (trained with AdamW, Cosine Annealing, 15 epochs, lr=1e-3, weight_decay=1e-4, batch_size=16, seed=42).
- `outputs/experiments/model_research/checkpoints/simvp_optimal_best.pt` corresponds to the same architecture and objective selected on validation loss in the initial model research program.

### Q7: Was the checkpoint selected using validation data or independent test results?
- **Strictly validation data.** Checkpoints were selected during training based on the lowest validation loss on July 5, 2025 ($N=33$ sequences). The independent test set (July 12–14, 2023) was never accessed during checkpoint selection or training.

---

## 2. Why the Original Objective Causes Heavy-Rain Attenuation

Although the original objective used Smooth L1 rather than pure MSE, it suffers from the same fundamental issue:
1. **Severe Extreme-Class Imbalance**: Over the Meghalaya domain, $> 70\%$ of grid cells are completely dry ($0\text{ mm/hr}$), and only $\sim 2.8\%$ of cells experience heavy rain ($\ge 5.0\text{ mm/hr}$).
2. **Conditional Expectation Collapse**: Under increasing temporal uncertainty (+180m to +360m), any unweighted regression loss ($L_1$, Smooth $L_1$, or $L_2$) minimizes the expected risk by predicting the conditional mean of the distribution ($\sim 0.8\text{--}1.5\text{ mm/hr}$).
3. Predicting a sharp $20\text{ mm/hr}$ convective core with a slight spatial displacement of $15\text{ km}$ incurs a massive penalty under both Smooth L1 and MSE. Conversely, predicting a flat $1.5\text{ mm/hr}$ blur minimizes the penalty across the entire domain, systematically eliminating convective hazard peaks.

---

## 3. Mathematical Formulation of Experimental Losses

All experimental losses are implemented in `src/models/experimental/spatial_losses.py`:

### 3.1 Experiment B: Intensity-Weighted L1 Loss
To force the model to preserve high-intensity convective amplitudes:
$$w(y) = 1.0 + \alpha \cdot \text{clamp}\left( \frac{y_{\text{cont}}}{q_{\text{th}}}, 0.0, 6.0 \right)$$
where $q_{\text{th}} = 5.0\text{ mm/hr}$ and $\alpha = 0.5$.
To keep the loss scale strictly comparable with the baseline, weights are normalized across valid pixels in the batch:
$$w_{\text{norm}}(y) = \frac{w(y)}{\text{mean}(w(y) \odot M) + \epsilon}$$
$$\mathcal{L}_{\text{Weighted-L1}} = \frac{\sum \left( w_{\text{norm}}(y) \cdot \text{SmoothL1}(\hat{q}, y_{\text{cont}}) \odot M \right)}{\max(\sum M, 1.0)}$$
- A dry pixel ($y=0$) receives normalized weight $\sim 0.85$.
- A convective storm cell ($y=25\text{ mm/hr}$) receives normalized weight $\sim 3.0\times$ higher, directing strong backpropagation gradients to maintain peak amplitudes.

### 3.2 Experiment C: Differentiable Multiscale Soft-FSS Loss
To directly reward placing convective rainfall mass within the regional neighborhood, we construct a smooth surrogate for Fractions Skill Score:
1. **Differentiable Soft Event Indicator**:
   $$\hat{I}_q(x, y) = \sigma\left( \kappa \cdot (\hat{q}(x, y) - q_{\text{th}}) \right)$$
   with $\kappa = 1.0$ and $q_{\text{th}} = 5.0\text{ mm/hr}$. For ground truth, $I_q = (y_{\text{cont}} \ge q_{\text{th}})$.
2. **Multi-Scale Spatial Fraction Fields**:
   Using precomputed 2D circular convolution kernels $K_k$ for radii $k \in \{1, 2, 3\}$ cells ($10, 20, 30\text{ km}$):
   $$\hat{F}_k = \frac{\text{conv2d}(\hat{I}_q \odot M, K_k)}{\text{conv2d}(M, K_k) + \epsilon}, \quad F_{k} = \frac{\text{conv2d}(I_q \odot M, K_k)}{\text{conv2d}(M, K_k) + \epsilon}$$
3. **Soft-FSS Surrogate Objective**:
   $$\mathcal{L}_{\text{Soft-FSS}} = \frac{1}{|K|} \sum_{k \in \{1, 2, 3\}} \frac{\sum (\hat{F}_k - F_k)^2 \odot M + \epsilon}{\sum (\hat{F}_k^2 + F_k^2) \odot M + \epsilon}$$
   - When fractional fields align perfectly, $\mathcal{L}_{\text{Soft-FSS}} \to 0.0$.
   - When predictions and observations have zero spatial overlap, $\mathcal{L}_{\text{Soft-FSS}} \to 1.0$.
   - Bounded in $[0, 1]$, fully differentiable with respect to $\hat{q}$.

### 3.3 Experiment D: Combined Objective
$$\mathcal{L}_{\text{Combined}} = 1.0 \cdot \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{Weighted-L1}}(\hat{q}, y_{\text{cont}}) + 0.5 \cdot \mathcal{L}_{\text{Soft-FSS}}(\hat{q}, y_{\text{cont}})$$
Balances occurrence calibration, extreme-intensity preservation, and multi-scale spatial placement.
