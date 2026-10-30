# Model Results & Research Findings Explained

**Project Title:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Program:** B.Tech Honours Major Project (2026–27), Royal Global University  
**Target Audience:** Students, Faculty Advisors, and External Examiners  
**File Path:** `outputs/reports/local_run_and_code_guide/MODEL_RESULTS_EXPLAINED.md`  

---

## 1. Which Model Is Actually Running in the Live Application?

> [!IMPORTANT]
> **Production vs Research Pilot:**  
> The live interactive web application running at `http://127.0.0.1:8000/` strictly loads and executes the **Canonical MultiTask ConvLSTM** checkpoint (`outputs/models/phase3/convlstm_seed42.pt`, 161,034 parameters).  
> The experimental **SimVP** architecture and the three-seed loss function experiments were conducted as an **offline research investigation** (Steps 4 through 8). They provide valuable comparative insights for the thesis dissertation, but were kept separate from the production server to maintain zero risk of runtime regressions.

---

## 2. Model Architectures: ConvLSTM vs. SimVP

| Architectural Dimension | Canonical ConvLSTM (Production) | Experimental SimVP (Research Pilot) |
| :--- | :--- | :--- |
| **Model Type** | Spatiotemporal Recurrent Neural Network (RNN) | Non-recurrent CNN Encoder–Translator–Decoder |
| **Parameter Count** | **161,034 parameters** (~161 K) | **337,426 parameters** (~337 K) |
| **Recurrence Mechanism** | Hidden state $h_t$ and memory cell $c_t$ updated sequentially across time steps via convolutional gates. | Fully feedforward; temporal dynamics modeled by stacked 2D convolutional Translator blocks. |
| **Computational Complexity** | Sequential $O(T)$ forward pass; higher memory footprint during backpropagation through time (BPTT). | Parallel spatial and temporal processing; lower training latency per epoch. |
| **Key Strength** | Strong inductive bias for continuous physical advection and local recurrent tracking. | Larger receptive field; captures multiscale mesoscale convective patterns without vanishing gradients. |
| **Key Limitation** | Recurrent feature blur; progressive smoothing of high-intensity rain peaks at lead times $\ge 120\text{ min}$. | Slightly more sensitive to random seed initialization on sparse extreme events. |

---

## 3. Spatiotemporal Setup: Inputs and Forecast Leads

The nowcasting system operates on a fixed half-hourly temporal cadence:
- **Historical Input Window ($T_{\text{in}} = 4$):**
  The model ingests 4 consecutive 30-minute observation frames covering the past 2 hours ($t-90\text{m}, t-60\text{m}, t-30\text{m}, t_0$).
- **Input Channels ($C = 3$):**
  1. Channel 0: NASA MERGIR Thermal Infrared (TIR-1) $4\text{ km}$ Brightness Temperature (detects convective cloud-top cooling).
  2. Channel 1: NASA GPM IMERG calibrated half-hourly precipitation rate ($\text{mm/hr}$).
  3. Channel 2: Satellite observation quality flag ($1.0 = \text{valid}, 0.0 = \text{missing}$).
- **Spatial Resolution & Domain:**
  $25 \times 40$ grid cells covering Meghalaya ($24.5^\circ - 27.0^\circ\text{N}, 89.5^\circ - 93.5^\circ\text{E}$), equivalent to $\approx 10\text{ km}$ per cell.
- **Forecast Horizons ($5\text{ Leads}$):**
  $$\text{Lead Times} \in \{+30\text{ min},\; +60\text{ min},\; +120\text{ min},\; +180\text{ min},\; +360\text{ min}\}$$

---

## 4. Loss Functions: Why Standard MSE Fails for Convective Storms

In traditional machine learning, models are trained using Mean Squared Error (MSE) or standard L1 loss. However, atmospheric precipitation fields are **sparse and highly skewed**: more than $92\%$ of grid cells have zero rainfall, while severe convective cells exceed $25\text{ mm/hr}$. 

When trained with standard MSE, a neural network quickly learns that predicting a blurry, low-intensity average minimizes squared errors, leading to the **"blurriness problem"** where severe storm peaks are erased.

To overcome this, our research developed and tested a composite objective function:

### 1. Weighted Binary Cross-Entropy Loss ($\mathcal{L}_{\text{BCE}}$)
Focuses on event detection for heavy convective rain ($\ge 5\text{ mm/hr}$):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i} \left[ w_{\text{pos}} y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
where $w_{\text{pos}} \approx 20.0$ counters the extreme class imbalance.

### 2. Intensity-Weighted L1 Loss ($\mathcal{L}_{\text{Int-L1}}$)
Penalizes underprediction of heavy rainfall more severely than light drizzle:
$$\mathcal{L}_{\text{Int-L1}} = \frac{1}{N} \sum_{i} \left( 1.0 + \alpha \cdot \frac{y_i}{\max(y) + \epsilon} \right) \cdot \left| y_i - \hat{y}_i \right|$$
This forces the model to maintain peak storm intensities rather than collapsing to the spatial mean.

### 3. Differentiable Multiscale Soft Fractions Skill Score ($\mathcal{L}_{\text{Soft-FSS}}$)
Standard spatial metrics like FSS are non-differentiable step functions. We formulated a smooth, differentiable surrogate using average-pooling kernels across multiple neighborhood scales ($k \in \{3, 5, 9\}$ grid cells):
$$\mathcal{L}_{\text{Soft-FSS}} = \frac{1}{K} \sum_{k} \left( 1.0 - \text{FSS}_k(\hat{\mathbf{Y}}, \mathbf{Y}) \right)$$
This rewards the network for placing rainfall in the approximately correct geographic basin, avoiding the harsh "double-penalty" of point-by-point pixel loss.

### Combined Research Objective (Experiment D):
$$\mathcal{L}_{\text{Combined}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{reg}} \mathcal{L}_{\text{Int-L1}} + \lambda_{\text{FSS}} \mathcal{L}_{\text{Soft-FSS}}$$

---

## 5. Evaluation Metrics Explained

### Critical Success Index (CSI) / Threat Score
Measures event detection accuracy for convective rain ($\ge 5\text{ mm/hr}$):
$$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$
- Value ranges from $0.0$ (no skill) to $1.0$ (perfect forecast).
- Unlike Accuracy, CSI completely ignores True Negatives (dry cells), making it the gold standard in operational meteorology.

### Fractions Skill Score (FSS)
Evaluates spatial neighborhood agreement across a moving window of size $n \times n$:
$$\text{FSS}_{(n)} = 1 - \frac{\text{MSE}_{(n)}}{\text{MSE}_{(n),\text{ref}}}$$
- $\text{FSS} = 0.0$: No spatial overlap.
- $\text{FSS} \ge 0.5 + \frac{f_0}{2}$: Statistically useful forecast (Roberts & Lean threshold).
- $\text{FSS} = 1.0$: Perfect spatial agreement.

### Root Mean Square Error (RMSE)
Measures the magnitude of continuous rain-rate prediction error in $\text{mm/hr}$:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

---

## 6. Three-Seed Replication Findings (Step 5 Audit)

To confirm whether the combined loss function provided genuine, reproducible benefits or was merely a fluke of random weight initialization, we conducted a rigorous 3-seed replication study (Seeds 42, 123, 456) comparing **Baseline SimVP (Exp A)** against **Combined Loss SimVP (Exp D)** on the independent validation split:

| Lead Time | Metric | Baseline Exp A (Mean $\pm$ SD) | Combined Loss Exp D (Mean $\pm$ SD) | Measured Effect |
| :---: | :---: | :---: | :---: | :--- |
| **+30 min** | CSI ($\ge 5\text{ mm/hr}$) | $0.284 \pm 0.012$ | **$0.312 \pm 0.009$** | $+9.9\%$ relative gain ($p < 0.05$) |
| **+30 min** | FSS ($3 \times 3$) | $0.518 \pm 0.015$ | **$0.554 \pm 0.011$** | $+6.9\%$ spatial alignment gain |
| **+60 min** | CSI ($\ge 5\text{ mm/hr}$) | $0.198 \pm 0.008$ | **$0.224 \pm 0.007$** | $+13.1\%$ relative gain |
| **+120 min** | CSI ($\ge 5\text{ mm/hr}$) | $0.115 \pm 0.006$ | **$0.138 \pm 0.005$** | $+20.0\%$ relative gain |
| **+180 min** | CSI ($\ge 5\text{ mm/hr}$) | $0.062 \pm 0.005$ | **$0.076 \pm 0.004$** | $+22.5\%$ attenuation reduction |
| **+360 min** | CSI ($\ge 5\text{ mm/hr}$) | $0.014 \pm 0.002$ | $0.016 \pm 0.003$ | Negligible / Boundary Limit |

### Key Takeaway for the Thesis:
1. The combined objective function **consistently improved CSI and FSS across all three random seeds** for lead times up to +180 minutes.
2. The multiscale Soft-FSS component significantly reduced spatial displacement errors.
3. However, at +360 minutes (6 hours), the improvement remains statistically marginal, confirming a fundamental physical boundary.

---

## 7. Why 6-Hour (+360 min) Forecasts Remain Weak

A frequent question from thesis examiners is:  
*"Why does model performance drop sharply at 3 and 6 hours?"*

The answer is rooted in **fundamental atmospheric physics**, not a flaw in the neural network:
1. **Convective Storm Lifetimes:**  
   Individual convective cells (thunderstorms) over the Meghalaya plateau typically have a physical lifecycle of **30 to 90 minutes**. Predicting the exact location of a new convective cell 6 hours into the future using only satellite extrapolations exceeds the physical predictability limit.
2. **Lack of Numerical Weather Prediction (NWP) Boundary Forcing:**  
   Our platform is a **data-driven nowcasting platform** ingesting satellite proxy observations. It does not solve the 3D hydrostatic Navier-Stokes fluid equations, nor does it ingest synoptic-scale lateral boundary wind fields from global NWP models (like GFS or ECMWF).
3. **Chaotic Error Cascades:**  
   In spatiotemporal recurrence, small uncertainties in cloud advection compound nonlinearly over 12 autoregressive steps ($12 \times 30\text{ min} = 360\text{ min}$).

---

## 8. Why Better Weather Forecasts Do Not Automatically Mean Better Landslide Predictions

Examiners may ask:  
*"If you improved the rainfall CSI by $20\%$, does that mean landslide prediction improved by $20\%$?"*

The scientifically correct answer is **NO**, due to four critical decoupling factors:

1. **Spatial Scale Mismatch:**  
   Satellite weather nowcasting operates on a coarse $\approx 10\text{ km}$ ($25 \times 40$) grid. Landslides, however, are highly localized slope-failure events occurring at individual hillside scales ($10\text{ m} - 90\text{ m}$).
2. **Antecedent Hydrological Hysteresis:**  
   Slope failure is rarely triggered solely by instantaneous rain rates. It depends on pore-water pressure accumulation, soil saturation, and multi-day antecedent infiltration. Heavy rainfall over dry, unsaturated bedrock may cause zero failures, while moderate rainfall on a hillside saturated over the prior 14 days may trigger catastrophic slides.
3. **Geomorphological Predisposition:**  
   Landslides cannot occur without steep slope gradients, unstable lithology, and structural discontinuities, regardless of how intense the rainfall is.
4. **Historical Inventory Sparsity & Spatial Overfitting:**  
   Historical verified landslide catalogs in Meghalaya are sparse ($N=38$ events). When evaluated under spatial block cross-validation, static susceptibility models achieve an AUC of $0.5913 \pm 0.04$, showing that spatial generalization in complex Himalayan terrain remains challenging.
