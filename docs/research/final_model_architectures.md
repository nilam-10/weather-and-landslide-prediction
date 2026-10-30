# Final Verified Model Architectures Specification

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Institutional Context**: RBU Honours Major Project (2026–27)  
**Study Domain**: Meghalaya, Northeast India ($89.5^\circ\text{–}93.5^\circ\text{E}, 24.5^\circ\text{–}27.0^\circ\text{N}$)  
**Hardware Accelerator**: NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`, 6,140.5 MiB VRAM)  
**Audit Status**: Verified against physical PyTorch checkpoints, state dictionaries, and execution logs.

---

## 1. Convective Weather Nowcasting Models

### 1.1 Multi-Horizon Weather ConvLSTM (Shi et al., NeurIPS 2015 Adaptation)
- **Architecture**: 2-Layer Spatiotemporal 2D Convolutional LSTM with 5 independent prediction heads.
- **Input Channels ($C=3$)**:
  1. `MERGIR IR Tb`: Normalized geostationary brightness temperature (~4 km resolution).
  2. `IMERG Precipitation P`: Normalized half-hourly precipitation rate (0.1° resolution).
  3. `Satellite Mask M`: Observation gap indicator ($1 = \text{valid}, 0 = \text{gap}$).
- **Sequence Length**: $T_{\text{in}} = 4$ past consecutive frames ($t-3, t-2, t-1, t$, covering 2 hours).
- **Recurrent Layer 1**: `ConvLSTMCell(3 -> 32, kernel=3, padding=1)`. Weight: `[128, 35, 3, 3]`, Bias: `[128]`. Parameters = 40,448.
- **Recurrent Layer 2**: `ConvLSTMCell(32 -> 32, kernel=3, padding=1)`. Weight: `[128, 64, 3, 3]`, Bias: `[128]`. Parameters = 73,856.
- **5 Prediction Heads**:
  Each head maps top hidden state $H2_T \in \mathbb{R}^{B \times 32 \times 25 \times 40}$ through:
  `Conv2d(32 -> 16, k=3, p=1)` -> `BatchNorm2d(16)` -> `ReLU()` -> `Dropout2d(0.1)` -> `Conv2d(16 -> 1, k=1)`.
  Parameters per head: $4,608 + 32 + 16 + 1 = 4,673$.
  Total across 5 heads: $5 \times 4,673 = 23,365$.
- **Total Trainable Parameters**:
  $$40,448 + 73,856 + 23,365 = \mathbf{137,669 \text{ parameters}}$$
- **Optimization**: AdamW (lr=0.002, weight_decay=1e-4), compound weighted BCE loss ($w_{\text{pos}} \approx 20$).
- **Probability Calibration**: Validation Platt scaling ($z_{\text{cal}} = a \cdot z + b$) reducing Brier score by 74.3% (0.0322 down to 0.0083) and ECE from 0.0714 down to 0.0027.

### 1.2 Semi-Lagrangian Optical Flow (Lucas-Kanade Extrapolation)
- **Method**: Dense gradient-based optical flow estimating displacement field $(u, v)$ from consecutive IMERG frames $I_{t-1}, I_t$.
- **Extrapolation**: Backward advection with bilinear interpolation and spatial Gaussian smoothing.
- **Parameters**: 0 (deterministic physical kinematic baseline).
- **Skill Horizon**: Demonstrates strong skill at $+30\text{m}$ (CSI = 0.3945) and $+60\text{m}$ (CSI = 0.2749), rapidly dissipating at $+120\text{m}$ (CSI = 0.1146) and $+180\text{m}$ (CSI = 0.0483).

### 1.3 Lagged Persistence Baseline
- **Method**: $P_{t+k} = P_t$ (zero-velocity Eulerian persistence).
- **Parameters**: 0.
- **Skill**: CSI drops from 0.3928 at $+30\text{m}$ to 0.0082 at $+360\text{m}$.

### 1.4 Tabular Baselines (+30m)
- **Regularized Logistic Regression**: L2 penalty ($C=1.0$), standardized pixel features ($T_b, P, M$). Parameters = 4. Test CSI = 0.3857.
- **PyTorch 3-Layer MLP**: Linear(3 -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 1). Parameters = 2,369. Test CSI = 0.3882.

---

## 2. Geomorphic Landslide Susceptibility Models (LSM)

### 2.1 Gradient Boosted Decision Trees (XGBoost)
- **Input Dimension**: 8 morphometric derivatives extracted from 90m SRTM DEM:
  - `elevation_mean`, `slope_mean`, `slope_max`, `relief`, `slope_std`, `aspect_northness`, `aspect_eastness`, `curvature_mean`.
- **Hyperparameters**: `n_estimators=50`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Validation**: 3-Fold Spatial Block Cross-Validation (West, Central, East blocks).
- **Symmetric Unbuffered Metric**: Mean ROC-AUC = **0.5913**, PR-AUC = **0.2285**.
- **Location-Uncertainty Filtered (<=5km)**: Mean ROC-AUC = **0.6131**, PR-AUC = **0.2407**.

### 2.2 Regularized Logistic Regression
- **Hyperparameters**: $C=1.0$, L2 penalty, Standardized features.
- **Symmetric Unbuffered Metric**: Mean ROC-AUC = **0.5379**, PR-AUC = **0.2554**.

---

## 3. Dynamic Hazard & Temporal Forecasting Audit

### 3.1 Custom Dynamic Antecedent Rainfall-Susceptibility Decision Matrix
- **Antecedent Formula**: $A = 0.6 \times P_{24\text{h}} + 0.4 \times P_{72\text{h}}$ (accumulated mm from IMERG V07B).
- **Coupling Strategy**: Categorical 2D Decision Matrix coupling static LSI tiers with antecedent rainfall tiers (Normal $<50$mm, Elevated $50\text{--}100$mm, Extreme $>100$mm).
- **Designation**: Author-Calibrated Decision Matrix (inspired by NASA LHASA principles, but distinct from global ML LHASA 2.0).

### 3.2 Supervised 24h Landslide Prediction (REQ-LND-04): BLOCKED_BY_DATA
- **Audit Result**: Positive class prevalence across cell-day decision space is $6.12 \times 10^{-6}$ ($< 1 \text{ in } 160,000$).
- **Integrity Directive**: Classified definitively as `BLOCKED_BY_DATA` rather than simulating false skill or misrepresenting static models as temporal forecasts.
