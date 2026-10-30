# Comprehensive Model Architectures Specification (Audited)

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Institutional Context**: RBU Honours Major Project (2026–27)  
**Audit Status**: Verified against actual PyTorch checkpoints and state dictionaries.

---

## 1. Module A: Convective Weather Nowcasting Architectures

### 1.1 Model 1: Spatiotemporal 2D ConvLSTM (Shi et al., NeurIPS 2015)

#### Verified Implementation & Discrepancy Resolution
- **Audit Finding**: An earlier documentation draft reported 69,410 parameters based on an unverified theoretical `(32, 16)` hidden-channel specification with 2 input channels.
- **Physical Reality**: The trained model instantiated in `scripts/train_weather_convlstm.py` and saved in `outputs/models/weather_convlstm/best_convlstm_model.pt` uses:
  - Input channels: $C = 3$ (MERGIR IR Brightness Temperature $T_b$, IMERG Precipitation Rate $P$, Satellite Availability Mask $M$)
  - Hidden channels: `(32, 32)` across both recurrent layers
  - Dual prediction heads: Conv2d(32 -> 16) -> BatchNorm2d(16) -> ReLU -> Dropout(0.1) -> Conv2d(16 -> 1)
  - **Exact Programmatic Parameter Count**: **123,650 trainable parameters** (123,716 total state dictionary elements including non-trainable BatchNorm running buffers).

```
Input Sequence: (B, T_in=4, C=3, H=25, W=40)
  Channel 0: Normalized MERGIR Cloud-Top Brightness Temperature (Tb)
  Channel 1: Normalized IMERG Precipitation Rate (P)
  Channel 2: Satellite Observation Availability Mask (M)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ ConvLSTM Layer 1                                            │
│  - Input Channels: 3                                        │
│  - Hidden Channels: 32                                      │
│  - Kernel Size: 3x3 (Padding: 1, Stride: 1)                 │
│  - Gates: 4x (i, f, g, o) via Conv2d(3+32 -> 128, 3x3)      │
│  - Parameters: 40,320 (weight) + 128 (bias) = 40,448        │
│  - Output State H1_t: (B, 32, 25, 40)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ ConvLSTM Layer 2                                            │
│  - Input Channels: 32                                       │
│  - Hidden Channels: 32                                      │
│  - Kernel Size: 3x3 (Padding: 1, Stride: 1)                 │
│  - Gates: 4x (i, f, g, o) via Conv2d(32+32 -> 128, 3x3)     │
│  - Parameters: 73,728 (weight) + 128 (bias) = 73,856        │
│  - Output State H2_t: (B, 32, 25, 40)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐┌──────────────────────────────┐
│ Head +30m: Conv2d Head       ││ Head +60m: Conv2d Head       │
│  - Conv2d(32 -> 16, k=3, p=1)││  - Conv2d(32 -> 16, k=3, p=1)│
│    (4,608 wt + 16 bias)      ││    (4,608 wt + 16 bias)      │
│  - BatchNorm2d(16)           ││  - BatchNorm2d(16)           │
│    (16 gamma + 16 beta)      ││    (16 gamma + 16 beta)      │
│  - ReLU() + Dropout(0.1)     ││  - ReLU() + Dropout(0.1)     │
│  - Conv2d(16 -> 1, k=1)      ││  - Conv2d(16 -> 1, k=1)      │
│    (16 wt + 1 bias)          ││    (16 wt + 1 bias)          │
│  - Sigmoid() -> (B, 1, 25, 40││  - Sigmoid() -> (B, 1, 25, 40│
│  - Parameters: 4,673         ││  - Parameters: 4,673         │
└──────────────────────────────┘└──────────────────────────────┘
Total Trainable Parameters: 40,448 + 73,856 + 4,673 + 4,673 = 123,650
```

#### Layer-by-Layer Verification Table

| Component | Layer Name | Tensor Shapes | Trainable Params | Non-Trainable Elements | Total Elements |
|---|---|---|---|---|---|
| **Recurrent Cell 1** | `cell1.conv.weight` | `[128, 35, 3, 3]` | 40,320 | 0 | 40,320 |
| | `cell1.conv.bias` | `[128]` | 128 | 0 | 128 |
| **Recurrent Cell 2** | `cell2.conv.weight` | `[128, 64, 3, 3]` | 73,728 | 0 | 73,728 |
| | `cell2.conv.bias` | `[128]` | 128 | 0 | 128 |
| **Head 0 (+30m)** | `lead_heads.0.0.weight` | `[16, 32, 3, 3]` | 4,608 | 0 | 4,608 |
| | `lead_heads.0.0.bias` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.0.1.weight` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.0.1.bias` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.0.1.running_mean` | `[16]` | 0 | 16 | 16 |
| | `lead_heads.0.1.running_var` | `[16]` | 0 | 16 | 16 |
| | `lead_heads.0.1.num_batches` | `[]` | 0 | 1 | 1 |
| | `lead_heads.0.4.weight` | `[1, 16, 1, 1]` | 16 | 0 | 16 |
| | `lead_heads.0.4.bias` | `[1]` | 1 | 0 | 1 |
| **Head 1 (+60m)** | `lead_heads.1.0.weight` | `[16, 32, 3, 3]` | 4,608 | 0 | 4,608 |
| | `lead_heads.1.0.bias` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.1.1.weight` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.1.1.bias` | `[16]` | 16 | 0 | 16 |
| | `lead_heads.1.1.running_mean` | `[16]` | 0 | 16 | 16 |
| | `lead_heads.1.1.running_var` | `[16]` | 0 | 16 | 16 |
| | `lead_heads.1.1.num_batches` | `[]` | 0 | 1 | 1 |
| | `lead_heads.1.4.weight` | `[1, 16, 1, 1]` | 16 | 0 | 16 |
| | `lead_heads.1.4.bias` | `[1]` | 1 | 0 | 1 |
| **Total** | | | **123,650** | **66** | **123,716** |

---

### 1.2 Model 2: Optical Flow (rainymotion / Lucas-Kanade Extrapolation)
- Dense gradient-based optical flow computing velocity vectors $(u, v)$ from consecutive IMERG frames $(t-1, t)$.
- Semi-Lagrangian backward warping extrapolation: $\hat{I}_{t+\tau}(x, y) = I_t(x - u \cdot \tau, y - v \cdot \tau)$.
- Zero learnable parameters; fully deterministic physical advection benchmark.
- Evaluated across 4 horizons: $+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}$.

---

### 1.3 Model 3: Tabular PyTorch MLP Baseline
- Architecture: Linear(6 -> 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 32) -> BatchNorm1d(32) -> ReLU -> Linear(32 -> 1)
- Total parameters: 2,786.

---

### 1.4 Model 4: Tabular Logistic Regression Baseline
- Architecture: Generalized Linear Model with L2 penalty ($C=1.0$), balanced class weighting.
- Total parameters: 7.

---

## 2. Module B: Landslide Hazard Modeling Architectures

### 2.1 Model 5: Static XGBoost Landslide Susceptibility Model
- Framework: `xgboost` v3.4.1.
- Input Features ($D=8$ from 90m SRTM DEM): `elevation_mean`, `slope_mean`, `slope_max`, `relief`, `slope_std`, `aspect_northness`, `aspect_eastness`, `curvature_mean`.
- Hyperparameters: `max_depth=3`, `n_estimators=60`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`.
- Total Trees: 60 regression trees (approx 480 parameters).

### 2.2 Model 6: Custom Rainfall-Susceptibility Decision Matrix (LHASA-Inspired Heuristic)
- **Scientific Re-Scoping**: This module is a **custom heuristic decision matrix** inspired by NASA LHASA principles, coupling 90m static terrain susceptibility with multi-day antecedent IMERG precipitation ($A = 0.6 P_{24\text{h}} + 0.4 P_{72\text{h}}$).
- **Explicit Research Caveat**: It does NOT reproduce the full machine-learned NASA LHASA 2.0 global model (which uses a 7-day weighted exponent $\sim 0.8$ and calibrated regional thresholds). It serves as an exploratory situational awareness prototype, NOT an operational warning system or an empirical landslide probability.
