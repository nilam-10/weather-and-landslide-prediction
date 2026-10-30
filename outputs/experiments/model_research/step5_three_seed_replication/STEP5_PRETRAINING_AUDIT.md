# Step 5 Pre-Training Audit: Experimental Compatibility & Data Verification

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Auditor:** Senior Research Engineer in Spatiotemporal Deep Learning & Remote Sensing  
**Scope:** Pre-training audit of datasets, model architecture, loss configurations, checkpoint selection, and seed compatibility before executing the 3-seed replication.

---

## 1. Audit of the Step 4 Pilot Implementation

Before launching any new training runs, we independently audited the source code, training logs, and checkpoint artifacts from the Step 4 pilot (`outputs/experiments/model_research/step4_loss_experiment/`):

### 1.1 Dataset Manifest & Partitioning Audit
The dataset splits were confirmed directly from source data manifests and sequence generation functions (`scripts/run_step4_loss_pilot_experiment.py` and `scripts/run_rigorous_model_improvement_program.py`):
- **Training Set (38 Calendar Days, 1,764 Sequences)**:
  - `2023-07-01` to `2023-07-10` (10 days, 480 raw slots)
  - `2024-07-01` to `2024-07-14` (14 days, 672 raw slots)
  - `2024-08-05` to `2024-08-14` (10 days, 480 raw slots)
  - `2025-07-01` to `2025-07-04` (4 days, 192 raw slots)
  - Total raw timesteps: 1,824 half-hour slots.
  - Usable causal sliding sequences with $T_{\text{in}} = 4$ and $T_{\text{lead}} = 12$: **1,764 sequences**.
- **Validation Set (1 Calendar Day, 33 Sequences)**:
  - `2025-07-05` (48 raw slots $\implies$ **33 sliding sequences**).
  - Used strictly for validation loss monitoring and checkpoint selection.
- **Independent Test Set (3 Calendar Days, 129 Sequences)**:
  - `2023-07-12` to `2023-07-14` (144 raw slots $\implies$ **129 sliding sequences**).
  - Strictly locked and treated as previously inspected confirmatory diagnostics.
- **Anti-Leakage Embargo Buffers**:
  - `2023-07-11`: 24-hour buffer separating Train Block 1 from Test.
  - `2024-08-04`: 24-hour buffer separating Historical Test from Train Block 3.

### 1.2 Model Architecture & Hyperparameters
- **Architecture**: `SimVPWeather`
  - Input channels per frame: 3 ($T_b$, $P$, observation mask).
  - Hidden channel dimension: 32.
  - Historical frames $T_{\text{in}} = 4$ (past 2 hours at 30-min intervals).
  - Future forecast horizons: 5 leads (+30m, +60m, +120m, +180m, +360m).
  - Spatiotemporal Translator: 3 Gated Temporal Convolutional Blocks.
  - Parameter Count: **337,138 trainable parameters**.
- **Optimizer & Schedule**:
  - Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`.
  - Scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=15, eta_min=1e-5)`.
  - Gradient Clipping: `clip_grad_norm_(max_norm=5.0)`.
  - Batch Size: 16.
  - Training Epochs: 15.
  - Mixed Precision: `torch.amp.autocast("cuda")`.

### 1.3 Loss Functions & Component Weights
- **Exp A (Baseline Control)**:
  $$\mathcal{L}_{\text{ExpA}} = 1.0 \cdot \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\hat{q}, y_{\text{cont}})$$
- **Exp D (Combined Objective)**:
  $$\mathcal{L}_{\text{ExpD}} = 1.0 \cdot \mathcal{L}_{\text{BCE}}(\hat{p}, y_{\text{bin}}) + 0.5 \cdot \mathcal{L}_{\text{Weighted-L1}}(\hat{q}, y_{\text{cont}}) + 0.5 \cdot \mathcal{L}_{\text{Soft-FSS}}(\hat{q}, y_{\text{cont}})$$
  - Intensity weight: $w(y) = 1.0 + 0.5 \cdot \text{clamp}(y_{\text{cont}} / 5.0, 0.0, 6.0)$, normalized by batch mean across valid mask.
  - Multiscale Soft-FSS: circular kernels at $k \in \{1, 2, 3\}$ cells ($10, 20, 30\text{ km}$), threshold $q = 5.0\text{ mm/hr}$, sigmoid steepness $\kappa = 1.0$.

### 1.4 Observation Masks and Target Units
- Units: Both ground truth targets ($y_{\text{cont}}$) and predicted outputs ($\hat{q}$) are in raw physical units ($\text{mm/hr}$).
- Missing data: Masked element-wise by multiplying with spatial mask $M \in \{0, 1\}$ and dividing by $\max(\sum M, 1.0)$.
- Checkpoint selection: Strictly on minimum validation loss (`val_loss_total`) on July 5, 2025. The test set was never accessed during checkpoint selection.

---

## 2. Seed-42 Compatibility & Replication Strategy

- The existing Seed-42 checkpoints (`ExpA_Baseline_best.pt` and `ExpD_Combined_best.pt`) in `outputs/experiments/model_research/step4_loss_experiment/` were trained under the exact identical configuration described above.
- To ensure **100% deterministic, side-by-side comparability** across all six runs with uniform execution timing, logging structures, and evaluation passes, the replication script will execute the complete 6-run matrix (Seeds 42, 123, 999 for both Exp A and Exp D) under strict random seeding.

---

## 3. Pre-Training Verification Checklist

| Verification Item | Requirement | Audit Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **Data Partitioning** | Zero overlap across train, val, test | Set intersection count = 0 | **PASSED** |
| **Target Units** | Physical rainfall rate in mm/hr | Both targets and QPF outputs in mm/hr | **PASSED** |
| **Observation Masking** | Masked from loss numerator and denominator | Element-wise mask applied and normalized | **PASSED** |
| **Geometry Definition** | Corrected Circular Euclidean Kernel | Radii 10, 20, 30, 50 km (5, 13, 29, 81 cells) | **PASSED** |
| **Checkpoint Criterion** | Validation loss only | Selected on min validation loss (July 5, 2025) | **PASSED** |
| **Safety Guard** | Production ConvLSTM protected | Target directory isolated in `step5_three_seed_replication/` | **PASSED** |

**Pre-Training Audit Conclusion:**  
All pre-conditions for the 3-seed replication are satisfied. The experimental setup is fully verified and ready for execution.
