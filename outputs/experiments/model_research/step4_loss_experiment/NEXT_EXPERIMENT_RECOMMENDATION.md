# Next Experiment Recommendation: Three-Seed Replication Protocol

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Auditor:** Senior Research Engineer in Spatiotemporal Deep Learning & Remote Sensing  
**Target Architecture:** `SimVPWeather` (~337K parameters)  
**Hardware Budget:** NVIDIA RTX 4050 Laptop GPU (~6 GB VRAM)

---

## 1. Experimental Synthesis & Justification for Replication

The Step 4 pilot experiment (Seed 42) demonstrated that the **Combined Loss Objective (Exp D: BCE + 0.5 * Intensity-Weighted L1 + 0.5 * Multiscale Soft-FSS)** successfully resolves the primary defect in our rainfall prediction system:

1. **At $+120\text{m}$ (2 Hours)**:
   - Circular FSS ($30\text{ km}$, $\ge 5.0\text{ mm/hr}$) elevated from **$0.2962$** (Baseline) to **$0.6043$** (Exp D).
   - Pointwise CSI elevated from **$0.0853$** to **$0.2130$**.
2. **At $+180\text{m}$ (3 Hours)**:
   - Circular FSS elevated from **$0.0846$** to **$0.4120$** ($4.9\times$), crossing the $0.40$ threshold for regional utility.
   - Pointwise CSI elevated from **$0.0194$** to **$0.1184$** ($6.1\times$).
3. **At $+360\text{m}$ (6 Hours)**:
   - Completely broke the zero-skill collapse: Circular FSS elevated from **$0.0000$** to **$0.1467$** ($N\text{-CSI} = 0.2261$).
   - Convective peak intensity restored from $4.12\text{ mm/hr}$ to **$17.70\text{ mm/hr}$**.

Because these results were obtained on a single random seed (Seed 42), the next scientific imperative is to determine whether these performance gains replicate across multiple random initializations and data shuffling permutations.

---

## 2. Proposed Three-Seed Replication Protocol

### Configurations to Compare:
To maintain clean experimental focus, the 3-seed replication should compare:
1. **Control Baseline (Exp A)**: Canonical BCE + 0.5 * Smooth L1.
2. **Winning Candidate (Exp D)**: Combined BCE + 0.5 * Weighted-L1 + 0.5 * Multiscale Soft-FSS.
3. *(Optional Conservative Alternative)* **Exp C (Soft-FSS alone)**: For applications strictly requiring zero RMSE inflation.

### Seeds:
- `Seed 42`, `Seed 123`, `Seed 999`.

### Training Protocol:
- **Architecture**: `SimVPWeather` (in_channels=3, hidden_dim=32, t_in=4, num_leads=5, blocks=3).
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`).
- **Scheduler**: `CosineAnnealingLR(T_max=15, eta_min=1e-5)`.
- **Epochs**: 15 with early stopping / checkpoint selection on validation loss.
- **Precision**: Mixed precision (`torch.amp.autocast`).
- **Hardware Footprint**: Each 15-epoch run takes ~85 seconds on the RTX 4050 GPU. Total training time for 6 runs (2 models $\times$ 3 seeds) is **~8.5 minutes**, consuming $< 3\text{ GB}$ VRAM.

### Success Criteria for Final Thesis Adoption:
1. **Statistical Significance**: Paired block bootstrap test between Exp D and Exp A across the 3-seed ensemble must yield $p < 0.01$ at $+120\text{m}$ and $+180\text{m}$.
2. **Non-Zero 6-Hour Skill**: Exp D must maintain mean circular FSS $> 0.10$ and predicted maximum $> 12.0\text{ mm/hr}$ at $+360\text{m}$ across all three seeds.
3. **Short-Term Safety**: Mean $+30\text{m}$ RMSE must not exceed $0.80\text{ mm/hr}$ (baseline: $0.74\text{ mm/hr}$).

---

## 3. Strict Safety Safeguards

1. **No Overwrites**:
   All replication checkpoints and results will be written to `outputs/experiments/model_research/step4_loss_experiment/REPLICATION/`.
2. **Production Isolation**:
   The production checkpoint `outputs/models/phase3/convlstm_seed42.pt`, the FastAPI backend inference service, the Leaflet map frontend, and the landslide susceptibility model remain **strictly untouched**.

---

## 4. Current Status: Pilot Complete, Paused for Review

In strict adherence to your instructions:
- **The pilot experiment has been fully executed, verified, and logged.**
- **Zero production models have been changed.**
- **We have stopped all training and are waiting for your review and explicit direction before starting the 3-seed replication campaign.**
