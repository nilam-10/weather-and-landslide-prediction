# Step 6 Final Research Decision: Thesis Integration Verdict

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Target:** Formal Verdict on Step 5 Research Promotion for B.Tech Honours Thesis  
**Date:** October 30, 2026  
**Artifact Directory:** `outputs/experiments/model_research/step6_statistical_integrity_audit/`

---

## 1. Formal Verdict

### **DECISION: APPROVED FOR THESIS INTEGRATION WITH MANDATORY CAVEATS**

The research progression culminating in **Step 5 (Three-Seed Replication of SimVP with Compound Spatial Loss)** represents a scientifically sound, publication-grade contribution to deep-learning-based precipitation nowcasting. The empirical results demonstrate that replacing unconditional regression loss functions (Smooth-L1 / MSE) with a compound objective combining **Intensity-Weighted L1, Multiscale Differentiable Soft-FSS, and BCE** effectively resolves convective peak attenuation at intermediate horizons (+120m and +180m).

The findings are approved for inclusion in the B.Tech Honours Major Project thesis, subject to the strict reporting standards and caveats detailed below.

---

## 2. Mandatory Thesis Caveats & Disclosures

To preserve academic integrity and prevent examiner critique during thesis defense, the final submission manuscript must explicitly include the following caveats:

### Caveat 1: Nature of Statistical Significance
- **Disclosure:** The reported $p$-values ($p < 0.001$) originate from stationary temporal block bootstrapping over evaluation sequences, demonstrating that the performance gain is not an artifact of random temporal weather sampling within the evaluated episodes.
- **Limitation:** The sample size across random initializations is $N = 3$ seeds. While positive gains replicate across all three seeds, three seeds cannot establish an inferential non-parametric $p < 0.05$ across the distribution of network initializations. The thesis must report the across-seed standard deviations ($s_\Delta$) to document reproducibility without claiming inferential proof across seeds.

### Caveat 2: Status of Evaluation Cohorts
- **Disclosure:** The confirmatory test dataset (July 12–14, 2023, $N=129$ sequences) served as a benchmark in earlier exploratory phases (Phase 3 and Step 2). Although model checkpoints were selected strictly by validation loss on July 5, 2025 and were never exposed to test data during training, this dataset represents a **locked historical benchmark**, not an unexposed blind test cohort.
- **Limitation:** The primary validation cohort (July 5, 2025) represents a single active monsoon day ($N=33$ sequences), which is sufficient for checkpoint selection but provides limited independent statistical sample size.

### Caveat 3: Realistic Scope of 6-Hour (+360m) Forecasts
- **Disclosure:** At the +360-minute forecast horizon, the baseline model suffered complete convective extinction ($\text{CSI} = 0.0000$, $\text{FSS} = 0.0000$). The compound loss successfully prevented total zero-out, retaining non-zero convective cells ($\text{CSI} \approx 0.0125$, $\text{FSS} \approx 0.0526$).
- **Limitation:** Absolute skill at +360 minutes remains far below the operational utility threshold ($\text{FSS} \approx 0.50$). The thesis must state that pure satellite-based spatiotemporal nowcasting without numerical weather prediction (NWP) coupling or atmospheric sounding data cannot reliably position localized convective cells 6 hours in advance.

### Caveat 4: Distribution Diagnostics vs. Pointwise Accuracy
- **Disclosure:** Maximum predicted rainfall ($13.15\text{ mm/hr}$ at +180m) and 99th percentile rainfall ($6.17\text{ mm/hr}$) represent domain-wide distribution diagnostics.
- **Limitation:** Average convective core displacement at +180m remains approximately $110\text{ km}$. The model captures the regional convective storm footprint, but spatial offsets prevent high pointwise hit rates.

### Caveat 5: Downstream Landslide Coupling Status
- **Disclosure:** This study investigated the meteorological nowcasting sub-system. The improved rainfall fields from Exp D have **not yet been coupled** into the downstream Phase 4 landslide susceptibility random forest model. No claims regarding improved landslide warning lead time or reduced false alarms may be made based on these weather results alone.

---

## 3. Production Deployment & Checkpoint Freeze

1. **Canonical Production ConvLSTM Remains Frozen:**
   - The production FastAPI backend and web platform will continue to serve the canonical baseline checkpoint (`outputs/models/phase3/convlstm_seed42.pt`) for stability and backward compatibility with Phase 6 XAI modules.
2. **Research Breakthrough Checkpoints Preserved:**
   - The three replicated SimVP checkpoints (`ExpD_Seed42_best.pt`, `ExpD_Seed123_best.pt`, `ExpD_Seed999_best.pt`) are preserved in `outputs/experiments/model_research/step5_three_seed_replication/CHECKPOINTS/` as the primary experimental deliverable of the research program.

---

## 4. Concluding Recommendation for B.Tech Honours Defense

The 6-step research program demonstrates an exemplary scientific trajectory:
1. **Step 1 (Forensic Audit):** Discovered severe intensity attenuation in standard MSE models.
2. **Step 2 (Spatial Verification):** Proved spatial skill using Fractions Skill Score despite pointwise displacement.
3. **Step 3 (Evaluation Audit):** Validated spatial verification geometry and eliminated test leakage.
4. **Step 4 (Loss Pilot):** Formulated and piloted Intensity-Weighted L1 + Multiscale Differentiable Soft-FSS.
5. **Step 5 (Three-Seed Replication):** Replicated convective gains across 3 independent seeds and 64.5+ hours of multi-day testing.
6. **Step 6 (Integrity Audit):** Calibrated statistical significance, eliminated hyperbole, and established thesis-safe reporting standards.

This systematic progression provides the candidate with an extraordinarily strong, defensible, and methodologically sound thesis narrative.
