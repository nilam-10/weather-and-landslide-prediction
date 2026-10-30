# Step 6 Audit: Thesis-Safe Claims and Scientific Language Guide

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Target:** Scientific Claim Classification and B.Tech Honours Thesis Reporting Standard  
**Audit Date:** October 30, 2026  
**Artifact Directory:** `outputs/experiments/model_research/step6_statistical_integrity_audit/`

---

## 1. Classification of Major Experimental Claims

Every major claim made in Step 4 and Step 5 has been audited against the underlying source code, checkpoints, and recalculated datasets. Claims are classified into one of four formal categories:
1. **Verified Directly from Source Data:** Mathematically and forensically confirmed with zero discrepancies.
2. **Supported but Limited:** Factually supported by the evidence, but subject to specific methodological or sample limitations.
3. **Not Independently Established:** Requires additional evidence, larger sample size, or untouched cohorts before it can be asserted.
4. **Incorrect or Inconsistent:** Methodologically flawed, statistically uncalibrated, or misstated.

---

### Master Claim Classification Table

| # | Specific Research Claim | Formal Classification | Evidence & Methodological Justification |
| :---: | :--- | :---: | :--- |
| **1** | **Combined loss (Exp D) restores convective skill at +120m and +180m over baseline (Exp A).** | **Verified Directly from Source Data** | Reconciled across both validation and test sets. At +180m, pointwise CSI increases from $0.0183 \to 0.1111$ on Val ($6.07\times$) and $0.0120 \to 0.1300$ on Test ($10.8\times$). FSS ($30\text{ km}$) increases from $0.0740 \to 0.4016$ on Val and $0.0527 \to 0.4333$ on Test. |
| **2** | **Convective gains at +120m and +180m replicate across Seeds 42, 123, and 999.** | **Verified Directly from Source Data** | In all 3 seeds, paired differences $\Delta_s = \text{ExpD}_s - \text{ExpA}_s$ are strictly positive at +120m and +180m for both CSI and FSS. Standard deviation across seeds is small ($\sigma_\Delta = 0.0089$ for CSI, $\sigma_\Delta = 0.0055$ for FSS at +180m). |
| **3** | **Exp D maintains bulk regression RMSE parity with Exp A.** | **Verified Directly from Source Data** | At +180m, RMSE on Val is $2.0145\text{ mm/hr}$ (Exp D) vs $2.0396\text{ mm/hr}$ (Exp A). On Test, RMSE is $1.5216\text{ mm/hr}$ (Exp D) vs $1.5224\text{ mm/hr}$ (Exp A). Differences are statistically insignificant ($p > 0.40$). |
| **4** | **Exp D cures heavy-rainfall intensity attenuation.** | **Supported but Limited** | Exp D retains higher forecast maxima ($13.15\text{ mm/hr}$ vs $7.06\text{ mm/hr}$ at +180m) and higher 99th percentiles ($6.17\text{ mm/hr}$ vs $3.05\text{ mm/hr}$). **Limitation:** Maxima are distribution-level diagnostics over all cells/times, not proof that a specific storm cell was predicted at its exact peak location. |
| **5** | **Exp D solves 6-hour (+360m) convective nowcasting.** | **Incorrect or Inconsistent** | While Exp D prevents complete zero-out (achieving $\text{CSI} = 0.0037$ on Val, $0.0125$ on Test vs baseline $0.0000$), absolute skill remains extremely low ($\text{FSS} = 0.0193$ to $0.0526$, well below the useful threshold of $0.50$). It represents a detectable mathematical signal, not an operational forecast. |
| **6** | **Step 5 established statistical significance across random initializations ($p = 0.000$).** | **Incorrect or Inconsistent** | With $N=3$ seeds, an exact sign test has minimum $p = (1/2)^3 = 0.125$. The bootstrap tested *temporal sampling uncertainty* for fixed models, not across-seed variance. Furthermore, empirical Monte Carlo with $B=1,000$ cannot yield exact $p=0.000$; it must be bounded ($p < 0.001$). |
| **7** | **The confirmatory test set (July 12–14, 2023) proves universal generalization.** | **Not Independently Established** | July 12–14, 2023 is a previously inspected 3-day monsoon event used throughout Phase 3 and Steps 2–5. While checkpoints were frozen without seeing test data, universal generalization cannot be claimed from a single 3-day meteorological episode. |
| **8** | **These weather forecasting improvements improve landslide risk prediction.** | **Not Independently Established** | The weather model outputs have not yet been coupled into the downstream Phase 4 random forest landslide susceptibility engine. No coupled landslide risk metrics were evaluated in Step 5. |

---

## 2. Thesis-Safe Language Guidelines

### 2.1 Prohibited Words and Phrases
To maintain scientific credibility and prevent examiner pushback during thesis defense, the following hyperbolic terms must **NOT** be used:

| Prohibited Term | Why It Is Unjustified | Approved Scientific Alternative |
| :--- | :--- | :--- |
| *"Unequivocally proven"* | Scientific deep learning models are empirically validated, not mathematically proven. | *"Consistently observed across the three evaluated random initializations"* |
| *"Solved the 6-hour forecast problem"* | Absolute skill at +360m is $\text{CSI} \le 0.013$ and $\text{FSS} \le 0.053$. | *"Mitigated complete convective zero-out at +360m, though absolute skill remains low"* |
| *"Proved statistical significance across random seeds"* | $N=3$ seeds is insufficient for an inferential hypothesis test across initializations. | *"Demonstrated consistent positive paired gains across three random seeds, with temporal sampling significance confirmed by block bootstrapping"* |
| *"p = 0.000"* | Empirical Monte Carlo with $B=1,000$ has finite resolution. | *"$p < 0.001$"* or *"$p \le 1/B$"* |
| *"Pristine / completely untouched test dataset"* | July 12–14, 2023 was evaluated in earlier project phases. | *"Locked confirmatory test benchmark"* |
| *"Operational readiness"* | Pure satellite nowcasting without NWP or Doppler radar is not ready for life-critical deployment. | *"Research prototype suitable for benchmark comparison"* |

---

## 3. Recommended Wording for Thesis Chapters

### 3.1 On Intermediate-Horizon Skill (+120m and +180m)
> **Recommended Thesis Wording:**  
> *"Incorporating the compound spatial loss objective (Intensity-Weighted L1 combined with Multiscale Differentiable Soft-FSS and BCE) systematically improved intermediate-range convective nowcasting performance compared to the standard Smooth-L1 baseline. At the +180-minute forecast horizon, pointwise Critical Success Index (CSI at $\ge 5.0\text{ mm/hr}$) increased from a baseline mean of $0.0183 \pm 0.0107$ to $0.1111 \pm 0.0184$ on the primary validation episode, and from $0.0120 \pm 0.0132$ to $0.1300 \pm 0.0231$ on the 3-day confirmatory test benchmark. Spatial verification using the Fractions Skill Score at a 30 km neighborhood radius confirmed a substantial improvement in spatial event placement ($0.0527 \to 0.4333$ on test), demonstrating that the compound objective successfully penalizes spatial displacement without degrading bulk precipitation root-mean-squared error."*

---

### 3.2 On Reproducibility Across Random Seeds
> **Recommended Thesis Wording:**  
> *"To ensure that observed convective gains were not artifacts of random weight initialization, a balanced $2 \times 3$ factorial replication experiment was conducted across three random seeds (Seeds 42, 123, and 999). Paired differences ($\Delta_s = \text{ExpD}_s - \text{ExpA}_s$) were positive across all three seeds at +120m and +180m for both pointwise CSI and circular FSS. At +180m, the paired CSI improvement averaged $+0.0928$ with an across-seed standard deviation of $s_\Delta = 0.0089$ on validation, and $+0.1180$ with $s_\Delta = 0.0235$ on the confirmatory test cohort, confirming consistent empirical replication across initializations."*

---

### 3.3 On Statistical Significance and Uncertainty
> **Recommended Thesis Wording:**  
> *"Statistical uncertainty was assessed using a stationary temporal block bootstrap ($B = 1,000$ resamples) with block lengths matching the forecast decorrelation window ($L=12$ sequences = 6.0 hours on the test cohort). For individual models and the pooled three-seed ensemble, the 95% bootstrap confidence intervals for paired CSI and FSS gains at +120m and +180m strictly excluded zero ($p < 0.001$). This inferential test specifically quantifies uncertainty with respect to temporal meteorological episode sampling, whereas across-seed variance is documented through the three-seed ensemble standard deviations."*

---

### 3.4 On Heavy-Rainfall Intensity Distribution
> **Recommended Thesis Wording:**  
> *"Intensity distribution diagnostics revealed that standard Smooth-L1 optimization induces severe conditional mean blur, attenuating 6-hour forecast maxima to below $2.0\text{ mm/hr}$ and reducing convective cell counts ($\ge 5.0\text{ mm/hr}$) by over 95%. In contrast, the compound loss retained realistic distribution tails, preserving +180m maximum rain rates of $13.15\text{ mm/hr}$ and restoring 99th percentile rainfall from $3.05\text{ mm/hr}$ to $6.17\text{ mm/hr}$. These intensity metrics represent domain-wide distribution diagnostics rather than exact pointwise storm peak predictions, as average convective core displacement at +180m remained approximately 110 km."*

---

### 3.5 On Long-Horizon (+360m) Limitations
> **Recommended Thesis Wording:**  
> *"At the 6-hour forecast horizon (+360 minutes), the baseline model suffered complete convective extinction ($\text{CSI} = 0.0000$, $\text{FSS} = 0.0000$). While the compound loss restored non-zero convective structure ($\text{CSI} = 0.0125 \pm 0.0129$, $\text{FSS} = 0.0526 \pm 0.0495$), absolute forecasting skill at this horizon remains modest and falls below the operational utility threshold ($\text{FSS} \approx 0.50$). This reflects the intrinsic physical limit of spatiotemporal extrapolation from infrared and satellite precipitation without explicit thermodynamic sounding data or numerical weather prediction coupling."*

---

### 3.6 On the Status of the Evaluation Cohort
> **Recommended Thesis Wording:**  
> *"Model selection was governed strictly by total validation loss on an independent active monsoon episode (July 5, 2025). Confirmatory evaluation was conducted on a locked 3-day benchmark dataset (July 12–14, 2023, 129 sequences). Although this benchmark cohort was evaluated in previous architecture comparisons, model checkpoints were frozen prior to evaluation and were never accessed during training, stopping criteria, or loss hyperparameter tuning."*
