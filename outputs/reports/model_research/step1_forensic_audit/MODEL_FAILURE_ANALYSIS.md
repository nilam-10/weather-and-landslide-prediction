# FORECAST QUALITY DEGRADATION & MODEL FAILURE ANALYSIS
## Comprehensive Empirical Investigation Across Horizons (+30m, +60m, +120m, +180m, +360m)

**Audit Conducted:** October 30, 2026  
**Auditor:** Senior Research Engineer in Spatiotemporal Deep Learning & Meteorology  
**Workspace:** `D:\Downloads\honours`  
**Evaluation Cohort:** Primary Independent Test Set (July 12–14, 2023; $N=129$ sequences; 129,000 grid points per lead time)  
**Evaluated Models:** Canonical ConvLSTM (161K params), Optimized SimVP (337K params), U-Net–ConvLSTM (270K params), Dense Optical Flow, Persistence Baseline  

---

## 1. EMPIRICAL DIAGNOSTIC MEASUREMENTS TABLE

All metrics computed directly from raw floating-point prediction tensors and verified ground-truth arrays:

| Horizon | Model | Max Prob | P99 Prob | N ($P \ge 0.5$) | Smoothness Ratio ($\frac{\|\nabla Q\|}{\|\nabla Y\|}$) | True Heavy Rain Mean (mm/hr) | Pred Heavy Rain Mean (mm/hr) | Heavy Rain Bias Ratio | TP ($\tau=0.5$) | FN ($\tau=0.5$) | CSI ($\tau=0.5$) | CSI ($\tau_{\text{val}}$) | Temporal Variance | Climatology Correlation ($r$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+30m** | ConvLSTM | 0.9989 | 0.8220 | 3,572 | 0.7361 | 7.83 | 6.13 | 78.3% | 2,830 | 1,472 | 0.5611 | 0.5564 | 1.8191 | 0.9931 |
| | SimVP | 1.0000 | 0.8512 | 3,701 | 0.7608 | 7.83 | 6.35 | 81.1% | 2,872 | 1,430 | 0.5597 | **0.5630** | 1.9729 | 0.9912 |
| **+60m** | ConvLSTM | 0.9894 | 0.6739 | 2,714 | 0.5713 | 7.85 | 4.69 | 59.7% | 1,886 | 2,318 | 0.3748 | 0.3892 | 1.3145 | 0.9822 |
| | SimVP | 0.9976 | 0.7240 | 2,961 | 0.5876 | 7.85 | 4.83 | 61.5% | 2,027 | 2,177 | **0.3945** | **0.4104** | 1.3528 | 0.9755 |
| **+120m**| ConvLSTM | 0.8366 | 0.4877 | 1,142 | 0.3321 | 7.89 | 2.43 | 30.8% | 629 | 3,433 | 0.1375 | 0.2353 | 0.5901 | 0.9496 |
| | SimVP | 0.9510 | 0.5642 | 1,822 | 0.3769 | 7.89 | 2.96 | 37.6% | 987 | 3,075 | **0.2016** | **0.2454** | 0.7626 | 0.9232 |
| **+180m**| ConvLSTM | 0.6834 | 0.3263 | 43 | 0.2244 | 7.91 | 1.33 | 16.8% | 24 | 3,962 | 0.0060 | 0.1635 | 0.2647 | 0.9199 |
| | SimVP | 0.8371 | 0.4954 | 1,246 | 0.2771 | 7.91 | 1.82 | 23.0% | 489 | 3,497 | **0.1031** | **0.1895** | 0.4267 | 0.8853 |
| **+360m**| ConvLSTM | 0.2840 | 0.1406 | **0** | **0.0981** | 7.92 | 0.39 | **4.9%** | **0** | 3,944 | **0.0000** | 0.0381 | **0.0259** | 0.8060 |
| | SimVP | 0.5665 | 0.2267 | 7 | 0.1415 | 7.92 | 0.49 | **6.2%** | 1 | 3,943 | **0.0003** | **0.0706** | **0.0572** | 0.7961 |

---

## 2. SYSTEMATIC ANALYSIS OF THE 10 FAILURE QUESTIONS

### Question 1: Do rainfall probabilities become too low?
**Verdict: YES (CONFIRMED).**  
- In ConvLSTM, the maximum predicted probability across all 129,000 spatial cells at +360m is **0.2840**. Because probabilities never reach 0.50, applying the default 0.50 classification threshold produces **exactly 0 True Positives, 0 False Positives, and 3,944 False Negatives**, mechanically yielding $\text{CSI} = 0.0000$.
- In SimVP, maximum predicted probability reaches **0.5665**, but only 7 pixels exceed 0.50.
- **Root Cause**: Probability shrinkage is mathematically normative under high epistemic uncertainty. The climatological base rate of heavy rain is only 3.54%. A well-calibrated neural network outputting posterior probability shrinks toward the base rate when input satellite signals become uninformative.

### Question 2: Does predicted rainfall become excessively smooth?
**Verdict: YES (CONFIRMED).**  
- The spatial gradient magnitude ratio ($\frac{\text{Mean}(\|\nabla Q_{\text{pred}}\|)}{\text{Mean}(\|\nabla Q_{\text{true}}\|)}$) drops from **0.7361 at +30m** to **0.0981 at +360m** for ConvLSTM (and 0.1415 for SimVP).
- At 6 hours, predicted fields retain less than 10% of the true spatial sharpness, creating washed-out, diffuse clouds that lack convective boundary edges.
- **Root Cause**: Deterministic regression losses (Smooth L1, MSE) penalize spatial displacement errors twice. To minimize expected loss under spatial uncertainty, the network predicts the conditional mean $\mathbb{E}[Y | X]$, which is inherently a smoothed average.

### Question 3: Is heavy rainfall systematically underestimated?
**Verdict: YES (CONFIRMED & QUANTIFIED).**  
- On actual heavy convective rain cells ($\ge 5.0\text{ mm/hr}$, where ground truth averages $7.8\text{ to }7.9\text{ mm/hr}$):
  - At +30m: Model predicts $6.13\text{ mm/hr}$ (78.3% of true intensity).
  - At +60m: Model predicts $4.69\text{ mm/hr}$ (59.7% of true intensity).
  - At +120m: Model predicts $2.43\text{ mm/hr}$ (30.8% of true intensity).
  - At +180m: Model predicts $1.33\text{ mm/hr}$ (16.8% of true intensity).
  - At +360m: Model predicts **$0.39\text{ mm/hr}$ (only 4.9% of true intensity)**!
- On extreme rainfall cells ($\ge 10.0\text{ mm/hr}$, true average $13.59\text{ mm/hr}$), the 6-hour prediction is also $0.39\text{ mm/hr}$, underestimating peak intensity by **97.1%**.

### Question 4: Are storm locations displaced?
**Verdict: YES (CONFIRMED BY DOUBLE-PENALTY EFFECT).**  
- When decision thresholds are tuned on validation data to restore positive detections ($\tau_{\text{val}} = 0.02$ for ConvLSTM, $\tau_{\text{val}} = 0.06$ for SimVP):
  - POD reaches $0.8202$ (ConvLSTM) and $0.3385$ (SimVP).
  - However, False Alarm Ratio (FAR) surges to **96.16%** for ConvLSTM and **91.82%** for SimVP.
  - Even the persistence baseline suffers an FAR of **89.91%** at +360m.
- This demonstrates that storm clusters are predicted in the general geographic vicinity (e.g. over the southern plateau), but slight spatial displacement of 10 to 30 km causes grid-cell misses ($10 \times 10\text{ km}$ cells).

### Question 5: Does the model miss newly developing rainfall?
**Verdict: YES (CONFIRMED AS THE PRIMARY PHYSICAL BOTTLENECK).**  
- Atmospheric convective thunderstorm cells over the Khasi and Garo Hills have typical lifespans of **30 to 90 minutes**.
- At +180m (3 hours) and +360m (6 hours), the convective raincells that exist in reality did not exist at observation time $t_0$.
- Because the input tensor contains only past satellite infrared brightness temperature and precipitation (with zero thermodynamic sounding data such as CAPE, convective inhibition, or 850 hPa wind shear), the model has **no physical mechanism to predict convective initiation** out of clear air. It can only advect existing cloud systems until they dissipate.

### Question 6: Do predictions collapse toward background or climatology?
**Verdict: YES (CONFIRMED WITH TEMPORAL VARIANCE PROOF).**  
- **Temporal variance of predicted precipitation drops by 98.6%**:
  - +30m temporal variance: **1.8191**
  - +60m temporal variance: **1.3145**
  - +120m temporal variance: **0.5901**
  - +180m temporal variance: **0.2647**
  - +360m temporal variance: **0.0259** (near stationary)
- Meanwhile, the spatial correlation with the static time-mean climatological rain map remains very high ($r = 0.8060$ to $0.9931$).
- At +360m, the model has ceased predicting dynamic storm motion; it outputs a stationary orographic background map with slightly elevated rain rates over Cherrapunji and zero rain elsewhere.

### Question 7: Do false alarms or missed events dominate?
**Verdict: REGIME-DEPENDENT (PROVED).**  
- **At fixed threshold $\tau=0.50$**: False Negatives (Missed Events) completely dominate. At +360m, FN = 3,944 while FP = 0.
- **At validation-tuned threshold $\tau=0.06$**: False Alarms dominate due to double-penalty spatial displacement (FP = 1,748 to 81,047 depending on model aggressiveness).

### Question 8: Do rainfall-rate regression and rainfall-event classification disagree?
**Verdict: NO (THEY AGREE IN LOCKSTEP).**  
- Overall pixel-by-pixel agreement between $P(\text{Heavy Rain}) \ge 0.50$ and $\text{QPF} \ge 5.0\text{ mm/hr}$ is **99.64% to 100.00%** across all five horizons.
- The dual heads degrade synchronously; regression does not predict 15 mm/hr while classification predicts 0.1 probability. Both heads collapse together.

### Question 9: Does performance differ across rainfall intensity categories?
**Verdict: YES (CONFIRMED).**  
- Widespread light stratiform rain (0.1–2.0 mm/hr) is predicted with acceptable correlation ($r \approx 0.60$ at +120m).
- Heavy convective cores ($\ge 5.0\text{ mm/hr}$) suffer immediate intensity attenuation.
- Extreme cloudbursts ($\ge 10.0\text{ mm/hr}$) are virtually eliminated beyond +60m.

### Question 10: Is the model learning useful temporal changes or reproducing common background patterns?
**Verdict: TEMPORAL LEARNING DOMINATES AT +30m/+60m; BACKGROUND CLIMATOLOGY DOMINATES AT +180m/+360m.**  
- At +30m and +60m, the model beats persistence and optical flow by substantial margins (ConvLSTM CSI 0.561 vs 0.446 for optical flow; SimVP CSI 0.395 vs 0.278 for optical flow), proving active learned spatiotemporal advection.
- At +360m, temporal variance drops to 0.0259, and the model merely reproduces the time-invariant orographic background pattern.
