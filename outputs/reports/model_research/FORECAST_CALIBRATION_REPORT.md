# FORECAST CALIBRATION & CROSS-EPISODE ROBUSTNESS REPORT

**Study Area:** Meghalaya Plateau ($25 	imes 40$ Grid)  
**Evaluated Architecture:** SimVP (loss_baseline)  
**Date:** October 29, 2026  

---

## 1. CALIBRATION PERFORMANCE & EXPECTED CALIBRATION ERROR (ECE)

| Forecast Horizon | Brier Score | Expected Calibration Error (ECE) | Val-Tuned Threshold ($	au$) | Maximum Predicted Prob | Reliability Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **+30m** | 0.013164 | 0.0036 | 0.36 | 1.0000 | Well-Calibrated |
| **+60m** | 0.018332 | 0.0034 | 0.24 | 0.9966 | Well-Calibrated |
| **+120m** | 0.024187 | 0.0053 | 0.20 | 0.9442 | Well-Calibrated |
| **+180m** | 0.027166 | 0.0046 | 0.16 | 0.7988 | Well-Calibrated |
| **+360m** | 0.029407 | 0.0067 | 0.06 | 0.5359 | Well-Calibrated |

---

## 2. CROSS-EPISODE GENERALIZABILITY COMPARISON

Evaluated across two completely independent historical monsoon episodes:
- **Episode 1:** July 12–14, 2023 ($N=129$ sequences, active monsoon surge)
- **Episode 2:** August 1–3, 2024 ($N=129$ sequences, widespread orographic depression)

| Metric | Horizon | Episode 1 (July 2023) | Episode 2 (August 2024) | Generalization Retention |
| :--- | :---: | :---: | :---: | :---: |
| **CSI (Val-Tuned)** | +30m | 0.5655 | 0.5824 | 103.0% |
| **CSI (Val-Tuned)** | +60m | 0.4112 | 0.4262 | 103.6% |
| **CSI (Val-Tuned)** | +120m | 0.2506 | 0.2604 | 103.9% |
| **CSI (Val-Tuned)** | +180m | 0.1795 | 0.1893 | 105.5% |
| **CSI (Val-Tuned)** | +360m | 0.0670 | 0.0878 | 131.0% |
| **QPF RMSE (mm/hr)** | +30m | 0.7382 | 0.9831 | Diff: 0.2449 |
| **QPF RMSE (mm/hr)** | +60m | 1.0122 | 1.3381 | Diff: 0.3259 |
| **QPF RMSE (mm/hr)** | +120m | 1.3396 | 1.8049 | Diff: 0.4653 |
| **QPF RMSE (mm/hr)** | +180m | 1.4994 | 2.0276 | Diff: 0.5282 |
| **QPF RMSE (mm/hr)** | +360m | 1.6662 | 2.2511 | Diff: 0.5849 |

---

## 3. SCIENTIFIC TAKEAWAYS ON CALIBRATION & 6-HOUR PROBABILITY DECAY
1. **Root Cause of Probability Decay:** At +360m (6 hours), the model appropriately expresses epistemic uncertainty by shrinking probabilities toward the climatological mean (~3%).
2. **The Validity of Threshold Tuning:** Lowering the threshold from 0.50 to 0.02–0.04 does not invent skill; it recalibrates the decision rule to match the model's posterior probability scale, converting weak synoptic moisture signals into actionable advisory alarms.
