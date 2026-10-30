# Spatial Metric Verification: Mathematical Formulation & Implementation Audit

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Module Audited:** `src/evaluation/spatial_metrics.py`  
**Test Suite:** `tests/test_spatial_metrics.py` (11/11 automated unit tests passing)

---

## 1. Mathematical Definition & Formula Verification

The canonical Fractions Skill Score (FSS) is defined by Roberts & Lean (2008) and Mittermaier (2014) as:

$$FSS = 1 - \frac{FBS}{FBS_{\text{worst}}}$$

where for a spatial domain of $N_V$ valid grid cells:

$$FBS = \frac{1}{N_V} \sum_{(x, y) \in \mathcal{D}_{\text{valid}}} \left[ M_k(x, y) - O_k(x, y) \right]^2$$

$$FBS_{\text{worst}} = \frac{1}{N_V} \sum_{(x, y) \in \mathcal{D}_{\text{valid}}} \left[ M_k(x, y)^2 + O_k(x, y)^2 \right]$$

and $M_k(x, y)$, $O_k(x, y)$ represent the spatial fraction of neighborhood cells exceeding the rainfall threshold $q$ in the forecast and observation fields, respectively.

### Source Code Audit (`src/evaluation/spatial_metrics.py`, lines 144–166):
```python
diff_sq = (p_fcst[val_cells] - p_obs[val_cells]) ** 2
sum_sq = p_fcst[val_cells] ** 2 + p_obs[val_cells] ** 2

fbs = float(np.sum(diff_sq) / n_valid)
fbs_worst = float(np.sum(sum_sq) / n_valid)

fss = 1.0 - (fbs / fbs_worst)
```

**Verification Verdict**:
- The normalization constant $\frac{1}{N_V}$ is applied identically to both numerator ($FBS$) and denominator ($FBS_{\text{worst}}$), canceling out cleanly:
  $$\frac{FBS}{FBS_{\text{worst}}} = \frac{\sum (M_k - O_k)^2}{\sum (M_k^2 + O_k^2)}$$
- This exactly matches the Roberts & Lean formulation.
- Hand-calculated analytic verification was implemented in `test_fss_denominator_and_hand_calculated_values` and verified to $< 10^{-6}$ precision.

---

## 2. Verification of Eight Methodological Pillars

### 2.1 Construction of Fractional Fields
- Fractions are computed using 2D spatial convolutions via `scipy.ndimage.convolve`.
- For each cell $(x, y)$, the fractional event frequency is normalized by the actual number of valid observation pixels present inside the kernel window:
  $$P(x, y) = \frac{\sum_{(u, v) \in \mathcal{N}_k} I(u, v) \cdot W(u, v)}{\sum_{(u, v) \in \mathcal{N}_k} W(u, v)}$$
- Verified: No artificial inflation occurs near boundaries or missing data areas.

### 2.2 Denominator & Upper/Lower Bounds
- By Cauchy-Schwarz inequality, $(M_k - O_k)^2 \le M_k^2 + O_k^2$ for any non-negative fractions $M_k, O_k \ge 0$.
- Consequently, $0 \le \frac{FBS}{FBS_{\text{worst}}} \le 1.0$ (when events exist).
- $FSS$ is strictly bounded in $[0.0, 1.0]$:
  - $FSS = 1.0$: Perfect fractional match everywhere.
  - $FSS = 0.0$: Total spatial displacement with zero overlap.

### 2.3 Domain Boundary & Edge Handling
- In standard image convolutions, "zero-padding" edges artificially lowers fractions for border pixels.
- In `SpatialVerificationEngine.compute_fraction_field()`, boundary normalization is explicitly applied:
  `mask_sum = ndi.convolve(m, kernel, mode="constant", cval=0.0)`
  `fractions[valid_denom] = event_sum[valid_denom] / mask_sum[valid_denom]`
- Verified in `test_boundary_normalization_correctness`: A single event placed at the top-left corner $(0, 0)$ with a $3 \times 3$ kernel yields fraction **$1/4 = 0.25$** (since only 4 cells exist in domain at that corner), NOT $1/9 = 0.111$.

### 2.4 Missing-Data Mask Handling
- The satellite observation validity mask ($W \in \{0, 1\}$) excludes masked/corrupted pixels from both numerator and denominator.
- Verified in `test_fss_mask_handling`: Synthetic rainfall introduced exclusively in masked cells does not alter the FSS of valid cells.

### 2.5 Explicit Handling of Zero-Event Frames
When neither the forecast nor the observation contains events exceeding threshold $q$:
- $FBS = 0.0$ and $FBS_{\text{worst}} = 0.0$ (division by zero).
- In `compute_single_fss()`, this condition is detected (`fbs_worst < 1e-12`):
  ```python
  if fbs_worst < 1e-12:
      return {"fss": 1.0, "fbs": 0.0, "fbs_worst": 0.0, "is_zero_event": True}
  ```
- In sequence evaluation (`evaluate_sequence_dataset`), the accumulated FSS accumulates total numerators and denominators across active frames, preventing zero-event frames from distorting accumulated skill.

### 2.6 Rainfall Units and Event Thresholds
- IMERG precipitation data is calibrated in $\text{mm/hr}$.
- Evaluated thresholds:
  - $0.1\text{ mm/hr}$: Measurable precipitation occurrence.
  - $1.0\text{ mm/hr}$: Moderate stratiform precipitation.
  - $5.0\text{ mm/hr}$: Heavy convective rainfall hazard threshold.
- The units in predictions and observations are identical ($\text{mm/hr}$).

### 2.7 Spatial Alignment of Forecasts and Observations
- Both predictions and observations are aligned on the identical $25 \times 40$ WGS-84 grid spanning $24.55^\circ\text{N}-26.95^\circ\text{N}$ and $89.55^\circ\text{E}-93.45^\circ\text{E}$.
- Array indices map to the exact same geographical coordinate coordinates.

### 2.8 Absence of Test-Set Information Leakage
- All decision thresholds in earlier classification steps were tuned on the validation day (July 5, 2025).
- In spatial FSS evaluation, thresholds ($0.1, 1.0, 5.0\text{ mm/hr}$) are **fixed physical meteorological standards**, not tuned on the test set.
- No model parameters were optimized or adjusted during evaluation.

---

## 3. Automated Test Suite Verification

The expanded unit test suite `tests/test_spatial_metrics.py` was executed:

```
tests/test_spatial_metrics.py::test_km_to_grid_radius_mapping PASSED     [  9%]
tests/test_spatial_metrics.py::test_fss_perfect_forecast PASSED          [ 18%]
tests/test_spatial_metrics.py::test_fss_zero_forecast_with_events PASSED [ 27%]
tests/test_spatial_metrics.py::test_fss_pure_false_alarm PASSED          [ 36%]
tests/test_spatial_metrics.py::test_fss_both_empty_zero_case PASSED      [ 45%]
tests/test_spatial_metrics.py::test_fss_displacement_properties PASSED   [ 54%]
tests/test_spatial_metrics.py::test_fss_mask_handling PASSED             [ 63%]
tests/test_spatial_metrics.py::test_neighborhood_contingency PASSED      [ 72%]
tests/test_spatial_metrics.py::test_circular_kernel_geometry PASSED      [ 81%]
tests/test_spatial_metrics.py::test_fss_denominator_and_hand_calculated_values PASSED [ 90%]
tests/test_spatial_metrics.py::test_boundary_normalization_correctness PASSED [100%]
============================= 11 passed in 1.24s ==============================
```

**Final Verification Verdict**:  
The spatial verification implementation in `src/evaluation/spatial_metrics.py` is **mathematically sound, fully compliant with international meteorological standards, and verified by 11 unit tests**.
