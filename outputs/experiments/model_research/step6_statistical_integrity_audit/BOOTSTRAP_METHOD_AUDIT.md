# Step 6 Audit: Bootstrap Methodology and Statistical Inferential Validity

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India  
**Audit Target:** Step 5 Bootstrap Statistical Analysis (`BOOTSTRAP_STATISTICAL_ANALYSIS.csv`)  
**Audit Date:** October 30, 2026  
**Artifact Directory:** `outputs/experiments/model_research/step6_statistical_integrity_audit/`

---

## 1. Executive Summary & Audit Mandate

In Step 5, a stationary block bootstrap analysis was executed with $B = 1,000$ iterations to evaluate the paired differences between **Exp D (Combined Objective)** and **Exp A (Baseline Control)**. The resulting report claimed statistical significance at $p < 0.001$ across multiple horizons and reported exact values of `p = 0.000` in `BOOTSTRAP_STATISTICAL_ANALYSIS.csv`.

This audit evaluates:
1. Exactly what was resampled and what the confidence intervals and $p$-values mathematically represent.
2. Whether baseline and Exp D were evaluated on matched timestamps and identical valid observations.
3. The validity of the chosen block length ($L=6$ sequences = 3.0 hours) relative to forecast lead times up to 6.0 hours (+360m).
4. The statistical trap of reporting `p = 0.000` from empirical Monte Carlo resampling.
5. The critical distinction between **temporal sampling uncertainty** and **across-seed initialization variance**.
6. The small-sample limitations of bootstrapping on the July 5, 2025 validation episode ($N=33$).

A corrected, fully transparent recomputation has been implemented in [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv).

---

## 2. Forensic Analysis of the Step 5 Bootstrap Implementation

### 2.1 What Was Resampled?
Inspection of `scripts/run_statistical_bootstrap_analysis.py` (lines 75–86) confirms that the resampling unit was **blocks of contiguous temporal sequences**:
```python
def generate_block_bootstrap_indices(
    n_samples: int, block_size: int, n_boot: int, seed: int = 42
) -> np.ndarray:
  rng = np.random.RandomState(seed)
  n_blocks_needed = int(np.ceil(n_samples / block_size))
  valid_starts = np.arange(max(1, n_samples - block_size + 1))
  boot_indices = np.zeros((n_boot, n_samples), dtype=int)
  for b in range(n_boot):
    starts = rng.choice(valid_starts, size=n_blocks_needed, replace=True)
    blocks = [np.arange(st, min(st + block_size, n_samples)) for st in starts]
    idx = np.concatenate(blocks)[:n_samples]
    boot_indices[b] = idx
  return boot_indices
```

- **Resampled Unit:** Indices of contiguous half-hour forecast sequences.
- **Block Length:** $L = 6$ sequences ($6 \times 30\text{ min} = 3.0\text{ hours}$).
- **Pairing Mechanism:** In every bootstrap iteration $b$, the **exact same block index array** `boot_idx[b]` was applied simultaneously to Exp A and Exp D predictions. The differences $\Delta^{[b]} = M_{\text{ExpD}}^{[b]} - M_{\text{ExpA}}^{[b]}$ were calculated strictly within each resampled cohort. **Paired alignment across identical timestamps and observations is verified.**

---

### 2.2 Methodological Flaw 1: Reporting `p = 0.000` as an Exact Probability
In Step 5, the $p$-value was calculated as:
```python
p_val = float(np.mean(deltas <= 0.0))
```
When all $B = 1,000$ bootstrap replicates yielded $\Delta > 0$, the script output `p_value = 0.000`.

**Scientific Audit Finding:**
- An empirical Monte Carlo permutation or bootstrap test with $B$ iterations can **never** establish a $p$-value of zero.
- The minimum resolvable probability is bounded by the number of resamples: $\frac{1}{B} = 0.001$.
- Following standard statistical practice (Davison & Hinkley, 1997), the unbiased, continuity-corrected estimate is:
  $$p = \frac{k + 1}{B + 1}$$
  where $k$ is the number of bootstrap replicates contrary to the alternative hypothesis. When $k = 0$ out of $B = 1,000$, the correct empirical bound is:
  $$p = \frac{0 + 1}{1000 + 1} = \frac{1}{1001} \approx 0.000999 \implies p < 0.001$$
- **Correction Applied:** In [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv), all uncalibrated `0.000` values have been replaced with bounded notation `p < 0.001`, and the exact count of contrary replicates $k$ is explicitly reported.

---

### 2.3 Methodological Flaw 2: Conflation of Temporal Uncertainty and Seed Variance
In Step 5, the bootstrap loop iterated over `for seed in SEEDS:`.
This means the bootstrap was executed **for each individual checkpoint pair** ($s=42, 123, 999$).

**Critical Inferential Distinction:**
1. **What each row in `BOOTSTRAP_STATISTICAL_ANALYSIS.csv` actually tested:**
   *"Given Checkpoint Exp A trained with Seed $s$ and Checkpoint Exp D trained with Seed $s$, what is the probability that Exp D outperforms Exp A across random 3-hour weather blocks drawn from that specific evaluation cohort?"*
   This is a test of **temporal episode sampling uncertainty**.
2. **What it did NOT test:**
   It did **not** test whether Exp D outperforms Exp A across random initializations of the neural network weights!
3. **The Three-Seed Limitation:**
   With only $N_{\text{seeds}} = 3$ random initializations, an exact non-parametric sign test across seeds has a minimum one-sided $p$-value of:
   $$p = \left(\frac{1}{2}\right)^3 = 0.125$$
   **It is mathematically impossible to achieve a non-parametric $p < 0.05$ across 3 seeds alone.**
   To claim that Step 5 "proved statistical significance across seeds" is a serious methodological error. What Step 5 demonstrated was **consistent empirical replication across 3 seeds**, while the statistical significance applies to **temporal episode sampling**.
- **Correction Applied:** In the corrected audit artifacts, the interpretation is clearly partitioned into (1) temporal sampling confidence intervals per seed, and (2) sample standard deviation ($s_\Delta$) across the 3 seeds.

---

### 2.4 Methodological Flaw 3: Block Length vs. Forecast Horizon Decorrelation
In Step 5, a single block length of $L = 6$ sequences ($3.0\text{ hours}$) was used for all five forecast horizons (+30m, +60m, +120m, +180m, +360m).

**Meteorological Autocorrelation Analysis:**
- For **+30m** (1 lead step = 30 min): A 3.0-hour block covers 6 consecutive lead verification times. Residual autocorrelation decay is adequate.
- For **+180m** (6 lead steps = 3.0 hours): A 3.0-hour block matches the forecast lead time.
- For **+360m** (12 lead steps = 6.0 hours): Two nowcast sequences initialized 30 minutes apart have a **5.5-hour overlap** in their verification target trajectories!
  Because the forecast window ($6.0\text{ hours}$) exceeds the bootstrap block size ($3.0\text{ hours}$), adjacent blocks in the Step 5 bootstrap were **not meteorologically independent** at +360m!
- **Correction Applied:** In [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv), we re-evaluated the 129-sequence test cohort using both:
  1. $L = 6$ sequences ($3.0\text{ hours}$, 22 blocks).
  2. $L = 12$ sequences ($6.0\text{ hours}$, 11 non-overlapping blocks), providing complete temporal decorrelation across the entire +360-minute forecast horizon.

---

### 2.5 Methodological Flaw 4: Small-Sample Limitations on the Validation Set ($N=33$)
On July 5, 2025, there are only $N = 33$ sequence intervals (16.5 hours of active rainfall).
With a block length of $L = 6$ sequences, a sample of 33 sequences consists of only:
$$\frac{33}{6} = 5.5 \text{ blocks}$$
- Sampling 6 blocks with replacement from a pool of only 28 overlapping block candidates yields high discrete granularity in the bootstrap distribution.
- Therefore, while the point estimate gains on July 5, 2025 are large ($\Delta \text{CSI} = +0.0928, \Delta \text{FSS} = +0.3276$), **the validation cohort alone lacks the sample size for high-powered statistical inference**.
- The confirmatory test cohort ($N=129$ sequences across July 12–14, 2023, yielding $\sim 22$ 3-hour blocks and $\sim 11$ 6-hour blocks) provides a much more robust empirical basis.

---

## 3. Recomputed Statistical Results (Corrected Block Bootstrap)

The corrected analysis was executed using 1,000 resamples on matched temporal blocks with continuity-corrected $p$-values. Sourced directly from [`CORRECTED_BOOTSTRAP_RESULTS.csv`](file:///d:/Downloads/honours/outputs/experiments/model_research/step6_statistical_integrity_audit/CORRECTED_BOOTSTRAP_RESULTS.csv):

### 3.1 Primary Intermediate Horizons (+120m and +180m)

#### Confirmatory Test Set (July 12–14, 2023, $N=129$) with Complete 6-Hour Block Decorrelation ($L=12$):

| Metric | Horizon | Exp A Point | Exp D Point | Point $\Delta$ | 95% Bootstrap CI | Contrary Replicates ($k$) | Bounded $p$-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pointwise CSI** | **+120m** | $0.1057$ | $0.1829$ | $+0.0772$ | $[+0.0461, +0.1039]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| (Seed 42) | **+180m** | $0.0050$ | $0.1050$ | $+0.1000$ | $[+0.0385, +0.1557]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| **Pointwise CSI** | **+120m** | $0.0937$ | $0.2395$ | $+0.1458$ | $[+0.1057, +0.1772]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| (Seed 123) | **+180m** | $0.0038$ | $0.1506$ | $+0.1468$ | $[+0.0784, +0.2078]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| **Pointwise CSI** | **+120m** | $0.1704$ | $0.2176$ | $+0.0472$ | $[+0.0151, +0.0818]$ | $1 / 1000$ | $\mathbf{p = 0.0020}$ | **YES** |
| (Seed 999) | **+180m** | $0.0272$ | $0.1343$ | $+0.1071$ | $[+0.0526, +0.1528]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| **Circular FSS (30 km)** | **+120m** | $0.3797$ | $0.5625$ | $+0.1827$ | $[+0.1265, +0.2307]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| (Seed 42) | **+180m** | $0.0225$ | $0.3895$ | $+0.3670$ | $[+0.1843, +0.4789]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| **Circular FSS (30 km)** | **+120m** | $0.3301$ | $0.6298$ | $+0.2998$ | $[+0.2104, +0.4132]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| (Seed 123) | **+180m** | $0.0188$ | $0.4741$ | $+0.4553$ | $[+0.2642, +0.5735]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |
| **Circular FSS (30 km)** | **+120m** | $0.5242$ | $0.6079$ | $+0.0837$ | $[+0.0246, +0.1517]$ | $2 / 1000$ | $\mathbf{p = 0.0030}$ | **YES** |
| (Seed 999) | **+180m** | $0.1170$ | $0.4368$ | $+0.3198$ | $[+0.1654, +0.4279]$ | $0 / 1000$ | $\mathbf{p < 0.001}$ | **YES** |

#### Three-Seed Pooled Ensemble Mean Difference ($L=12$, 6.0-hour blocks):
- **+120m Pointwise CSI:** Mean $\bar{\Delta} = \mathbf{+0.0901}$, $95\%\text{ CI } [+0.0633, +0.1139]$, $p < 0.001$ ($0/1000$ contrary replicates).
- **+180m Pointwise CSI:** Mean $\bar{\Delta} = \mathbf{+0.1180}$, $95\%\text{ CI } [+0.0652, +0.1654]$, $p < 0.001$ ($0/1000$ contrary replicates).
- **+120m Circular FSS (30 km):** Mean $\bar{\Delta} = \mathbf{+0.1887}$, $95\%\text{ CI } [+0.1340, +0.2526]$, $p < 0.001$ ($0/1000$ contrary replicates).
- **+180m Circular FSS (30 km):** Mean $\bar{\Delta} = \mathbf{+0.3807}$, $95\%\text{ CI } [+0.2241, +0.4789]$, $p < 0.001$ ($0/1000$ contrary replicates).
- **+180m QPF RMSE:** Mean $\bar{\Delta} = \mathbf{-0.0008\text{ mm/hr}}$, $95\%\text{ CI } [-0.0617, +0.0545]$, $p = 0.449$ ($449/1000$ positive replicates). **RMSE is statistically indistinguishable between Exp A and Exp D.**

---

### 3.2 Long Horizon (+360m / 6 Hours Ahead)

With complete 6-hour block decorrelation ($L=12$):
- **Pointwise CSI (Seed 42):** $\Delta = +0.0104$, $95\%\text{ CI } [+0.0013, +0.0194]$, $p = 0.0040 \implies$ Statistically Significant ($p < 0.01$).
- **Pointwise CSI (Seed 123):** $\Delta = +0.0008$, $95\%\text{ CI } [0.0000, +0.0026]$, $p = 0.2830 \implies$ Not Statistically Significant.
- **Pointwise CSI (Seed 999):** $\Delta = +0.0264$, $95\%\text{ CI } [+0.0093, +0.0469]$, $p < 0.001 \implies$ Statistically Significant ($p < 0.001$).
- **Three-Seed Mean Difference:** $\bar{\Delta} = \mathbf{+0.0125}$, $95\%\text{ CI } [+0.0038, +0.0223]$, $p = 0.0020$.

**Scientific Interpretation:**
While Exp D demonstrates a statistically detectable nonzero difference over Exp A's absolute collapse ($0.0000$), the absolute magnitude of $\text{CSI} \approx 0.0125$ and $\text{FSS} \approx 0.0526$ remains **far below the threshold of operational utility**.

---

## 4. Summary of Audit Recommendations for Thesis Manuscript

1. **Replace `p = 0.000` with Bounded Notation:**
   All tables must report $p < 0.001$ or explicit empirical fractions $(k+1)/(B+1)$.
2. **Explicitly State the Inferential Sampling Unit:**
   The thesis must state:
   *"Statistical significance was evaluated using stationary temporal block bootstrapping ($B = 1,000$ iterations, block size = 6.0 hours) across the test sequences. This quantifies uncertainty arising from temporal weather episodes. Across random network initializations, consistency was verified across $N=3$ independent random seeds."*
3. **Do Not Claim Proof Across Random Seeds:**
   Do not state that $N=3$ seeds establish statistical significance across initializations. Instead, report the sample standard deviation ($s_\Delta$) across seeds to document reproducibility.
4. **Disclose Sample Limitations on Validation Split:**
   Acknowledge that the July 5, 2025 validation episode ($N=33$) is a single convective day; primary inferential power derives from the multi-day test cohort ($N=129$).
