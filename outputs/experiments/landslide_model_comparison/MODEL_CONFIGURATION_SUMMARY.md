# Landslide Model Architecture & Configuration Summary

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Component:** Landslide Susceptibility Machine Learning Suite  
**Location:** `outputs/experiments/landslide_model_comparison/MODEL_CONFIGURATION_SUMMARY.md`  
**Date:** October 30, 2026  

---

## 1. Candidate Model Assessment & Suitability Table

| Model Architecture | Purpose | Key Advantages | Inherent Limitations | Suitability for Meghalaya Dataset ($N=228$) |
| :--- | :--- | :--- | :--- | :---: |
| **XGBoost (Baseline)** | Baseline gradient boosting on decision trees | Fast, handles tabular collinearities, native L1/L2 regularization | Can overfit to regional terrain quirks if trees are too deep | **HIGH (Baseline reference)** |
| **XGBoost (Tuned & Balanced)** | Regularized shallow boosting with class weighting | `scale_pos_weight=5.0` boosts positive recall; shallow depth (2) limits overfitting | Lower specificity on broad lowland plains | **VERY HIGH (Best recall / F1 balance)** |
| **HistGradientBoosting** | Binning-based histogram gradient boosting (Scikit-Learn native LightGBM) | Discretizes continuous terrain features into integer bins; robust to extreme elevation outliers | Requires careful regularizer tuning to avoid leaf collapse | **EXCELLENT (Rank 1: Highest Spatial ROC-AUC = 0.6776)** |
| **LightGBM** | Leaf-wise gradient boosting with histogram binning | Fast, highly expressive split logic | Risk of leaf-wise overfitting on tiny sample sizes ($N < 500$) | **HIGH (Rank 5: Spatial ROC-AUC = 0.6227)** |
| **CatBoost** | Ordered boosting with symmetric oblivious trees | Built-in resistance to prediction shift and target leakage | Oblivious trees can struggle when one single feature dominates (e.g. slope) | **HIGH (Rank 8: Spatial ROC-AUC = 0.6127)** |
| **Random Forest (Baseline)** | Bagging ensemble of deep unconstrained trees | Reduces variance via bootstrap aggregation | Tends to memorize local training block terrain; huge train-test gap (AUC 1.0 vs 0.55) | **MODERATE (High overfitting risk)** |
| **Random Forest (Tuned & Balanced)** | Constrained shallow trees with class balancing | `class_weight='balanced'` and `max_depth=4` prevent branch over-specialization | Coarse decision boundaries across transition zones | **HIGH (Balanced Acc = 0.5959, F1 = 0.2878)** |
| **Extra Trees** | Extremely randomized decision trees | Random split thresholds maximize ensemble diversity | Excess randomness dilutes strong deterministic slope signals | **MODERATE (Lower discriminative power)** |
| **Gradient Boosting (GBDT)** | Classical Friedman greedy stage-wise boosting | Smooth gradient descent in function space | Slower, lacks native histogram binning; sensitive to outliers | **MODERATE (Outperformed by HistGBDT)** |
| **Logistic Regression (L2)** | Linear log-odds baseline with Ridge penalty | Convex optimization, highly interpretable coefficients | Cannot model nonlinear slope-curvature thresholds | **MODERATE (Underfits complex geomorphology)** |
| **Logistic Regression (Balanced)** | Linear model with inverse-frequency class weighting | Higher sensitivity for rare positive failure events | Increased false positive rate across moderate slopes | **MODERATE (Good baseline benchmark)** |
| **SVM (RBF Kernel)** | Non-linear maximum-margin hyperplane in Hilbert space | Effective in non-linear boundaries; Platt probability calibration | Sensitive to scaling; small $N$ can lead to support vector saturation | **MODERATE (Spatial ROC-AUC = 0.5989)** |
| **SVM (Linear Kernel)** | Linear maximum-margin separator | Simple structural risk minimization | Fails to separate non-linear valley/escarpment boundaries | **LOW (Spatial ROC-AUC < 0.50, worse than random)** |
| **Regularized MLP / ANN** | Small Multi-Layer Perceptron neural network | Captures complex multi-feature cross-interactions | Data hungry; severe risk of memorizing $N=228$ samples without massive regularization | **LOW TO MODERATE (Train AUC 1.0 vs Test 0.59)** |
| **Stacking Ensemble** | Meta-learner combining diverse top tree + linear models | Blends complementary error profiles | Adds complexity; requires strict nested cross-validation to prevent leakage | **HIGH (Solid spatial generalizability = 0.6151)** |

---

## 2. Terrain Feature Set Definitions

### Baseline Feature Set ($D = 8$)
Extracted directly from the 90m SRTM Digital Elevation Model NetCDF:
1. `elevation_mean`: Mean elevation above sea level (meters).
2. `slope_mean`: Mean slope angle across the 90m pixel (degrees).
3. `slope_max`: Maximum sub-pixel slope angle (degrees).
4. `relief`: Local elevation range ($\text{elev}_{\max} - \text{elev}_{\min}$).
5. `slope_std`: Standard deviation of slope, quantifying topographic roughness.
6. `aspect_northness`: $\cos(\text{aspect})$, ranging from $-1$ (due south) to $+1$ (due north).
7. `aspect_eastness`: $\sin(\text{aspect})$, ranging from $-1$ (due west) to $+1$ (due east).
8. `curvature_mean`: Surface profile/planform curvature (negative = hollow/gully, positive = nose/crest).

### Engineered Feature Set ($D = 12$)
Includes the 8 baseline features plus 4 scientifically derived geomorphometric indices:
9. `elevation_normalized`: Relative topographic position index:
   $$\text{TPI} = \frac{\text{elev} - \text{elev}_{\min}}{\text{elev}_{\max} - \text{elev}_{\min} + \epsilon} \in [0, 1]$$
10. `slope_roughness`: Relative slope coefficient of variation:
    $$\text{CV}_{\text{slope}} = \frac{\text{slope\_std}}{\text{slope\_mean} + \epsilon}$$
11. `relief_ratio`: Kinematic energy and relief interaction:
    $$\text{Relief\_Ratio} = \frac{\text{slope\_mean} \cdot \text{relief}}{1000.0}$$
12. `curvature_slope_interaction`: Convergent hollow shear stress proxy:
    $$\text{Interaction} = \text{curvature\_mean} \cdot \text{slope\_mean}$$

---

## 3. Exact Hyperparameter Specifications for All Models

### 1. HistGradientBoosting (Top Performing Model)
```python
HistGradientBoostingClassifier(
    max_iter=60,
    max_depth=3,
    learning_rate=0.04,
    l2_regularization=1.0,
    min_samples_leaf=10,
    random_state=42,
)
```
- **Why it won:** Binning terrain morphometry into 256 discrete integer intervals naturally buffers against extreme elevation and slope anomalies, producing the highest out-of-fold generalization across unseen geographic blocks (**ROC-AUC 0.6776** with only $\pm 0.0148$ standard deviation).

### 2. XGBoost (Tuned & Balanced)
```python
xgb.XGBClassifier(
    n_estimators=60,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=5.0,
    reg_alpha=0.5,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=42,
)
```
- **Characteristics:** Restricting tree depth to 2 prevents high-order spatial co-adaptation, while `scale_pos_weight=5.0` forces the model to balance false negatives against false positives, boosting recall to $43.2\%$ and PR-AUC to $0.2853$.

### 3. LightGBM Classifier
```python
lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=3,
    num_leaves=7,
    learning_rate=0.03,
    scale_pos_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
```

### 4. CatBoost Classifier
```python
cb.CatBoostClassifier(
    iterations=60,
    depth=3,
    learning_rate=0.04,
    auto_class_weights="Balanced",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
)
```

### 5. Random Forest (Tuned & Balanced)
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=4,
    min_samples_split=6,
    class_weight="balanced",
    random_state=42,
)
```

### 6. Stacking Ensemble
```python
StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)),
        ("lr", LogisticRegression(C=0.5, class_weight="balanced", random_state=42)),
        ("hgb", HistGradientBoostingClassifier(max_iter=40, max_depth=3, l2_regularization=1.0, random_state=42)),
    ],
    final_estimator=LogisticRegression(C=0.5, random_state=42),
    cv=3,
)
```

### 7. Regularized MLP / ANN
```python
MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    alpha=0.1,  # Strong L2 regularization penalty
    learning_rate_init=0.01,
    max_iter=800,
    early_stopping=False,
    random_state=42,
)
```

### 8. Support Vector Machine (RBF)
```python
SVC(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42,
)
```

---

## 4. Cross-Validation & Validation Safeguards

1. **Spatial Leakage Elimination:** Folds were strictly segregated by longitude into West Garo, Central Khasi, and East Jaintia. No training pixel was closer than several dozen kilometers to the evaluation block.
2. **Strict Training-Fold Preprocessing:** Standard scaling (`StandardScaler`) was fitted exclusively on the 2 training blocks in each iteration, preventing test fold feature leakage.
3. **Overfitting Gap Measurement:** For every model, we recorded the difference between training ROC-AUC and held-out spatial ROC-AUC. Deep models (Random Forest baseline, MLP) achieved training AUCs of $1.000$ but dropped to $0.55 - 0.59$ on unseen spatial blocks, exposing severe memorization that is masked when using standard random splits.
