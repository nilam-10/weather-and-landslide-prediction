"""Landslide Susceptibility Comprehensive Model Comparison Suite.

Rigorous evaluation across 15 candidate machine-learning models:
- XGBoost (Baseline & Tuned)
- Random Forest (Baseline & Tuned)
- Extra Trees
- LightGBM
- CatBoost
- HistGradientBoosting
- Gradient Boosting
- Logistic Regression (Standard & Balanced)
- Support Vector Machine (RBF & Linear)
- Regularized Artificial Neural Network (MLP)
- Stacking Ensemble

Evaluation Protocol:
- Strictly symmetric unbuffered feature extraction (no positive scarp-max bias)
- 3-Fold Spatial Block Cross-Validation (West Garo, Central Khasi, East Jaintia)
- Preprocessing, scaling, and class balancing strictly within training folds
- Comprehensive metrics: ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, Precision,
  Recall/Sensitivity, Specificity, F1, Brier Score, and Confusion Matrix.
- Also evaluates Random 5-Fold CV to demonstrate spatial autocorrelation inflation.

Outputs:
- outputs/experiments/landslide_model_comparison/MODEL_METRICS.csv
- outputs/experiments/landslide_model_comparison/SPATIAL_CROSS_VALIDATION_RESULTS.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import matplotlib.path as mpath
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERRAIN_NC = PROJECT_ROOT / "data" / "processed" / "terrain" / "meghalaya_terrain_features_90m.nc4"
BOUNDARY_GEOJSON = PROJECT_ROOT / "data" / "raw" / "boundary" / "meghalaya_boundary.geojson"
EVENTS_CSV = PROJECT_ROOT / "outputs" / "landslide" / "meghalaya_landslide_events.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "landslide_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_terrain() -> Dict[str, np.ndarray]:
    with h5py.File(TERRAIN_NC, "r") as f:
        return {
            "lat": f["lat"][:],
            "lon": f["lon"][:],
            "elevation_mean": f["elevation_mean"][:],
            "elevation_min": f["elevation_min"][:],
            "elevation_max": f["elevation_max"][:],
            "slope_mean": f["slope_mean"][:],
            "slope_max": f["slope_max"][:],
            "relief": f["relief"][:],
            "slope_std": f["slope_std"][:],
            "aspect_northness": f["aspect_northness"][:],
            "aspect_eastness": f["aspect_eastness"][:],
            "curvature_mean": f["curvature_mean"][:],
            "meghalaya_mask": f["meghalaya_mask"][:],
            "valid_mask": f["valid_mask"][:],
        }


def load_verified_events() -> List[Dict[str, Any]]:
    with open(BOUNDARY_GEOJSON, "r", encoding="utf-8") as f:
        bdata = json.load(f)
    poly = mpath.Path(bdata["features"][0]["geometry"]["coordinates"][0])

    events = []
    with open(EVENTS_CSV, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            if poly.contains_point([lon, lat]):
                trig = (r.get("landslide_trigger") or "").strip().lower()
                if trig in ["continuous_rain", "downpour", "rain"]:
                    events.append(r)
    return events


def sample_pseudo_absences(
    terrain: Dict[str, np.ndarray],
    events: List[Dict[str, Any]],
    n_samples: int = 190,
    seed: int = 42,
) -> List[Tuple[float, float, str]]:
    np.random.seed(seed)
    lats = terrain["lat"]
    lons = terrain["lon"]
    m_mask = terrain["meghalaya_mask"]
    v_mask = terrain["valid_mask"]

    event_grid = np.zeros_like(m_mask, dtype=bool)
    for ev in events:
        lat = float(ev["latitude"])
        lon = float(ev["longitude"])
        r_idx = int(np.argmin(np.abs(lats - lat)))
        c_idx = int(np.argmin(np.abs(lons - lon)))
        if 0 <= r_idx < len(lats) and 0 <= c_idx < len(lons):
            event_grid[r_idx, c_idx] = True

    from scipy.ndimage import binary_dilation
    y, x = np.ogrid[-33:34, -33:34]
    selem = (x**2 + y**2) <= 33**2
    exclusion_mask = binary_dilation(event_grid, structure=selem)

    allowed = (m_mask == 1) & (v_mask == 1) & (~exclusion_mask)
    allowed_idx = np.argwhere(allowed)

    chosen = np.random.choice(len(allowed_idx), size=n_samples, replace=False)
    bg = []
    for idx in chosen:
        r, c = allowed_idx[idx]
        plat = float(lats[r])
        plon = float(lons[c])
        block = "West" if plon < 91.0 else ("Central" if plon < 92.0 else "East")
        bg.append((plat, plon, block))
    return bg


def extract_features(
    points: List[Tuple[float, float, int, str]],
    terrain: Dict[str, np.ndarray],
    include_engineered: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    lats = terrain["lat"]
    lons = terrain["lon"]

    base_feature_names = [
        "elevation_mean",
        "slope_mean",
        "slope_max",
        "relief",
        "slope_std",
        "aspect_northness",
        "aspect_eastness",
        "curvature_mean",
    ]
    if include_engineered:
        feature_names = base_feature_names + [
            "elevation_normalized",
            "slope_roughness",
            "relief_ratio",
            "curvature_slope_interaction",
        ]
    else:
        feature_names = base_feature_names

    X_list = []
    y_list = []
    block_list = []

    for lat, lon, label, block in points:
        r_idx = int(np.argmin(np.abs(lats - lat)))
        c_idx = int(np.argmin(np.abs(lons - lon)))

        # Strictly symmetric point extraction
        elev = float(terrain["elevation_mean"][r_idx, c_idx])
        elev_min = float(terrain["elevation_min"][r_idx, c_idx])
        elev_max = float(terrain["elevation_max"][r_idx, c_idx])
        slope_m = float(terrain["slope_mean"][r_idx, c_idx])
        slope_mx = float(terrain["slope_max"][r_idx, c_idx])
        rel = float(terrain["relief"][r_idx, c_idx])
        s_std = float(terrain["slope_std"][r_idx, c_idx])
        north = float(terrain["aspect_northness"][r_idx, c_idx])
        east = float(terrain["aspect_eastness"][r_idx, c_idx])
        curv = float(terrain["curvature_mean"][r_idx, c_idx])

        row = [elev, slope_m, slope_mx, rel, s_std, north, east, curv]

        if include_engineered:
            elev_norm = (elev - elev_min) / (elev_max - elev_min + 1e-4)
            slope_rough = s_std / (slope_m + 1e-4)
            relief_ratio = (slope_m * rel) / 1000.0
            curv_slope = curv * slope_m
            row.extend([elev_norm, slope_rough, relief_ratio, curv_slope])

        X_list.append(row)
        y_list.append(label)
        block_list.append(block)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), block_list, feature_names


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": 0.5, "pr_auc": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0,
            "precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0, "brier": 0.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

    roc_auc = float(roc_auc_score(y_true, y_prob))
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))
    brier = float(brier_score_loss(y_true, y_prob))

    # Binary classification at threshold 0.5
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / max(tn + fp, 1))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
        "n_pos": int(sum(y_true)),
    }


def get_model_zoo(seed: int = 42) -> Dict[str, Any]:
    """Instantiate diverse, scientifically regularized candidate models."""
    return {
        # 1. Existing baseline
        "XGBoost (Baseline)": xgb.XGBClassifier(
            n_estimators=40, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed,
        ),
        # 2. Tuned XGBoost with shallow depth & class weighting
        "XGBoost (Tuned & Balanced)": xgb.XGBClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.03, subsample=0.7,
            colsample_bytree=0.7, scale_pos_weight=5.0, reg_alpha=0.5, reg_lambda=2.0,
            eval_metric="logloss", random_state=seed,
        ),
        # 3. Baseline Random Forest
        "Random Forest (Baseline)": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=4, random_state=seed,
        ),
        # 4. Tuned Random Forest with balanced weights
        "Random Forest (Tuned & Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_split=6, class_weight="balanced", random_state=seed,
        ),
        # 5. Extra Trees Classifier (reduces variance via extreme random splits)
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=100, max_depth=4, min_samples_split=6, class_weight="balanced", random_state=seed,
        ),
        # 6. LightGBM Classifier
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=50, max_depth=3, num_leaves=7, learning_rate=0.03,
            scale_pos_weight=5.0, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        ),
        # 7. CatBoost Classifier
        "CatBoost Classifier": cb.CatBoostClassifier(
            iterations=60, depth=3, learning_rate=0.04, auto_class_weights="Balanced",
            l2_leaf_reg=3.0, random_seed=seed, verbose=False,
        ),
        # 8. HistGradientBoosting (LightGBM-style in Scikit-Learn)
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=50, max_depth=3, learning_rate=0.04, l2_regularization=1.0, random_state=seed,
        ),
        # 9. Conventional Gradient Boosting
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=40, max_depth=2, learning_rate=0.04, subsample=0.8, random_state=seed,
        ),
        # 10. Logistic Regression (L2 Baseline)
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed,
        ),
        # 11. Logistic Regression (Balanced)
        "Logistic Regression (Balanced)": LogisticRegression(
            C=0.5, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=seed,
        ),
        # 12. Support Vector Machine (RBF Kernel)
        "SVM (RBF Kernel)": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed,
        ),
        # 13. Support Vector Machine (Linear Kernel)
        "SVM (Linear Kernel)": SVC(
            C=0.5, kernel="linear", probability=True, class_weight="balanced", random_state=seed,
        ),
        # 14. Regularized Multi-Layer Perceptron (ANN)
        "Regularized MLP / ANN": MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu", alpha=0.1,
            learning_rate_init=0.01, max_iter=800, early_stopping=False, random_state=seed,
        ),
        # 15. Stacking Ensemble
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=seed)),
                ("lr", LogisticRegression(C=0.5, class_weight="balanced", random_state=seed)),
                ("hgb", HistGradientBoostingClassifier(max_iter=40, max_depth=3, l2_regularization=1.0, random_state=seed)),
            ],
            final_estimator=LogisticRegression(C=0.5, random_state=seed),
            cv=3,
        ),
    }


def evaluate_models_spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    blocks: List[str],
    models: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute 3-Fold Spatial Block Cross-Validation across all models."""
    unique_blocks = ["West", "Central", "East"]

    fold_records = []
    summary_records = []

    for name, model in models.items():
        block_metrics = []
        train_roc_aucs = []

        for test_block in unique_blocks:
            train_idx = [i for i, b in enumerate(blocks) if b != test_block]
            test_idx = [i for i, b in enumerate(blocks) if b == test_block]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale within training fold
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Fit model
            # For tree models that don't strictly require scaling, scaled features are still well-behaved
            import copy
            m = copy.deepcopy(model)

            # Some models like XGBoost, LightGBM, CatBoost can take unscaled or scaled
            # We pass scaled features consistently to ensure fair comparison
            m.fit(X_train_s, y_train)

            # Test predictions
            p_test = m.predict_proba(X_test_s)[:, 1]
            m_test = calculate_metrics(y_test, p_test)
            m_test["model_name"] = name
            m_test["test_block"] = test_block
            m_test["validation_type"] = "Spatial Block CV (Leave-One-Block-Out)"

            # Train predictions to measure overfitting gap
            p_train = m.predict_proba(X_train_s)[:, 1]
            train_auc = float(roc_auc_score(y_train, p_train))
            m_test["train_roc_auc"] = round(train_auc, 4)
            m_test["overfitting_gap"] = round(train_auc - m_test["roc_auc"], 4)

            fold_records.append(m_test)
            block_metrics.append(m_test)
            train_roc_aucs.append(train_auc)

        # Average across the 3 spatial blocks
        mean_roc = float(np.mean([m["roc_auc"] for m in block_metrics]))
        std_roc = float(np.std([m["roc_auc"] for m in block_metrics]))
        mean_pr = float(np.mean([m["pr_auc"] for m in block_metrics]))
        std_pr = float(np.std([m["pr_auc"] for m in block_metrics]))
        mean_acc = float(np.mean([m["accuracy"] for m in block_metrics]))
        mean_bal_acc = float(np.mean([m["balanced_accuracy"] for m in block_metrics]))
        mean_prec = float(np.mean([m["precision"] for m in block_metrics]))
        mean_rec = float(np.mean([m["recall"] for m in block_metrics]))
        mean_spec = float(np.mean([m["specificity"] for m in block_metrics]))
        mean_f1 = float(np.mean([m["f1"] for m in block_metrics]))
        mean_brier = float(np.mean([m["brier"] for m in block_metrics]))
        mean_train_auc = float(np.mean(train_roc_aucs))
        overfitting_gap = mean_train_auc - mean_roc

        summary_records.append({
            "model_name": name,
            "validation_scheme": "3-Fold Spatial Block CV (Leave-One-Block-Out)",
            "mean_roc_auc": round(mean_roc, 4),
            "std_roc_auc": round(std_roc, 4),
            "mean_pr_auc": round(mean_pr, 4),
            "std_pr_auc": round(std_pr, 4),
            "mean_accuracy": round(mean_acc, 4),
            "mean_balanced_accuracy": round(mean_bal_acc, 4),
            "mean_precision": round(mean_prec, 4),
            "mean_recall_sensitivity": round(mean_rec, 4),
            "mean_specificity": round(mean_spec, 4),
            "mean_f1_score": round(mean_f1, 4),
            "mean_brier_score": round(mean_brier, 4),
            "mean_train_roc_auc": round(mean_train_auc, 4),
            "overfitting_gap_auc": round(overfitting_gap, 4),
            "target_0.80_achieved": "YES" if mean_roc >= 0.80 else "NO",
        })

    return fold_records, summary_records


def evaluate_models_random_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Any],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate standard Random 5-Fold CV to quantify the spatial autocorrelation inflation artifact."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    random_summary = []

    for name, model in models.items():
        fold_aucs = []
        fold_prs = []
        fold_f1s = []

        for train_idx, test_idx in kf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            import copy
            m = copy.deepcopy(model)
            m.fit(X_train_s, y_train)
            p_test = m.predict_proba(X_test_s)[:, 1]

            fold_aucs.append(float(roc_auc_score(y_test, p_test)))
            prec, rec, _ = precision_recall_curve(y_test, p_test)
            fold_prs.append(float(auc(rec, prec)))
            y_pred = (p_test >= 0.5).astype(int)
            fold_f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))

        random_summary.append({
            "model_name": name,
            "validation_scheme": "Random 5-Fold CV (Spatial Autocorrelation Inflated)",
            "random_cv_mean_roc_auc": round(float(np.mean(fold_aucs)), 4),
            "random_cv_std_roc_auc": round(float(np.std(fold_aucs)), 4),
            "random_cv_mean_pr_auc": round(float(np.mean(fold_prs)), 4),
            "random_cv_mean_f1": round(float(np.mean(fold_f1s)), 4),
        })

    return random_summary


def run_experiment() -> None:
    print("=" * 75)
    print("STARTING RIGOROUS LANDSLIDE MODEL COMPARISON SUITE (15 MODELS)")
    print("=" * 75)

    # 1. Load data
    terrain = load_terrain()
    events = load_verified_events()
    print(f"Verified rainfall-triggered landslide events inside boundary: {len(events)}")
    bg_samples = sample_pseudo_absences(terrain, events, n_samples=190, seed=42)
    print(f"Sampled pseudo-absence background controls (>3km buffer): {len(bg_samples)}")

    # Assemble dataset
    points = []
    for ev in events:
        lat = float(ev["latitude"])
        lon = float(ev["longitude"])
        block = "West" if lon < 91.0 else ("Central" if lon < 92.0 else "East")
        points.append((lat, lon, 1, block))
    for lat, lon, block in bg_samples:
        points.append((lat, lon, 0, block))

    # Feature extraction: Baseline 8 features vs Engineered 12 features
    print("\nExtracting terrain features...")
    X_base, y_base, blocks_base, feat_base = extract_features(points, terrain, include_engineered=False)
    X_eng, y_eng, blocks_eng, feat_eng = extract_features(points, terrain, include_engineered=True)
    print(f"Dataset shape (Baseline 8 features): {X_base.shape}")
    print(f"Dataset shape (Engineered 12 features): {X_eng.shape}")

    # Model zoo
    models = get_model_zoo(seed=42)
    print(f"\nEvaluating {len(models)} candidate models under 3-Fold Spatial Block Cross-Validation...")

    # Spatial CV on Baseline Features
    fold_records_base, summary_base = evaluate_models_spatial_cv(X_base, y_base, blocks_base, models)

    # Spatial CV on Engineered Features
    models_eng = {f"{k} (+Engineered Feats)": v for k, v in models.items()}
    fold_records_eng, summary_eng = evaluate_models_spatial_cv(X_eng, y_eng, blocks_eng, models_eng)

    # Random 5-Fold CV on Baseline (to demonstrate the spatial autocorrelation inflation)
    print("Evaluating Random 5-Fold CV to demonstrate spatial autocorrelation bias...")
    random_cv_base = evaluate_models_random_cv(X_base, y_base, models, seed=42)
    random_map = {r["model_name"]: r for r in random_cv_base}

    # Combine summary results
    all_summary = []
    for r in summary_base:
        r_copy = dict(r)
        m_name = r["model_name"]
        if m_name in random_map:
            r_copy["random_cv_roc_auc"] = random_map[m_name]["random_cv_mean_roc_auc"]
            r_copy["spatial_penalty_gap"] = round(r_copy["random_cv_roc_auc"] - r_copy["mean_roc_auc"], 4)
        all_summary.append(r_copy)

    for r in summary_eng:
        all_summary.append(r)

    # Sort by mean spatial ROC-AUC descending
    all_summary.sort(key=lambda x: x["mean_roc_auc"], reverse=True)

    # Ensure all fieldnames are present across all rows
    all_keys = []
    for r in all_summary:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    metrics_csv = OUTPUT_DIR / "MODEL_METRICS.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_summary)

    folds_csv = OUTPUT_DIR / "SPATIAL_CROSS_VALIDATION_RESULTS.csv"
    all_folds = fold_records_base + fold_records_eng
    with open(folds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_folds[0].keys()))
        writer.writeheader()
        writer.writerows(all_folds)

    print("\n" + "=" * 75)
    print("TOP 5 PERFORMING MODELS UNDER HONEST SPATIAL BLOCK CROSS-VALIDATION:")
    print("=" * 75)
    for idx, r in enumerate(all_summary[:5], start=1):
        print(f"{idx}. {r['model_name']}: Spatial ROC-AUC = {r['mean_roc_auc']:.4f} ± {r['std_roc_auc']:.4f} | PR-AUC = {r['mean_pr_auc']:.4f} | F1 = {r['mean_f1_score']:.4f} | BalAcc = {r['mean_balanced_accuracy']:.4f}")

    print("\nBASELINE REFERENCE:")
    xgb_base = [r for r in all_summary if r["model_name"] == "XGBoost (Baseline)"][0]
    print(f"XGBoost (Baseline): Spatial ROC-AUC = {xgb_base['mean_roc_auc']:.4f} ± {xgb_base['std_roc_auc']:.4f} | PR-AUC = {xgb_base['mean_pr_auc']:.4f}")

    print(f"\nSaved master model metrics to: {metrics_csv}")
    print(f"Saved fold-level spatial breakdown to: {folds_csv}")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
