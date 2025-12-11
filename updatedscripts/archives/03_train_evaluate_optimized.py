#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_evaluate_optimized.py (mode-aware)
Faster, mode-controlled training + evaluation script.

Modes:
 - dev      : fastest (use while iterating; small search, fewer CV folds, optional sampling)
 - balanced : reasonable tradeoff (moderate search, cv=5)
 - final    : full run (larger search, cv=5, more repeats) — use for final reporting only

This script reads artifacts from exports/IO2/ and writes outputs to exports/IO3/.
"""
import os
import time
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# -------------------------
# CLI and mode settings
# -------------------------
parser = argparse.ArgumentParser(description="Train/evaluate pipeline (mode: dev|balanced|final)")
parser.add_argument("--mode", choices=["dev", "balanced", "final"], default="dev", help="Run mode to control runtime vs thoroughness")
parser.add_argument("--sample_frac", type=float, default=None, help="Optional: sample fraction (0-1) to run heavy steps on subset")
args = parser.parse_args()
MODE = args.mode
SAMPLE_FRAC = args.sample_frac

print(f"Running 03_train_evaluate_optimized.py in '{MODE}' mode. sample_frac={SAMPLE_FRAC}")

# Mode-specific knobs
if MODE == "dev":
    RF_N_ITER = 8
    RF_CV = 3
    LR_CV = 3
    DT_CV = 3
    PERM_REPEATS = 8
    CV_FOLDS_FINAL = 3
    SAMPLE_FOR_HEAVY = True
    SAMPLE_FRAC_DEFAULT = 0.25
elif MODE == "balanced":
    RF_N_ITER = 16
    RF_CV = 5
    LR_CV = 5
    DT_CV = 5
    PERM_REPEATS = 12
    CV_FOLDS_FINAL = 5
    SAMPLE_FOR_HEAVY = False
    SAMPLE_FRAC_DEFAULT = None
else:  # final
    RF_N_ITER = 30
    RF_CV = 5
    LR_CV = 5
    DT_CV = 5
    PERM_REPEATS = 30
    CV_FOLDS_FINAL = 5
    SAMPLE_FOR_HEAVY = False
    SAMPLE_FRAC_DEFAULT = None

if SAMPLE_FRAC is None and SAMPLE_FOR_HEAVY:
    SAMPLE_FRAC = SAMPLE_FRAC_DEFAULT

# -------------------------
# IO setup
# -------------------------
os.makedirs("exports", exist_ok=True)
os.makedirs(os.path.join("exports", "IO2"), exist_ok=True)
os.makedirs(os.path.join("exports", "IO3"), exist_ok=True)
os.makedirs("sid", exist_ok=True)

IN = os.path.join("exports", "IO2")
OUT = os.path.join("exports", "IO3")

required = [
    os.path.join(IN, "X_train.csv"),
    os.path.join(IN, "X_test.csv"),
    os.path.join(IN, "y_train.csv"),
    os.path.join(IN, "y_test.csv"),
    os.path.join(IN, "scaler.pkl"),
    os.path.join(IN, "final_feature_list.pkl")
]
for p in required:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

# -------------------------
# Load data and artifacts
# -------------------------
t0 = time.time()
X_train = pd.read_csv(os.path.join(IN, "X_train.csv"))
X_test = pd.read_csv(os.path.join(IN, "X_test.csv"))
y_train = pd.read_csv(os.path.join(IN, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(IN, "y_test.csv")).values.ravel()
scaler = joblib.load(os.path.join(IN, "scaler.pkl"))
final_features = joblib.load(os.path.join(IN, "final_feature_list.pkl"))

# Align columns defensively
X_train = X_train[final_features]
X_test = X_test[final_features]

# Optional sampling for heavy steps (dev mode)
if SAMPLE_FRAC is not None and 0 < SAMPLE_FRAC < 1.0:
    print(f"Sampling {SAMPLE_FRAC*100:.1f}% of training data for heavy steps (dev mode).")
    sample_idx = X_train.sample(frac=SAMPLE_FRAC, random_state=42).index
    X_train_sample = X_train.loc[sample_idx].reset_index(drop=True)
    y_train_sample = pd.Series(y_train).loc[sample_idx].reset_index(drop=True).values
else:
    X_train_sample = X_train
    y_train_sample = y_train

# Scale for linear models
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_sample_scaled = scaler.transform(X_train_sample) if SAMPLE_FRAC else X_train_scaled

print("Data loaded and preprocessed in {:.1f}s".format(time.time() - t0))

# -------------------------
# CV objects per mode
# -------------------------
cv_lr = StratifiedKFold(n_splits=LR_CV, shuffle=True, random_state=42)
cv_rf = StratifiedKFold(n_splits=RF_CV, shuffle=True, random_state=42)
cv_dt = StratifiedKFold(n_splits=DT_CV, shuffle=True, random_state=42)

# -------------------------
# Models and searches
# -------------------------
print("Starting model searches...")

# Logistic Regression (GridSearch on scaled data)
t1 = time.time()
lr = LogisticRegression(multi_class="ovr", class_weight="balanced", max_iter=2000)
lr_grid = {"C": [0.01, 0.1, 1.0], "penalty": ["l2"], "solver": ["liblinear"]}
gs_lr = GridSearchCV(lr, lr_grid, scoring="roc_auc_ovr", cv=cv_lr, n_jobs=-1, verbose=1)
gs_lr.fit(X_train_sample_scaled if SAMPLE_FRAC else X_train_scaled, y_train_sample if SAMPLE_FRAC else y_train)
joblib.dump(gs_lr, os.path.join(OUT, "gs_logistic.pkl"))
print(f"LR search done in {time.time()-t1:.1f}s; best={gs_lr.best_score_}")

# RandomForest (RandomizedSearch on unscaled data)
t2 = time.time()
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
rf_param_dist = {
    "n_estimators": [200, 500, 800],
    "max_depth": [None, 15, 30],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=RF_N_ITER, scoring="roc_auc_ovr",
                           cv=cv_rf, n_jobs=-1, random_state=42, verbose=1)
rs_rf.fit(X_train_sample if SAMPLE_FRAC else X_train, y_train_sample if SAMPLE_FRAC else y_train)
joblib.dump(rs_rf, os.path.join(OUT, "rs_randomforest.pkl"))
print(f"RF search done in {time.time()-t2:.1f}s; best={rs_rf.best_score_}")

# DecisionTree baseline (GridSearch)
t3 = time.time()
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_grid = {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]}
gs_dt = GridSearchCV(dt, dt_grid, scoring="roc_auc_ovr", cv=cv_dt, n_jobs=-1, verbose=1)
gs_dt.fit(X_train_sample if SAMPLE_FRAC else X_train, y_train_sample if SAMPLE_FRAC else y_train)
joblib.dump(gs_dt, os.path.join(OUT, "gs_decisiontree.pkl"))
print(f"DT search done in {time.time()-t3:.1f}s; best={gs_dt.best_score_}")

# -------------------------
# Evaluate on holdout
# -------------------------
print("Evaluating best estimators on test set...")
models = {
    "LogisticRegression": gs_lr.best_estimator_,
    "RandomForest": rs_rf.best_estimator_,
    "DecisionTree": gs_dt.best_estimator_
}

results = {}
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    start = time.time()
    X_eval = X_test_scaled if name == "LogisticRegression" else X_test
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)
        try:
            auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
        except Exception:
            auc = None
    else:
        auc = None
    results[name] = {"accuracy": acc, "roc_auc_ovr": auc, "confusion_matrix": cm.tolist(), "classification_report": cr}
    print(f"{name} eval done in {time.time()-start:.1f}s; acc={acc:.4f}, auc={auc}")

# -------------------------
# Select best model and calibrate if RF chosen
# -------------------------
best_name = None
best_score = -np.inf
for name, res in results.items():
    score = res["roc_auc_ovr"] if res["roc_auc_ovr"] is not None else res["accuracy"]
    if score is not None and score > best_score:
        best_score = score
        best_name = name
print("Best model selected:", best_name, "score:", best_score)
best_model = models[best_name]

if best_name == "RandomForest":
    print("Calibrating RandomForest probabilities (isotonic)...")
    tcal = time.time()
    calibrated = CalibratedClassifierCV(best_model, cv=min(5, CV_FOLDS_FINAL), method="isotonic")
    # calibrate on full training set (not sampled) for better probabilities
    calibrated.fit(X_train, y_train)
    best_model = calibrated
    print("Calibration done in {:.1f}s".format(time.time() - tcal))

# -------------------------
# Final metrics and save
# -------------------------
X_eval = X_test_scaled if best_name == "LogisticRegression" else X_test
y_pred = best_model.predict(X_eval)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, output_dict=True)
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_eval)
    try:
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
    except Exception:
        auc = None
else:
    auc = None

final_results = {"selected_model": best_name, "accuracy": acc, "roc_auc_ovr": auc, "confusion_matrix": cm.tolist(), "classification_report": cr}
joblib.dump(best_model, os.path.join(OUT, f"best_model_{best_name}.pkl"))

with open(os.path.join(OUT, "model_evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"final_results": final_results,
               "cv_best_scores": {"lr": gs_lr.best_score_, "rf": rs_rf.best_score_, "dt": gs_dt.best_score_},
               "mode": MODE}, f, indent=2)

print("Saved best model and summary to", OUT)

# -------------------------
# Permutation importance (fewer repeats in dev)
# -------------------------
try:
    print(f"Computing permutation importance (n_repeats={PERM_REPEATS})...")
    tperm = time.time()
    estimator_for_perm = best_model
    if hasattr(best_model, "base_estimator_"):
        estimator_for_perm = best_model.base_estimator_
    perm = permutation_importance(estimator_for_perm, X_eval, y_test, n_repeats=PERM_REPEATS, random_state=42, n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=final_features).sort_values(ascending=False)
    perm_series.head(50).to_csv(os.path.join(OUT, "permutation_importance_top50.csv"))
    print("Permutation importance saved in {:.1f}s".format(time.time() - tperm))
except Exception as e:
    print("Permutation importance failed:", e)

# -------------------------
# Repeated CV summary (final check)
# -------------------------
try:
    print("Running repeated CV for selected model (robustness check)...")
    tcv = time.time()
    cv_estimator = best_model
    if hasattr(best_model, "base_estimator_"):
        cv_estimator = best_model.base_estimator_
    # use scaled or unscaled depending on model type
    X_for_cv = X_train if best_name != "LogisticRegression" else X_train_scaled
    scores = cross_val_score(cv_estimator, X_for_cv, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS_FINAL, shuffle=True, random_state=42),
                             scoring="roc_auc_ovr", n_jobs=-1)
    cv_summary = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
    with open(os.path.join(OUT, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)
    print("CV summary saved in {:.1f}s".format(time.time() - tcv))
except Exception as e:
    print("Repeated CV failed:", e)

# -------------------------
# ROC plot (if probabilities)
# -------------------------
if hasattr(best_model, "predict_proba"):
    try:
        y_prob = best_model.predict_proba(X_eval)
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            try:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f"{best_name} - {cls}")
            except Exception:
                continue
        plt.title(f"ROC Curves for {best_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"roc_{best_name}.png"))
        plt.close()
        print("ROC plot saved")
    except Exception as e:
        print("ROC plotting failed:", e)

print("03_train_evaluate_optimized.py complete. Mode:", MODE)
