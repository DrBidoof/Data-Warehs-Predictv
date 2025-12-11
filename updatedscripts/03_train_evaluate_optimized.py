#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_evaluate_optimized.py
Hyperparameter search, calibration, permutation importance, repeated CV.
Reads artifacts from exports/IO2/ and writes outputs to exports/IO3/.
Saves: gs_logistic.pkl, rs_randomforest.pkl, gs_decisiontree.pkl, best_model_*.pkl,
model_evaluation_summary.json, permutation_importance_top50.csv, cv_summary.json, roc_*.png
"""
import os
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

# Ensure folders
os.makedirs("exports", exist_ok=True)
os.makedirs(os.path.join("exports", "IO2"), exist_ok=True)
os.makedirs(os.path.join("exports", "IO3"), exist_ok=True)
os.makedirs("sid", exist_ok=True)

IN = os.path.join("exports", "IO2")
OUT = os.path.join("exports", "IO3")

# Required artifacts
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

X_train = pd.read_csv(os.path.join(IN, "X_train.csv"))
X_test = pd.read_csv(os.path.join(IN, "X_test.csv"))
y_train = pd.read_csv(os.path.join(IN, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(IN, "y_test.csv")).values.ravel()
scaler = joblib.load(os.path.join(IN, "scaler.pkl"))
final_features = joblib.load(os.path.join(IN, "final_feature_list.pkl"))

# Defensive alignment
X_train = X_train[final_features]
X_test = X_test[final_features]

# Scale for linear models
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LogisticRegression grid
lr = LogisticRegression(multi_class="ovr", class_weight="balanced", max_iter=2000)
lr_grid = {"C": [0.01, 0.1, 1.0], "penalty": ["l2"], "solver": ["liblinear"]}
gs_lr = GridSearchCV(lr, lr_grid, scoring="roc_auc_ovr", cv=cv, n_jobs=-1, verbose=1)
gs_lr.fit(X_train_scaled, y_train)
joblib.dump(gs_lr, os.path.join(OUT, "gs_logistic.pkl"))

# Randomized search for RandomForest
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
rf_param_dist = {
    "n_estimators": [200, 500, 800],
    "max_depth": [None, 15, 30],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=30, scoring="roc_auc_ovr", cv=cv, n_jobs=-1, random_state=42, verbose=1)
rs_rf.fit(X_train, y_train)
joblib.dump(rs_rf, os.path.join(OUT, "rs_randomforest.pkl"))

# DecisionTree baseline grid
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_grid = {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]}
gs_dt = GridSearchCV(dt, dt_grid, scoring="roc_auc_ovr", cv=cv, n_jobs=-1, verbose=1)
gs_dt.fit(X_train, y_train)
joblib.dump(gs_dt, os.path.join(OUT, "gs_decisiontree.pkl"))

# Evaluate best estimators on test set
models = {
    "LogisticRegression": gs_lr.best_estimator_,
    "RandomForest": rs_rf.best_estimator_,
    "DecisionTree": gs_dt.best_estimator_
}

results = {}
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
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

# Select best model
best_name = None
best_score = -np.inf
for name, res in results.items():
    score = res["roc_auc_ovr"] if res["roc_auc_ovr"] is not None else res["accuracy"]
    if score is not None and score > best_score:
        best_score = score
        best_name = name

best_model = models[best_name]

# Calibrate if RandomForest selected
if best_name == "RandomForest":
    calibrated = CalibratedClassifierCV(best_model, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)
    best_model = calibrated

# Recompute final metrics on test set
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

# Save evaluation summary
with open(os.path.join(OUT, "model_evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"final_results": final_results, "cv_best_scores": {"lr": gs_lr.best_score_, "rf": rs_rf.best_score_, "dt": gs_dt.best_score_}}, f, indent=2)

# Permutation importance (explainability)
try:
    estimator_for_perm = best_model
    if hasattr(best_model, "base_estimator_"):
        estimator_for_perm = best_model.base_estimator_
    perm = permutation_importance(estimator_for_perm, X_eval, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=final_features).sort_values(ascending=False)
    perm_series.head(50).to_csv(os.path.join(OUT, "permutation_importance_top50.csv"))
except Exception:
    pass

# Repeated CV summary for selected model
try:
    cv_estimator = best_model
    if hasattr(best_model, "base_estimator_"):
        cv_estimator = best_model.base_estimator_
    scores = cross_val_score(cv_estimator, X_train if best_name != "LogisticRegression" else X_train_scaled,
                             y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                             scoring="roc_auc_ovr", n_jobs=-1)
    cv_summary = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
    with open(os.path.join(OUT, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)
except Exception:
    pass

# ROC plot for final model (if probabilities available)
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
    except Exception:
        pass

print("Training and evaluation complete. Artifacts saved to exports/IO3/")
