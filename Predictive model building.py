# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:39:17 2025

@author: dartb

Part 3: Predictive Model Building and Evaluation
Models:
- Logistic Regression
- Decision Tree
Evaluations:
- Accuracy
- Confusion Matrix
- ROC curves
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import time

# ============================================================
# 1. LOAD EXPORTED DATA FROM PART 2
# ============================================================

X_train = pd.read_csv("exports/X_train.csv")
X_test = pd.read_csv("exports/X_test.csv")
y_train = pd.read_csv("exports/y_train.csv").values.ravel()
y_test = pd.read_csv("exports/y_test.csv").values.ravel()

# Load scaler
scaler = joblib.load("exports/scaler.pkl")
# added after goup meeting

X_train_scaled = scaler.transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# ============================================================
# 2. TRAIN MODELS
# ============================================================

# ----- Logistic Regression -----
log_reg = LogisticRegression(
    max_iter=1000,
    multi_class="ovr",
    class_weight="balanced"
)
log_reg.fit(X_train_scaled, y_train)

# ----- Decision Tree -----
tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)
tree.fit(X_train, y_train)

# ============================================================
# 3. PREDICTIONS
# ============================================================

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_tree = tree.predict(X_test)

# ============================================================
# 4. METRICS
# ============================================================

print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

print("\n\n=== DECISION TREE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))

# ============================================================
# 5. ROC CURVES (One-vs-Rest multiclass)
# ============================================================

# Binarize target classes for multi-class ROC
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)

# Logistic Regression ROC
y_prob_lr = log_reg.predict_proba(X_test_scaled)
auc_lr = roc_auc_score(y_test_bin, y_prob_lr, multi_class="ovr")

# Decision Tree ROC
y_prob_tree = tree.predict_proba(X_test)
auc_tree = roc_auc_score(y_test_bin, y_prob_tree, multi_class="ovr")

print("\nROC AUC (Logistic Regression):", auc_lr)
print("ROC AUC (Decision Tree):", auc_tree)

# Plot ROC curves for both models
plt.figure(figsize=(8, 6))
for i, cls in enumerate(classes):
    fpr_lr, tpr_lr, _ = roc_curve(y_test_bin[:, i], y_prob_lr[:, i])
    fpr_tree, tpr_tree, _ = roc_curve(y_test_bin[:, i], y_prob_tree[:, i])

    plt.plot(fpr_lr, tpr_lr, label=f"LR - {cls}")
    plt.plot(fpr_tree, tpr_tree, linestyle="--", label=f"Tree - {cls}")

plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
# Save ROC figure instead of showing
OUT_DIR = Path("outputs") / "predictive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
def _save_fig(name=None):
    ts = int(time.time()*1000)
    fname = f"{ts}" if name is None else f"{name}_{ts}"
    path = OUT_DIR / f"{fname}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

_save_fig("roc_curves")

# ============================================================
# 6. MODEL RECOMMENDATION
# ============================================================

print("\n\n=== MODEL RECOMMENDATION ===")

if auc_lr > auc_tree:
    print("Logistic Regression performs better overall based on ROC AUC.")
else:
    print("Decision Tree performs better overall based on ROC AUC.")
    
    


joblib.dump(tree, "exports/decision_tree_model.pkl")
joblib.dump(log_reg, "exports/logistic_regression_model.pkl")

print("\nModels successfully exported:")
print(" - exports/decision_tree_model.pkl")
print(" - exports/logistic_regression_model.pkl")


# ============================================================
# FEATURE IMPORTANCE -- DECISION TREE
# ============================================================

tree_importances = pd.Series(
    tree.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
tree_importances.head(15).plot(kind='bar')
plt.title("Decision Tree - Top 15 Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
_save_fig("tree_importances")

# ============================================================
# FEATURE IMPORTANCE -- LOGISTIC REGRESSION
# ============================================================

# Logistic regression uses coefficients
lr_coefs = pd.Series(
    np.abs(log_reg.coef_).mean(axis=0),
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
lr_coefs.head(15).plot(kind='bar', color='green')
plt.title("Logistic Regression - Top 15 Feature Importances (|coefficients|)")
plt.ylabel("Mean |Coefficient| Across Classes")
plt.tight_layout()
_save_fig("lr_importances")

