#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_prepare_features_optimized.py
Feature engineering, deterministic leakage removal, SelectKBest pre-prune,
RFECV stability selection, imputation, scaling. Reads dataset and writes artifacts to exports/IO2/.
Saves: num_imputer.pkl, selectkbest_k.pkl, feature_selection_frequency.csv,
final_feature_list.pkl, scaler.pkl, X_train.csv, X_test.csv, y_train.csv, y_test.csv.
"""
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression
import joblib

load_dotenv()
theft_path = os.getenv("THEFT5000")
if theft_path is None or not os.path.exists(theft_path):
    raise ValueError("THEFT5000 env var must point to the dataset (set in .env)")

# Ensure folders
os.makedirs("exports", exist_ok=True)
os.makedirs(os.path.join("exports", "IO1"), exist_ok=True)
os.makedirs(os.path.join("exports", "IO2"), exist_ok=True)
os.makedirs("sid", exist_ok=True)

OUT = os.path.join("exports", "IO2")

# Read dataset
df = pd.read_csv(theft_path, low_memory=False)
print("Loaded dataset shape:", df.shape)

# --- Feature engineering (dates, time buckets, season, weekend)
date_cols = [c for c in df.columns if "date" in c.lower()]
if date_cols:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["YEAR"] = df[date_col].dt.year
    df["MONTH"] = df[date_col].dt.month
    df["DAY"] = df[date_col].dt.day
    df["DOW_NUM"] = df[date_col].dt.weekday
else:
    if "OCC_YEAR" in df.columns:
        df["YEAR"] = df["OCC_YEAR"]
    if "OCC_MONTH" in df.columns:
        df["MONTH"] = df["OCC_MONTH"]
    if "OCC_DAY" in df.columns:
        df["DAY"] = df["OCC_DAY"]
    if "OCC_DOW" in df.columns:
        df["DOW_NUM"] = df["OCC_DOW"]

def time_of_day_safe(x):
    try:
        h = int(x)
    except Exception:
        return np.nan
    if 0 <= h < 6:
        return "Night"
    if 6 <= h < 12:
        return "Morning"
    if 12 <= h < 18:
        return "Afternoon"
    return "Evening"

if "OCC_HOUR" in df.columns:
    df["TIME_OF_DAY"] = df["OCC_HOUR"].apply(time_of_day_safe)

def season_of_month(m):
    if pd.isna(m):
        return np.nan
    m = int(m)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"

if "MONTH" in df.columns:
    df["SEASON"] = df["MONTH"].apply(season_of_month)
if "DOW_NUM" in df.columns:
    df["IS_WEEKEND"] = df["DOW_NUM"].apply(lambda x: 1 if pd.notnull(x) and int(x) >= 5 else 0)

# Spatial clustering if coordinates exist
if {"LAT_WGS84", "LONG_WGS84"}.issubset(df.columns):
    coords = df[["LAT_WGS84", "LONG_WGS84"]].dropna()
    if not coords.empty:
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        df.loc[coords.index, "ZONE"] = labels.astype(int)

# --- Target and cleaning
target_col = "PREMISES_TYPE"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found")
df_model = df.dropna(subset=[target_col]).copy()
drop_cols = ["OBJECTID", "EVENT_UNIQUE_ID", "OCC_DATE", "REPORT_DATE"]
df_model = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")

# --- Build X, y and encode
X = df_model.drop(columns=[target_col])
y = df_model[target_col].astype(str)
X_encoded = pd.get_dummies(X, drop_first=True)
print("Encoded shape before leakage removal:", X_encoded.shape)

# --- Deterministic leakage removal
leakage_tokens = [
    "LOCATION_TYPE", "OFFENCE", "MCI_CATEGORY",
    "DIVISION", "HOOD_", "NEIGHBOURHOOD",
    "ZONE", "LAT_WGS84", "LONG_WGS84", "x", "y"
]
cols_to_drop = [c for c in X_encoded.columns if any(tok in c for tok in leakage_tokens)]
if cols_to_drop:
    X_encoded = X_encoded.drop(columns=cols_to_drop, errors="ignore")
print("Shape after leakage removal:", X_encoded.shape)

# --- Imputation and fillna
numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_encoded[numeric_cols] = num_imputer.fit_transform(X_encoded[numeric_cols])
    joblib.dump(num_imputer, os.path.join(OUT, "num_imputer.pkl"))
X_encoded = X_encoded.fillna(0)

# Save small head for audit
X_encoded.head(5).to_csv(os.path.join(OUT, "X_encoded_head.csv"), index=False)

# --- Pre-pruning: SelectKBest
n_features = X_encoded.shape[1]
k_initial = max(50, min(1000, n_features // 2))
selector_k = SelectKBest(score_func=f_classif, k=k_initial)
selector_k.fit(X_encoded, y)
cols_kbest = X_encoded.columns[selector_k.get_support()].tolist()
X_kbest = X_encoded[cols_kbest]
joblib.dump(selector_k, os.path.join(OUT, "selectkbest_k.pkl"))
print("SelectKBest reduced features to:", X_kbest.shape[1])

# --- Wrapper: RFECV stability across seeds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
estimator = LogisticRegression(max_iter=2000, solver="liblinear", penalty="l2", class_weight="balanced")
seeds = [0, 7, 42, 99, 123]
supports = []
for seed in seeds:
    rfecv = RFECV(estimator=estimator, step=1,
                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                  scoring="roc_auc_ovr", n_jobs=-1, verbose=0)
    rfecv.fit(X_kbest, y)
    supports.append(rfecv.support_)
    print(f"RFECV seed {seed} selected {supports[-1].sum()} features")

supports_arr = np.vstack(supports)
freq = supports_arr.mean(axis=0)
freq_series = pd.Series(freq, index=X_kbest.columns).sort_values(ascending=False)
freq_series.to_csv(os.path.join(OUT, "feature_selection_frequency.csv"))

stable_threshold = 0.6
stable_cols = freq_series[freq_series >= stable_threshold].index.tolist()
if len(stable_cols) == 0:
    stable_cols = X_kbest.columns[supports_arr[-1].astype(bool)].tolist()
print("Stable features kept:", len(stable_cols))
joblib.dump(stable_cols, os.path.join(OUT, "final_feature_list.pkl"))

# Final feature matrix
X_final = X_kbest[stable_cols]

# Train/test split and scaler
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train)
joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))

# Save splits and final feature list to IO2
X_train.to_csv(os.path.join(OUT, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUT, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUT, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUT, "y_test.csv"), index=False)
pd.Series(stable_cols).to_csv(os.path.join(OUT, "final_features.csv"), index=False, header=False)

print("Feature preparation artifacts saved to exports/IO2/")
