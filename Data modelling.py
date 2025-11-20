# -*- coding: utf-8 -*-
"""
Part 2: Data Modelling (Theft Over $5000)
Goal: Prepare dataset for predicting PREMISES_TYPE
This script performs:
- Data cleaning
- Feature engineering
- Categorical encoding
- Scaling (optional)
- Train/test split
"""

from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# 0. LOAD PATHS SAFELY
# ============================================================

load_dotenv()

homicide_path = os.getenv("HOMICIDE")
theft5000_path = os.getenv("THEFT5000")

if theft5000_path is None:
    raise ValueError("Environment variable THEFT5000 is not set. Check .env file.")

if not os.path.exists(theft5000_path):
    raise FileNotFoundError(f"Theft CSV not found at: {theft5000_path}")

# ============================================================
# 1. READ THEFT DATA
# ============================================================

df_theft = pd.read_csv(theft5000_path)
print("Original theft dataset shape:", df_theft.shape)

# ============================================================
# 2. HANDLE MISSING DATA
# ============================================================

df_theft_clean = df_theft.dropna(subset=[
    "OCC_YEAR", "OCC_MONTH", "OCC_DAY", "OCC_DOY", "OCC_DOW"
])

print("After dropping missing OCC_* rows:", df_theft_clean.shape)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def extract_date_features(df, date_col):
    """Convert date column to datetime and add YEAR, MONTH, DAY, DOW."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["YEAR"] = df[date_col].dt.year
    df["MONTH"] = df[date_col].dt.month
    df["DAY"] = df[date_col].dt.day
    df["DOW_NUM"] = df[date_col].dt.weekday
    return df

df_theft_fe = extract_date_features(df_theft_clean, "OCC_DATE")

# ---- TIME OF DAY BUCKETS ----
def time_of_day(hour):
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"

df_theft_fe["TIME_OF_DAY"] = df_theft_fe["OCC_HOUR"].apply(time_of_day)

# ---- SEASON ----
def season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df_theft_fe["SEASON"] = df_theft_fe["MONTH"].apply(season)

# ---- WEEKEND FLAG ----
df_theft_fe["IS_WEEKEND"] = df_theft_fe["DOW_NUM"].apply(lambda x: 1 if x >= 5 else 0)

# ---- SPATIAL CLUSTER (ZONE) ----
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_theft_fe["ZONE"] = kmeans.fit_predict(df_theft_fe[["LAT_WGS84", "LONG_WGS84"]])

# ============================================================
# 4. DROP UNUSED COLUMNS
# ============================================================

cols_to_drop = [
    "OBJECTID",
    "EVENT_UNIQUE_ID",
    "OCC_DATE",
    "REPORT_DATE"
]

df_theft_fe = df_theft_fe.drop(columns=cols_to_drop, errors="ignore")

# ============================================================
# 5. SELECT TARGET VARIABLE
# ============================================================

target_col = "PREMISES_TYPE"

df_model = df_theft_fe.dropna(subset=[target_col]).copy()
print("Dataset after selecting target:", df_model.shape)

# ============================================================
# 6. BUILD X AND y
# ============================================================

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# ---- ONE-HOT ENCODING ----
X_encoded = pd.get_dummies(X, drop_first=True)
print("Encoded feature matrix shape:", X_encoded.shape)

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ============================================================
# 8. OPTIONAL: SCALING
# ============================================================

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPart 2 complete — Dataset transformed and ready for modelling.")
print("Proceed to Part 3 for model training and evaluation.")
