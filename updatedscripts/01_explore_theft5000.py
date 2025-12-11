#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_explore_theft5000.py
EDA and diagnostics. Writes artifacts to exports/IO1/ and ensures exports/ and sid/ exist.
Saves: column_summary.csv, numeric_summary.csv, missingness.csv,
categorical_top_values.txt, date_parse_info.csv, sample_head_200.csv, plots.
"""
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
theft_path = os.getenv("THEFT5000")
if theft_path is None:
    raise ValueError("Environment variable THEFT5000 is not set. Add THEFT5000=/path/to/file.csv to .env")
if not os.path.exists(theft_path):
    raise FileNotFoundError(f"Theft CSV not found at: {theft_path}")

# Ensure folders
os.makedirs("exports", exist_ok=True)
os.makedirs(os.path.join("exports", "IO1"), exist_ok=True)
os.makedirs("sid", exist_ok=True)

OUT = os.path.join("exports", "IO1")

# Read dataset defensively
df = pd.read_csv(theft_path, low_memory=False)
print("Loaded dataset shape:", df.shape)

# Column summary
col_summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null_count": df.notnull().sum(),
    "unique_values": df.nunique(dropna=False)
})
col_summary.to_csv(os.path.join(OUT, "column_summary.csv"))

# Numeric summary
num = df.select_dtypes(include=[np.number])
if not num.empty:
    num.describe().T.to_csv(os.path.join(OUT, "numeric_summary.csv"))

# Missingness
missing_counts = df.isnull().sum().sort_values(ascending=False)
missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_df = pd.concat([missing_counts, missing_pct], axis=1)
missing_df.columns = ["missing_count", "missing_pct"]
missing_df.to_csv(os.path.join(OUT, "missingness.csv"))

# Top categorical values (reasonable cardinality)
cat_candidates = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=False) < 500]
with open(os.path.join(OUT, "categorical_top_values.txt"), "w", encoding="utf-8") as f:
    for c in cat_candidates:
        f.write(f"=== {c} (top 20) ===\n")
        f.write(df[c].value_counts(dropna=False).head(20).to_string())
        f.write("\n\n")

# Date parse check
date_cols = [c for c in df.columns if "date" in c.lower()]
if date_cols:
    date_col = date_cols[0]
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    parsed_info = {
        "date_column": date_col,
        "parsed_non_null": int(parsed.notnull().sum()),
        "earliest": str(parsed.min()),
        "latest": str(parsed.max())
    }
    pd.DataFrame([parsed_info]).to_csv(os.path.join(OUT, "date_parse_info.csv"))

# Save sample head
df.head(200).to_csv(os.path.join(OUT, "sample_head_200.csv"), index=False)

# Lightweight plots saved to IO1
try:
    plt.figure(figsize=(10, 6))
    top_missing = missing_df.head(20)
    sns.barplot(x=top_missing.index.astype(str), y=top_missing["missing_count"])
    plt.xticks(rotation=90)
    plt.title("Top 20 Missing Columns")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "missing_top20.png"))
    plt.close()

    if "PREMISES_TYPE" in df.columns:
        plt.figure(figsize=(10, 6))
        df["PREMISES_TYPE"].value_counts().head(30).plot(kind="bar")
        plt.title("PREMISES_TYPE (top 30)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "premises_top30.png"))
        plt.close()
except Exception:
    pass

print("EDA artifacts saved to exports/IO1/")
