# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:07:00 2025

@author: dartb
"""

from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

load_dotenv()

# ==== 0. LOAD PATHS SAFELY ====

homicide_path = os.getenv("HOMICIDE")
theft5000_path = os.getenv("THEFT5000")

if homicide_path is None:
    raise ValueError("Environment variable HOMICIDE is not set. Check your .env file.")

if theft5000_path is None:
    raise ValueError("Environment variable THEFT5000 is not set. Check your .env file.")

if not os.path.exists(homicide_path):
    raise FileNotFoundError(f"Homicide CSV not found at: {homicide_path}")

if not os.path.exists(theft5000_path):
    raise FileNotFoundError(f"Theft CSV not found at: {theft5000_path}")

# ==== 1. READ DATA ====

df_homicide = pd.read_csv(homicide_path)
df_theft5000 = pd.read_csv(theft5000_path)

# ==== 2. BASIC DATA OVERVIEW ====

print("===== HOMICIDE DATA PREVIEW =====")
print(df_homicide.head(10))

print("\n===== THEFT OVER $5000 DATA PREVIEW =====")
print(df_theft5000.head(10))

# Column names
print("\nHomicide Columns:", df_homicide.columns.tolist())
print("Theft Columns:", df_theft5000.columns.tolist())

# Data types
print("\nHomicide Data Types:")
print(df_homicide.dtypes)

print("\nTheft Data Types:")
print(df_theft5000.dtypes)

# Basic descriptive statistics
print("\n=== Numeric Stats: Homicide ===")
print(df_homicide.select_dtypes(include="number").describe())

print("\n=== Numeric Stats: Theft 5000 ===")
print(df_theft5000.select_dtypes(include="number").describe())

# ==== 3. STATISTICS & CORRELATIONS ====

print("\n=== HOMICIDE Mean Values ===")
print(df_homicide.select_dtypes(include="number").mean())

print("\n=== THEFT Mean Values ===")
print(df_theft5000.select_dtypes(include="number").mean())

print("\n=== HOMICIDE Correlations ===")
corr_homicide = df_homicide.select_dtypes(include="number").corr()
print(corr_homicide)

print("\n=== THEFT Correlations ===")
corr_theft = df_theft5000.select_dtypes(include="number").corr()
print(corr_theft)

# ==== 4. MISSING DATA (COUNTS & PERCENTAGES) ====

print("\n=== Missing Data: HOMICIDE ===")
print(df_homicide.isnull().sum())

print("\n=== Missing Data %: HOMICIDE ===")
print(df_homicide.isnull().mean() * 100)

print("\n=== Missing Data: THEFT ===")
print(df_theft5000.isnull().sum())

print("\n=== Missing Data %: THEFT ===")
print(df_theft5000.isnull().mean() * 100)

# ===== 4B. MISSING DATA BAR CHARTS (OPTION 1) =====

for name, df in [("Homicide", df_homicide), ("Theft Over $5000", df_theft5000)]:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print(f"\nNo missing values in {name} dataset – skipping missing-data bar chart.")
        continue

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing.index.astype(str), y=missing.values)
    plt.xticks(rotation=90)
    plt.ylabel("Number of Missing Values")
    plt.title(f"Missing Values per Column – {name} Dataset")
    plt.tight_layout()
    plt.show()

# ==== 5. BASIC VISUALIZATIONS (CATEGORICAL & CORR HEATMAPS) ====

# --- Homicide by Type ---
plt.figure(figsize=(10, 6))
print("Homicide columns:", df_homicide.columns.tolist())

if "HOMICIDE_TYPE" in df_homicide.columns:
    sns.countplot(data=df_homicide, x="HOMICIDE_TYPE")
    plt.title("Homicide Counts by Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Homicide_Type' not found. Available columns:", df_homicide.columns.tolist())

# --- Theft by Object of Theft ---
plt.figure(figsize=(10, 6))

if "OFFENCE" in df_theft5000.columns:
    sns.countplot(data=df_theft5000, x="OFFENCE")
    plt.title("Theft Over $5000 by Item Stolen")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'OFFENCE' not found. Available columns:", df_theft5000.columns.tolist())

# --- Correlation heatmaps (only if at least 2 numeric cols) ---

if corr_homicide.shape[0] > 1 and corr_homicide.shape[1] > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_homicide, annot=True, cmap="coolwarm")
    plt.title("Homicide Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric columns for homicide correlation heatmap.")

if corr_theft.shape[0] > 1 and corr_theft.shape[1] > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_theft, annot=True, cmap="coolwarm")
    plt.title("Theft Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric columns for theft correlation heatmap.")

# ===== 6. HOMICIDES BY YEAR =====

possible_year_cols = ["Year", "year", "YEAR", "Reported_Year", "OCC_YEAR"]

year_col_h = None
for col in possible_year_cols:
    if col in df_homicide.columns:
        year_col_h = col
        break

if year_col_h:
    plt.figure(figsize=(10, 5))
    (
        df_homicide.groupby(year_col_h)
        .size()
        .plot(kind="line", marker="o")
    )
    plt.title("Homicides Over Time")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No year column found in homicide dataset.")

# ===== 7. THEFT OVER TIME =====

year_col_t = None
for col in possible_year_cols:
    if col in df_theft5000.columns:
        year_col_t = col
        break

if year_col_t:
    plt.figure(figsize=(10, 5))
    (
        df_theft5000.groupby(year_col_t)
        .size()
        .plot(kind="line", marker="o", color="green")
    )
    plt.title("Theft Over $5000 Over Time")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No year column found in theft dataset.")

# ===== 8. SEASONAL PATTERNS =====

def add_date_parts(df):
    """Add Year, Month, Month_Name columns if a date column exists."""
    date_cols = [col for col in df.columns if "date" in col.lower()]

    if len(date_cols) == 0:
        print("No date-like column found, skipping date parts.")
        return df  # return df unchanged if no date column

    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Month_Name"] = df[date_col].dt.month.apply(
        lambda x: calendar.month_abbr[int(x)] if pd.notnull(x) else None
    )

    return df

# Apply to both datasets (if date column exists)
df_homicide = add_date_parts(df_homicide)
df_theft5000 = add_date_parts(df_theft5000)

# --- Seasonal Plot: Homicides ---
if "Month_Name" in df_homicide.columns:
    plt.figure(figsize=(12, 5))
    sns.countplot(
        data=df_homicide,
        x="Month_Name",
        order=calendar.month_abbr[1:]  # Jan–Dec
    )
    plt.title("Seasonal Pattern of Homicides")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("No usable date column for homicide seasonal pattern.")

# --- Seasonal Plot: Theft ---
if "Month_Name" in df_theft5000.columns:
    plt.figure(figsize=(12, 5))
    sns.countplot(
        data=df_theft5000,
        x="Month_Name",
        order=calendar.month_abbr[1:]  # Jan–Dec
    )
    plt.title("Seasonal Pattern of Theft Over $5000")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("No usable date column for theft seasonal pattern.")
