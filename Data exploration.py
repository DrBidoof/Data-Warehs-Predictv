# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:07:00 2025

@author: dartb
"""

from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()  

homicide_path = os.getenv("HOMICIDE")

theft5000_path = os.getenv("THEFT5000")

## read
df_homicide = pd.read_csv(homicide_path)
df_theft5000 = pd.read_csv(theft5000_path)

# ==== 1. BASIC DATA OVERVIEW ====

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
print(df_homicide.describe())

print("\n=== Numeric Stats: Theft 5000 ===")
print(df_theft5000.describe())
