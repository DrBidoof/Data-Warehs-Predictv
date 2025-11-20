# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:08:42 2025

@author: dartb
"""

#Part 2 of project

from dotenv import load_dotenv
import os
import pandas as pd

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
    
# ==== 2. HANDLE MISSING DATA ====

df_theft5000_clean = df_theft5000.dropna(subset=[
    "OCC_YEAR", "OCC_MONTH", "OCC_DAY", "OCC_DOY", "OCC_DOW"
])

df_homicide_clean = df_homicide.copy()  # no missing values

# ==== 3. DATE CONVERSION ====

#Step 4 — Drop Unwanted Columns
#Step 5 — Encoding Categorical Features
#Step 6 — Choose a Target Variable
#Step 7 — Split into Train/Test
#Step 8 — Handle Imbalanced Classes
