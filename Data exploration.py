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