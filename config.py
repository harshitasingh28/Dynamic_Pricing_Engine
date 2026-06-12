# config.py
# ============================================================
# GLOBAL CONFIGURATION FILE
# All settings for the project live here.
# Changing one value here updates it everywhere automatically.
# ============================================================

import os

# ----------------------------
# PROJECT ROOT PATH
# ----------------------------
# os.path.dirname(__file__) gives us the folder where config.py lives
# That IS the project root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# DATA PATHS
# ----------------------------
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# File names for our datasets
RAW_DATA_FILE       = os.path.join(RAW_DATA_DIR, "pricing_data.csv")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "features.csv")

# ----------------------------
# MODEL PATHS
# ----------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# REPORTS & FIGURES
# ----------------------------
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# ----------------------------
# DATASET SIMULATION SETTINGS
# ----------------------------
RANDOM_SEED  = 42        # For reproducibility (same random numbers every time)
N_SAMPLES    = 10000     # Number of rows in our simulated dataset

# ----------------------------
# MODEL TRAINING SETTINGS
# ----------------------------
TEST_SIZE    = 0.2       # 20% data for testing, 80% for training
TARGET_COL   = "demand"  # What we're predicting

# ----------------------------
# BUSINESS LOGIC SETTINGS
# ----------------------------
MIN_PRICE    = 5.0       # Minimum allowed price (in dollars)
MAX_PRICE    = 500.0     # Maximum allowed price (in dollars)