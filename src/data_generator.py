# src/data_generator.py
# ============================================================
# DYNAMIC PRICING ENGINE — DATA GENERATOR
# ============================================================
# PURPOSE : Simulate a realistic e-commerce pricing dataset
#           using mathematical demand curves and real-world
#           pricing factors.
# OUTPUT  : data/raw/pricing_data.csv (10,000 rows)
# ============================================================

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# STEP 1: SET THE RANDOM SEED
# ============================================================
# This ensures that every time you run this script, you get
# EXACTLY the same "random" data. This is crucial for
# reproducibility — your teammate should get the same dataset.
np.random.seed(config.RANDOM_SEED)

# ============================================================
# STEP 2: DEFINE SIMULATION PARAMETERS
# ============================================================

N = config.N_SAMPLES  # Total rows to generate (10,000)

# --- Price Parameters ---
# Products will be priced between $5 and $500
# np.random.uniform picks random prices in this range
BASE_PRICE_MIN = 10.0
BASE_PRICE_MAX = 200.0

# --- Demand Parameters ---
BASE_DEMAND      = 200   # Average units sold at a "neutral" price
MAX_DEMAND       = 500   # Demand can never exceed this (physical limit)
PRICE_SENSITIVITY = 0.8  # How strongly demand reacts to price changes
                          # Higher = demand drops faster with price increase

# --- Effect Strengths ---
COMPETITOR_EFFECT = 0.5  # How much competitor pricing influences demand
TIME_EFFECT       = 40   # Demand boost during peak hours
SEASON_EFFECT     = 30   # Demand boost during peak seasons
INVENTORY_EFFECT  = 20   # Scarcity effect on demand urgency
PROMO_EFFECT      = 50   # Demand boost when promotions are active

# ============================================================
# STEP 3: GENERATE INDEPENDENT FEATURES (INPUTS)
# ============================================================
# These are the "causes" — factors that influence demand and price

print("🔧 Generating features...")

# --- 3.1 PRICE ---
# Our product price: randomly chosen between $10 and $200
price = np.random.uniform(BASE_PRICE_MIN, BASE_PRICE_MAX, N)

# --- 3.2 COMPETITOR PRICE ---
# Competitor prices their product slightly differently from ours
# Their price = our price + some random variation (-30% to +30%)
competitor_price = price * np.random.uniform(0.7, 1.3, N)

# --- 3.3 DAY OF WEEK ---
# 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
day_of_week = np.random.randint(0, 7, N)

# --- 3.4 IS WEEKEND ---
# Binary flag: 1 if Saturday or Sunday, 0 otherwise
# Weekends = higher demand for most consumer products
is_weekend = (day_of_week >= 5).astype(int)

# --- 3.5 HOUR OF DAY ---
# Hour from 0 (midnight) to 23 (11 PM)
hour_of_day = np.random.randint(0, 24, N)

# --- 3.6 IS PEAK HOUR ---
# Peak hours: 9AM–12PM (morning rush) or 5PM–9PM (evening rush)
# These are high-traffic shopping windows
is_peak_hour = (
    ((hour_of_day >= 9) & (hour_of_day <= 12)) |
    ((hour_of_day >= 17) & (hour_of_day <= 21))
).astype(int)

# --- 3.7 MONTH ---
# Month from 1 (January) to 12 (December)
month = np.random.randint(1, 13, N)

# --- 3.8 SEASON FACTOR ---
# Demand is higher in certain months (holiday shopping, summer, etc.)
# We model this as a continuous value using a sine wave
# sin wave peaks in summer (month 6-7) and winter holidays (month 12)
season_factor = np.sin(2 * np.pi * month / 12) + \
                np.sin(4 * np.pi * month / 12)
# Result: values between -2 and +2, peaking in summer and winter

# --- 3.9 INVENTORY LEVEL ---
# How many units are left in stock (1 to 100)
# Low inventory = scarcity = demand urgency increases
inventory_level = np.random.randint(1, 101, N)

# --- 3.10 IS PROMOTION ---
# Binary flag: 1 if a promotional campaign is active
# About 20% of transactions have active promotions
is_promotion = np.random.choice([0, 1], N, p=[0.8, 0.2])

# --- 3.11 PRODUCT CATEGORY ---
# Different product types have different base demand levels
# We use 5 categories: Electronics, Clothing, Food, Books, Toys
categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys']
product_category = np.random.choice(categories, N)

# --- 3.12 CATEGORY MULTIPLIER ---
# Each category has a different sensitivity to price
# Electronics: less sensitive (people pay more), Food: very sensitive
category_multiplier = {
    'Electronics' : 0.6,  # People are LESS price sensitive for electronics
    'Clothing'    : 0.9,
    'Food'        : 1.2,  # People are MORE price sensitive for food
    'Books'       : 1.0,
    'Toys'        : 0.8
}
# Map each row's category to its multiplier
cat_mult = np.array([category_multiplier[c] for c in product_category])

# --- 3.13 CUSTOMER RATING ---
# Product rating between 1.0 and 5.0
# Higher ratings → higher demand (people trust good products)
customer_rating = np.random.uniform(1.0, 5.0, N)

# ============================================================
# STEP 4: CALCULATE DEMAND (THE TARGET VARIABLE)
# ============================================================
# This is the core math — demand is a function of all above features
# This formula encodes real-world economics into our dataset

print("📐 Calculating demand using economic model...")

# --- Random noise ---
# Real-world demand is never perfectly predictable
# We add Gaussian (bell-curve) noise: mean=0, std=15
noise = np.random.normal(0, 15, N)

# --- Compute raw demand using our formula ---
raw_demand = (
    BASE_DEMAND                                        # Start with base
    - (PRICE_SENSITIVITY * cat_mult * price)          # Price effect (negative)
    + (COMPETITOR_EFFECT * (competitor_price - price))# Competitor effect
    + (TIME_EFFECT * is_peak_hour)                    # Peak hour boost
    + (SEASON_EFFECT * season_factor)                 # Season effect
    + (INVENTORY_EFFECT * (1.0 / inventory_level))    # Scarcity effect
    + (PROMO_EFFECT * is_promotion)                   # Promotion boost
    + (10 * (customer_rating - 3.0))                  # Rating effect
    + noise                                            # Random variation
)

# --- Clip demand to realistic range [0, MAX_DEMAND] ---
# Demand cannot be negative or unrealistically high
demand = np.clip(raw_demand, 0, MAX_DEMAND).astype(int)

# ============================================================
# STEP 5: CALCULATE REVENUE (BUSINESS METRIC)
# ============================================================
# Revenue = Price × Demand
# This will be useful for business optimization later
revenue = price * demand

# ============================================================
# STEP 6: ASSEMBLE THE DATAFRAME
# ============================================================

print("🗃️  Assembling dataset...")

df = pd.DataFrame({
    # --- Identifiers ---
    'transaction_id'   : range(1, N + 1),

    # --- Price Features ---
    'price'            : np.round(price, 2),
    'competitor_price' : np.round(competitor_price, 2),

    # --- Time Features ---
    'day_of_week'      : day_of_week,
    'is_weekend'       : is_weekend,
    'hour_of_day'      : hour_of_day,
    'is_peak_hour'     : is_peak_hour,
    'month'            : month,
    'season_factor'    : np.round(season_factor, 4),

    # --- Product Features ---
    'product_category' : product_category,
    'inventory_level'  : inventory_level,
    'is_promotion'     : is_promotion,
    'customer_rating'  : np.round(customer_rating, 2),

    # --- Target & Business Metrics ---
    'demand'           : demand,
    'revenue'          : np.round(revenue, 2)
})

# ============================================================
# STEP 7: SAVE THE DATASET
# ============================================================

# Make sure the output folder exists
os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

# Save to CSV
df.to_csv(config.RAW_DATA_FILE, index=False)

print(f"\n✅ Dataset saved to: {config.RAW_DATA_FILE}")

# ============================================================
# STEP 8: PRINT A SUMMARY REPORT
# ============================================================

print("\n" + "="*55)
print("         📊 DATASET SUMMARY REPORT")
print("="*55)
print(f"  Total Rows (Transactions) : {len(df):,}")
print(f"  Total Columns (Features)  : {len(df.columns)}")
print(f"\n  📦 PRICE STATISTICS")
print(f"     Min Price   : ${df['price'].min():.2f}")
print(f"     Max Price   : ${df['price'].max():.2f}")
print(f"     Mean Price  : ${df['price'].mean():.2f}")
print(f"\n  📈 DEMAND STATISTICS")
print(f"     Min Demand  : {df['demand'].min()} units")
print(f"     Max Demand  : {df['demand'].max()} units")
print(f"     Mean Demand : {df['demand'].mean():.1f} units")
print(f"\n  💰 REVENUE STATISTICS")
print(f"     Min Revenue : ${df['revenue'].min():.2f}")
print(f"     Max Revenue : ${df['revenue'].max():.2f}")
print(f"     Mean Revenue: ${df['revenue'].mean():.2f}")
print(f"\n  🏷️  CATEGORY DISTRIBUTION")
print(df['product_category'].value_counts().to_string())
print(f"\n  🔖 PROMOTIONS ACTIVE")
promo_pct = df['is_promotion'].mean() * 100
print(f"     {promo_pct:.1f}% of transactions had active promotions")
print("="*55)

# Show first 5 rows
print("\n  🔍 FIRST 5 ROWS OF DATASET:")
print(df.head().to_string())
print("\n✅ Data generation complete!")