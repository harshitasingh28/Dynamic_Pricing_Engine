# src/feature_engineering.py
# ============================================================
# DYNAMIC PRICING ENGINE — FEATURE ENGINEERING PIPELINE
# ============================================================
# PURPOSE : Transform raw pricing data into ML-ready features
#           by encoding, scaling, deriving new features, and
#           creating interaction terms.
# INPUT   : data/raw/pricing_data.csv
# OUTPUT  : data/processed/features.csv
#           data/processed/feature_names.txt
# RUN     : python src/feature_engineering.py
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add project root to path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# SECTION 1: LOAD RAW DATA
# ============================================================

def load_raw_data():
    """
    Load the raw simulated dataset from CSV.
    Returns a pandas DataFrame.
    """
    print("📂 Loading raw dataset...")
    df = pd.read_csv(config.RAW_DATA_FILE)
    print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ============================================================
# SECTION 2: BASIC CLEANING
# ============================================================

def clean_data(df):
    """
    Perform basic data cleaning steps:
    - Drop duplicate rows
    - Handle missing values
    - Remove the ID column (not useful for ML)
    - Validate data ranges
    """
    print("\n🧹 Cleaning data...")

    original_len = len(df)

    # --- Drop exact duplicate rows ---
    # Duplicates can bias the model by making it memorize specific rows
    df = df.drop_duplicates()
    dropped = original_len - len(df)
    print(f"   Removed {dropped} duplicate rows")

    # --- Drop identifier columns ---
    # transaction_id is just a row number — has no predictive value
    if 'transaction_id' in df.columns:
        df = df.drop(columns=['transaction_id'])
        print("   Dropped: transaction_id (identifier, not a feature)")

    # --- Check for missing values ---
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found")
    else:
        print(f"   ⚠️  Missing values detected:\n{missing[missing > 0]}")
        # Fill numerical missing values with column median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   Filled '{col}' missing values with median: {median_val:.2f}")

    # --- Validate demand (target) is non-negative ---
    invalid_demand = (df['demand'] < 0).sum()
    if invalid_demand > 0:
        print(f"   ⚠️  Found {invalid_demand} negative demand values — clipping to 0")
        df['demand'] = df['demand'].clip(lower=0)

    # --- Validate price is within expected range ---
    df = df[
        (df['price'] >= config.MIN_PRICE) &
        (df['price'] <= config.MAX_PRICE)
    ]
    print(f"   ✅ Data after cleaning: {len(df):,} rows")
    return df


# ============================================================
# SECTION 3: DERIVED FEATURES (Feature Creation)
# ============================================================

def create_derived_features(df):
    """
    Create new features by mathematically combining existing ones.
    These capture relationships that raw features can't express alone.
    """
    print("\n⚙️  Creating derived features...")

    # -----------------------------------------------------------
    # 3.1 PRICE RATIO — Competitive Positioning
    # -----------------------------------------------------------
    # How expensive are we RELATIVE to the competitor?
    # < 1.0 = We are cheaper  → Competitive advantage
    # = 1.0 = Same price      → Neutral
    # > 1.0 = We are pricier  → Competitive disadvantage
    df['price_ratio'] = df['price'] / (df['competitor_price'] + 1e-9)
    # Note: We add 1e-9 (a tiny number) to prevent division by zero
    print("   ✅ price_ratio = price / competitor_price")

    # -----------------------------------------------------------
    # 3.2 PRICE DIFFERENCE — Absolute Gap
    # -----------------------------------------------------------
    # How many dollars cheaper/more expensive are we vs competitor?
    # Positive = we are more expensive
    # Negative = we are cheaper
    df['price_diff'] = df['price'] - df['competitor_price']
    print("   ✅ price_diff = price - competitor_price")

    # -----------------------------------------------------------
    # 3.3 PRICE SQUARED — Capture Non-Linear Demand Curve
    # -----------------------------------------------------------
    # The demand-price relationship is a CURVE, not a straight line.
    # price² allows linear models to capture this curve.
    # Normalized by /1000 to keep values manageable
    df['price_squared'] = (df['price'] ** 2) / 1000
    print("   ✅ price_squared = (price²) / 1000")

    # -----------------------------------------------------------
    # 3.4 INVENTORY SCARCITY — Urgency Signal
    # -----------------------------------------------------------
    # When inventory is LOW, urgency is HIGH.
    # Instead of raw inventory (1-100), we flip it:
    # scarcity = 1 - (inventory / 100)
    # Low inventory (5)  → scarcity = 0.95 (HIGH urgency)
    # High inventory (95) → scarcity = 0.05 (LOW urgency)
    df['inventory_scarcity'] = 1 - (df['inventory_level'] / 100)
    print("   ✅ inventory_scarcity = 1 - (inventory_level / 100)")

    # -----------------------------------------------------------
    # 3.5 INTERACTION: Price × Promotion
    # -----------------------------------------------------------
    # A high-priced item on promotion behaves very differently
    # from a low-priced item on promotion.
    # This feature captures that combined signal.
    df['price_x_promotion'] = df['price'] * df['is_promotion']
    print("   ✅ price_x_promotion = price × is_promotion")

    # -----------------------------------------------------------
    # 3.6 INTERACTION: Price × Peak Hour
    # -----------------------------------------------------------
    # During peak hours, people are less price-sensitive.
    # A $200 item might sell fine at 8 PM but not at 3 AM.
    df['price_x_peak'] = df['price'] * df['is_peak_hour']
    print("   ✅ price_x_peak = price × is_peak_hour")

    # -----------------------------------------------------------
    # 3.7 INTERACTION: Price × Weekend
    # -----------------------------------------------------------
    # Weekend shoppers have different price sensitivity
    # than weekday shoppers.
    df['price_x_weekend'] = df['price'] * df['is_weekend']
    print("   ✅ price_x_weekend = price × is_weekend")

    # -----------------------------------------------------------
    # 3.8 REVENUE POTENTIAL — Expected Revenue at This Price
    # -----------------------------------------------------------
    # Based purely on price × average_demand_rate
    # This is a naive estimate, but gives the model context
    # about which price points have historically generated revenue
    # We use a normalized version to prevent huge values
    df['revenue_potential'] = df['price'] * (df['customer_rating'] / 5.0)
    print("   ✅ revenue_potential = price × (rating / 5)")

    # -----------------------------------------------------------
    # 3.9 HOUR BINS — Categorical Time Groups
    # -----------------------------------------------------------
    # Instead of raw hour (0-23), group into meaningful business periods:
    # Night: 0-5, Morning: 6-11, Afternoon: 12-16, Evening: 17-23
    def get_time_period(hour):
        if 0 <= hour <= 5:
            return 0    # Night
        elif 6 <= hour <= 11:
            return 1    # Morning
        elif 12 <= hour <= 16:
            return 2    # Afternoon
        else:
            return 3    # Evening

    df['time_period'] = df['hour_of_day'].apply(get_time_period)
    print("   ✅ time_period = {0:Night, 1:Morning, 2:Afternoon, 3:Evening}")

    # -----------------------------------------------------------
    # 3.10 SEASON BINS — Categorical Seasons
    # -----------------------------------------------------------
    # Convert month into 4 seasons (winter, spring, summer, fall)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0   # Winter
        elif month in [3, 4, 5]:
            return 1   # Spring
        elif month in [6, 7, 8]:
            return 2   # Summer
        else:
            return 3   # Fall

    df['season'] = df['month'].apply(get_season)
    print("   ✅ season = {0:Winter, 1:Spring, 2:Summer, 3:Fall}")

    # -----------------------------------------------------------
    # 3.11 RATING TIER — High/Medium/Low Quality Signal
    # -----------------------------------------------------------
    # Instead of raw decimal rating, classify products into tiers
    # This gives the model a cleaner categorical signal
    def rating_tier(r):
        if r >= 4.0:
            return 2   # High rated
        elif r >= 2.5:
            return 1   # Medium rated
        else:
            return 0   # Low rated

    df['rating_tier'] = df['customer_rating'].apply(rating_tier)
    print("   ✅ rating_tier = {0:Low(<2.5), 1:Medium(2.5-4), 2:High(>4)}")

    print(f"\n   📊 Total features after derivation: {len(df.columns)}")
    return df


# ============================================================
# SECTION 4: ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_categorical_features(df):
    """
    Convert text/categorical columns into numerical format
    using One-Hot Encoding.

    WHY One-Hot and NOT Label Encoding?
    - Label: Electronics=0, Clothing=1 implies Clothing > Electronics
    - One-Hot: Each category gets its own 0/1 column — no false ordering
    """
    print("\n🔠 Encoding categorical features...")

    # One-hot encode product_category
    # drop_first=True removes one column to avoid multicollinearity
    # (if we have 5 categories, 4 columns are enough to represent all 5)
    dummies = pd.get_dummies(
        df['product_category'],
        prefix='cat',          # Column names become: cat_Electronics, cat_Food, etc.
        drop_first=True,       # Drop first category to avoid dummy variable trap
        dtype=int              # Use integer 0/1 instead of True/False
    )

    # Log the new columns created
    print(f"   One-hot encoded 'product_category' into {len(dummies.columns)} columns:")
    for col in dummies.columns:
        print(f"      → {col}")

    # Join the new columns to main dataframe
    df = pd.concat([df, dummies], axis=1)

    # Drop the original text column (no longer needed)
    df = df.drop(columns=['product_category'])
    print("   Dropped original 'product_category' column")

    return df


# ============================================================
# SECTION 5: SELECT FINAL FEATURE SET
# ============================================================

def select_features(df):
    """
    Select exactly which columns go into X (features) and y (target).

    We deliberately EXCLUDE:
    - 'demand'   → This is our TARGET (what we predict), not a feature
    - 'revenue'  → This is derived FROM demand — would cause data leakage!

    DATA LEAKAGE WARNING:
    If we include 'revenue' as a feature, the model gets to "see"
    the answer (since revenue = price × demand). This makes accuracy
    look perfect but the model would FAIL in production. Always exclude
    variables that are derived from the target.
    """
    print("\n🎯 Selecting final feature set...")

    # Columns to EXCLUDE from features
    exclude_cols = [
        'demand',           # TARGET variable
        'revenue',          # Derived from target (DATA LEAKAGE!)
    ]

    # Get all feature columns (everything that's NOT excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n   ✅ Selected {len(feature_cols)} features:")
    print(f"   🚫 Excluded: {exclude_cols} (target + leakage)")

    # Print feature list neatly
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:02d}. {col}")

    return feature_cols


# ============================================================
# SECTION 6: SCALE NUMERICAL FEATURES
# ============================================================

def scale_features(df, feature_cols, fit_scaler=True):
    """
    Apply StandardScaler to normalize numerical features.

    WHY SCALE?
    - ML models are sensitive to feature magnitude
    - price (range: 10-200) would dominate is_promotion (range: 0-1)
    - After scaling, ALL features have mean=0 and std=1
    - This makes gradient-based optimization converge faster

    WHY SAVE THE SCALER?
    - At prediction time, new data must be scaled using THE SAME
      mean and std from training data
    - If we fit a new scaler on test data, we get different values!
    - So we fit once on training data, save it, reuse for new data

    Parameters:
        df         : DataFrame with all features
        feature_cols: List of feature column names
        fit_scaler : True = fit new scaler (training phase)
                     False = use saved scaler (prediction phase)
    """
    print("\n📏 Scaling numerical features...")

    # Identify which feature columns are numerical
    # (we DON'T want to scale binary 0/1 columns — they're already normalized)
    binary_cols = [
        'is_weekend', 'is_peak_hour', 'is_promotion',
        'cat_Clothing', 'cat_Electronics', 'cat_Food', 'cat_Toys'
    ]

    # All other numeric columns get scaled
    cols_to_scale = [
        col for col in feature_cols
        if col not in binary_cols
        and df[col].dtype in [np.float64, np.int64, float, int]
    ]

    print(f"   Scaling {len(cols_to_scale)} numerical columns:")
    for col in cols_to_scale:
        print(f"      → {col}")

    if fit_scaler:
        # Create a new StandardScaler and FIT it to training data
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Save the scaler to disk so we can reuse it later
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        scaler_path = os.path.join(config.MODELS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"\n   ✅ Scaler fitted and saved → {scaler_path}")
    else:
        # Load existing scaler and TRANSFORM (don't fit again!)
        scaler_path = os.path.join(config.MODELS_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        print(f"   ✅ Loaded saved scaler from → {scaler_path}")

    return df, scaler


# ============================================================
# SECTION 7: TRAIN / TEST SPLIT
# ============================================================

def split_data(df, feature_cols):
    """
    Split data into training set and test set.

    WHY SPLIT?
    We need to EVALUATE our model on data it has NEVER seen.
    If we test on the same data we trained on, the model just
    memorizes answers — like studying only the exam questions.

    We want to simulate: "How will this model perform on NEW data?"

    SPLIT RATIO: 80% train, 20% test (set in config.py)

    WHY NOT RANDOM SHUFFLE FOR PRICING DATA?
    In real pricing, you train on past data and predict future prices.
    Ideally we'd do a time-based split. Since our data is simulated
    without timestamps, we use a random split for now.
    """
    from sklearn.model_selection import train_test_split

    print(f"\n✂️  Splitting data: {int((1-config.TEST_SIZE)*100)}% train "
          f"/ {int(config.TEST_SIZE*100)}% test...")

    # Separate features (X) from target (y)
    X = df[feature_cols]         # Feature matrix (input)
    y = df[config.TARGET_COL]    # Target vector (output = demand)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,    # 20% for testing
        random_state=config.RANDOM_SEED # Reproducible split
    )

    print(f"   ✅ Training set  : {X_train.shape[0]:,} rows × "
          f"{X_train.shape[1]} features")
    print(f"   ✅ Test set      : {X_test.shape[0]:,} rows × "
          f"{X_test.shape[1]} features")
    print(f"   ✅ Target (y) range: [{y.min()}, {y.max()}]")

    return X_train, X_test, y_train, y_test


# ============================================================
# SECTION 8: SAVE PROCESSED DATA
# ============================================================

def save_processed_data(X_train, X_test, y_train, y_test, feature_cols):
    """
    Save the processed features and targets to disk.
    This avoids re-running feature engineering every time we train.
    """
    print("\n💾 Saving processed data...")

    # Create output directory
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Combine X and y back together for saving
    train_df = X_train.copy()
    train_df['demand'] = y_train.values
    train_df.to_csv(
        os.path.join(config.PROCESSED_DATA_DIR, 'train.csv'),
        index=False
    )

    test_df = X_test.copy()
    test_df['demand'] = y_test.values
    test_df.to_csv(
        os.path.join(config.PROCESSED_DATA_DIR, 'test.csv'),
        index=False
    )

    # Save the list of feature column names (critical for consistency)
    feature_names_path = os.path.join(
        config.PROCESSED_DATA_DIR, 'feature_names.txt'
    )
    with open(feature_names_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')

    print(f"   ✅ train.csv saved → {config.PROCESSED_DATA_DIR}/train.csv")
    print(f"   ✅ test.csv  saved → {config.PROCESSED_DATA_DIR}/test.csv")
    print(f"   ✅ feature_names.txt saved → {feature_names_path}")


# ============================================================
# SECTION 9: FEATURE ENGINEERING SUMMARY REPORT
# ============================================================

def print_feature_summary(X_train, feature_cols):
    """
    Print a comprehensive summary of the final feature matrix.
    Helps us understand what the ML model will be trained on.
    """
    print("\n" + "="*60)
    print("      📋 FINAL FEATURE ENGINEERING REPORT")
    print("="*60)

    print(f"\n  Total Features   : {len(feature_cols)}")
    print(f"  Training Samples : {len(X_train):,}")

    print(f"\n  {'#':<4} {'Feature Name':<28} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")

    for i, col in enumerate(feature_cols, 1):
        mean = X_train[col].mean()
        std  = X_train[col].std()
        mn   = X_train[col].min()
        mx   = X_train[col].max()
        print(f"  {i:<4} {col:<28} {mean:>8.2f} {std:>8.2f} "
              f"{mn:>8.2f} {mx:>8.2f}")

    print("="*60)


# ============================================================
# MAIN — RUN THE FULL FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering():
    """
    Master function that runs the entire feature engineering pipeline
    in the correct order. Returns train/test splits ready for ML.
    """
    print("="*60)
    print("  ⚙️  FEATURE ENGINEERING PIPELINE")
    print("="*60)

    # Step 1: Load raw data
    df = load_raw_data()

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Create derived features
    df = create_derived_features(df)

    # Step 4: Encode categorical features
    df = encode_categorical_features(df)

    # Step 5: Select feature columns
    feature_cols = select_features(df)

    # Step 6: Scale numerical features
    df, scaler = scale_features(df, feature_cols, fit_scaler=True)

    # Step 7: Train/test split
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)

    # Step 8: Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, feature_cols)

    # Step 9: Print summary
    print_feature_summary(X_train, feature_cols)

    print("\n✅ Feature Engineering Complete!")
    print("   Your ML-ready data is saved in data/processed/")
    print("   The fitted scaler is saved in models/scaler.pkl")
    print("\n   ➡️  Ready for Model Training (Step 5)!\n")

    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()