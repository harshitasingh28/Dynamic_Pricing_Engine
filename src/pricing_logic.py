# src/pricing_logic.py
# ============================================================
# DYNAMIC PRICING ENGINE — BUSINESS PRICING LOGIC
# ============================================================
# PURPOSE : Use the trained ML model to find the optimal price
#           that maximizes revenue/profit under business constraints.
#
# CORE IDEA:
#   1. Take current market conditions as input
#   2. Sweep through candidate prices ($10 to $200)
#   3. Predict demand at each price using ML model
#   4. Calculate revenue = price × demand
#   5. Return the price with maximum revenue
#
# INPUT   : Trained XGBoost model (models/xgboost.pkl)
#           Fitted scaler        (models/scaler.pkl)
#           Feature names        (data/processed/feature_names.txt)
# OUTPUT  : Optimal price recommendation + full analysis report
# RUN     : python src/pricing_logic.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# GLOBAL PLOT SETTINGS
# ============================================================
sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'

COLORS = {
    'primary'   : '#2E86AB',
    'secondary' : '#E84855',
    'accent'    : '#F9A03F',
    'success'   : '#3BB273',
    'purple'    : '#7B2D8B',
    'dark'      : '#1C1C1E',
    'gold'      : '#FFD700',
}


# ============================================================
# SECTION 1: MARKET CONDITIONS (INPUT STRUCTURE)
# ============================================================

@dataclass
class MarketConditions:
    """
    Represents the current market state for a pricing decision.

    This is a Python dataclass — a clean way to group related
    data together. Think of it as a form that collects all the
    information our pricing engine needs.

    A 'dataclass' automatically generates __init__, __repr__,
    and other boilerplate methods based on the field definitions.

    Parameters (all inputs to the pricing engine):
    ---------------------------------------------------
    competitor_price : float
        What price is the main competitor currently charging?

    day_of_week : int
        Current day (0=Monday ... 6=Sunday)

    hour_of_day : int
        Current hour (0=midnight ... 23=11PM)

    month : int
        Current month (1=January ... 12=December)

    inventory_level : int
        How many units are currently in stock (1-100)

    is_promotion : int
        Is a promotional campaign currently active? (0 or 1)

    customer_rating : float
        Average product rating (1.0 to 5.0)

    product_category : str
        One of: 'Electronics', 'Clothing', 'Food', 'Books', 'Toys'

    cost_per_unit : float
        Our cost to produce/acquire one unit (for profit calculation)
        Default: $20

    current_price : float (optional)
        What we're currently charging (for change-limit constraint)
    """
    competitor_price  : float
    day_of_week       : int
    hour_of_day       : int
    month             : int
    inventory_level   : int
    is_promotion      : int
    customer_rating   : float
    product_category  : str
    cost_per_unit     : float = 20.0
    current_price     : Optional[float] = None


# ============================================================
# SECTION 2: LOAD TRAINED ARTIFACTS
# ============================================================

def load_model_artifacts():
    """
    Load the trained XGBoost model, fitted StandardScaler,
    and feature names list from disk.

    We load XGBoost (not Linear Regression or Random Forest)
    because it had the highest R² and lowest MAPE on test data.

    Returns:
        model        : Trained XGBoost model
        scaler       : Fitted StandardScaler
        feature_cols : List of feature column names (in correct order!)
    """
    print("📦 Loading trained model artifacts...")

    # --- Load XGBoost Model ---
    model_path = os.path.join(config.MODELS_DIR, 'xgboost.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please run: python src/train.py first!"
        )
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded    → {model_path}")

    # --- Load StandardScaler ---
    scaler_path = os.path.join(config.MODELS_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Scaler loaded   → {scaler_path}")

    # --- Load Feature Names ---
    # CRITICAL: Features must be in the EXACT SAME ORDER as during training
    # If order changes, the model gets wrong input → garbage predictions
    feature_path = os.path.join(
        config.PROCESSED_DATA_DIR, 'feature_names.txt'
    )
    with open(feature_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    print(f"   ✅ Features loaded → {len(feature_cols)} features")

    return model, scaler, feature_cols


# ============================================================
# SECTION 3: BUILD FEATURE VECTOR FROM MARKET CONDITIONS
# ============================================================

def build_feature_vector(price, conditions, feature_cols):
    """
    Convert a MarketConditions object + a candidate price
    into a single feature vector (one row) ready for prediction.

    This MIRRORS the exact same feature engineering from Step 4.
    If Step 4 created a feature, we must recreate it here too.
    Consistency between training and inference is critical.

    Parameters:
        price        : float — candidate price to evaluate
        conditions   : MarketConditions — current market state
        feature_cols : list  — ordered list of feature names

    Returns:
        DataFrame with exactly 1 row and len(feature_cols) columns
    """

    # ── Derived time features ──
    is_weekend   = 1 if conditions.day_of_week >= 5 else 0
    is_peak_hour = 1 if (
        (9 <= conditions.hour_of_day <= 12) or
        (17 <= conditions.hour_of_day <= 21)
    ) else 0

    # ── Season factor (same sine formula as data_generator.py) ──
    season_factor = (
        np.sin(2 * np.pi * conditions.month / 12) +
        np.sin(4 * np.pi * conditions.month / 12)
    )

    # ── Derived price features ──
    price_ratio     = price / (conditions.competitor_price + 1e-9)
    price_diff      = price - conditions.competitor_price
    price_squared   = (price ** 2) / 1000

    # ── Inventory scarcity ──
    inventory_scarcity = 1 - (conditions.inventory_level / 100)

    # ── Interaction features ──
    price_x_promotion = price * conditions.is_promotion
    price_x_peak      = price * is_peak_hour
    price_x_weekend   = price * is_weekend

    # ── Revenue potential ──
    revenue_potential = price * (conditions.customer_rating / 5.0)

    # ── Time period bucket ──
    h = conditions.hour_of_day
    if   0  <= h <= 5:  time_period = 0   # Night
    elif 6  <= h <= 11: time_period = 1   # Morning
    elif 12 <= h <= 16: time_period = 2   # Afternoon
    else:               time_period = 3   # Evening

    # ── Season bucket ──
    m = conditions.month
    if   m in [12, 1, 2]: season = 0   # Winter
    elif m in [3, 4, 5]:  season = 1   # Spring
    elif m in [6, 7, 8]:  season = 2   # Summer
    else:                  season = 3   # Fall

    # ── Rating tier ──
    r = conditions.customer_rating
    if   r >= 4.0: rating_tier = 2
    elif r >= 2.5: rating_tier = 1
    else:          rating_tier = 0

    # ── One-hot encode product category ──
    # The original encoding dropped 'Books' (first alphabetically)
    # So we only create columns for the remaining 4 categories
    cat = conditions.product_category
    cat_clothing    = 1 if cat == 'Clothing'    else 0
    cat_electronics = 1 if cat == 'Electronics' else 0
    cat_food        = 1 if cat == 'Food'        else 0
    cat_toys        = 1 if cat == 'Toys'        else 0
    # If cat == 'Books': all four are 0 (that's the dropped reference)

    # ── Assemble raw feature dictionary ──
    raw_features = {
        'price'              : price,
        'competitor_price'   : conditions.competitor_price,
        'day_of_week'        : conditions.day_of_week,
        'is_weekend'         : is_weekend,
        'hour_of_day'        : conditions.hour_of_day,
        'is_peak_hour'       : is_peak_hour,
        'month'              : conditions.month,
        'season_factor'      : season_factor,
        'inventory_level'    : conditions.inventory_level,
        'is_promotion'       : conditions.is_promotion,
        'customer_rating'    : conditions.customer_rating,
        'price_ratio'        : price_ratio,
        'price_diff'         : price_diff,
        'price_squared'      : price_squared,
        'inventory_scarcity' : inventory_scarcity,
        'price_x_promotion'  : price_x_promotion,
        'price_x_peak'       : price_x_peak,
        'price_x_weekend'    : price_x_weekend,
        'revenue_potential'  : revenue_potential,
        'time_period'        : time_period,
        'season'             : season,
        'rating_tier'        : rating_tier,
        'cat_Clothing'       : cat_clothing,
        'cat_Electronics'    : cat_electronics,
        'cat_Food'           : cat_food,
        'cat_Toys'           : cat_toys,
    }

    # ── Build DataFrame with columns in exact training order ──
    # This is CRITICAL — if column order is wrong, model predictions
    # will be completely wrong (features get mapped to wrong weights)
    row = pd.DataFrame([raw_features])

    # Keep only the columns that were used during training
    # (handles case where feature list differs slightly)
    available_cols = [c for c in feature_cols if c in row.columns]
    row = row[available_cols]

    return row


# ============================================================
# SECTION 4: PREDICT DEMAND AT A GIVEN PRICE
# ============================================================

def predict_demand(price, conditions, model, scaler, feature_cols):
    """
    Predict demand for a single price point given market conditions.

    Steps:
    1. Build feature vector (raw features)
    2. Apply StandardScaler (same scaling as training)
    3. Run through XGBoost model
    4. Clip to valid range [0, MAX_DEMAND]

    Returns:
        float — predicted demand (in units)
    """
    # Step 1: Build raw feature row
    row = build_feature_vector(price, conditions, feature_cols)

    # Step 2: Scale using the SAME scaler fitted during training
    # We must identify which columns are scaled (same logic as Step 4)
    binary_cols = [
        'is_weekend', 'is_peak_hour', 'is_promotion',
        'cat_Clothing', 'cat_Electronics', 'cat_Food', 'cat_Toys'
    ]
    cols_to_scale = [
        col for col in row.columns
        if col not in binary_cols
    ]

    # Apply transform (NOT fit_transform — never refit on new data!)
    row_scaled = row.copy()
    row_scaled[cols_to_scale] = scaler.transform(row[cols_to_scale])

    # Step 3: Predict demand
    demand_pred = model.predict(row_scaled)[0]

    # Step 4: Clip to valid range
    demand_pred = float(np.clip(demand_pred, 0, 500))

    return demand_pred


# ============================================================
# SECTION 5: PRICE SWEEP — THE OPTIMIZATION ENGINE
# ============================================================

def sweep_prices(conditions, model, scaler, feature_cols,
                 price_min=None, price_max=None,
                 n_points=200, cost_per_unit=None):
    """
    Sweep through all candidate prices and compute key metrics
    at each price point.

    This is the core optimization function:
    For each candidate price in [price_min, price_max]:
        1. Predict demand using ML model
        2. Compute revenue = price × demand
        3. Compute profit  = (price - cost) × demand
        4. Compute margin  = (price - cost) / price × 100%

    Parameters:
        conditions    : MarketConditions object
        model         : Trained XGBoost model
        scaler        : Fitted StandardScaler
        feature_cols  : Feature column names
        price_min     : Minimum price to consider (default: config.MIN_PRICE)
        price_max     : Maximum price to consider (default: config.MAX_PRICE)
        n_points      : Number of price points to evaluate (resolution)
        cost_per_unit : Cost to produce one unit

    Returns:
        DataFrame with columns:
        [price, demand, revenue, profit, margin_pct]
    """

    # Use config defaults if not specified
    price_min     = price_min or config.MIN_PRICE
    price_max     = price_max or config.MAX_PRICE
    cost_per_unit = cost_per_unit or conditions.cost_per_unit

    # Generate evenly spaced candidate prices
    # np.linspace(10, 200, 200) → [10.0, 10.95, 11.90, ..., 200.0]
    candidate_prices = np.linspace(price_min, price_max, n_points)

    results = []

    for price in candidate_prices:
        # Predict demand at this price
        demand = predict_demand(
            price, conditions, model, scaler, feature_cols
        )

        # Calculate business metrics
        revenue = price * demand
        profit  = (price - cost_per_unit) * demand

        # Margin = profit as a % of revenue
        # Guard against division by zero
        margin_pct = ((price - cost_per_unit) / price * 100
                      if price > 0 else 0)

        results.append({
            'price'      : round(price, 2),
            'demand'     : round(demand, 1),
            'revenue'    : round(revenue, 2),
            'profit'     : round(profit, 2),
            'margin_pct' : round(margin_pct, 1),
        })

    return pd.DataFrame(results)


# ============================================================
# SECTION 6: APPLY BUSINESS CONSTRAINTS
# ============================================================

def apply_business_constraints(sweep_df, conditions,
                                max_competitor_premium=0.25,
                                max_price_change_pct=0.30):
    """
    Filter the sweep results to only keep prices that satisfy
    all business constraints.

    Business constraints are real-world rules that prevent
    the engine from recommending technically-optimal but
    practically-harmful prices.

    Parameters:
        sweep_df              : DataFrame from sweep_prices()
        conditions            : MarketConditions object
        max_competitor_premium: Max % above competitor price allowed
                                (default 25% — don't charge >25% more)
        max_price_change_pct  : Max % change from current price
                                (default 30% — don't change >30% at once)

    Returns:
        Filtered DataFrame satisfying all constraints
    """
    df = sweep_df.copy()
    original_count = len(df)

    # ── Constraint 1: Minimum price (never sell below cost floor) ──
    # We want at least some margin, so min_price = cost * 1.1 (10% margin)
    min_viable_price = conditions.cost_per_unit * 1.10
    df = df[df['price'] >= min_viable_price]

    # ── Constraint 2: Maximum competitor premium ──
    # Don't charge more than (1 + max_competitor_premium) × competitor_price
    # Example: competitor=$100, max_premium=0.25 → max_price=$125
    max_allowed = conditions.competitor_price * (1 + max_competitor_premium)
    df = df[df['price'] <= max_allowed]

    # ── Constraint 3: Price change limit ──
    # If we have a current price, don't change it by more than 30%
    # Sudden large price changes confuse/anger customers
    if conditions.current_price is not None:
        current = conditions.current_price
        lower   = current * (1 - max_price_change_pct)
        upper   = current * (1 + max_price_change_pct)
        df      = df[(df['price'] >= lower) & (df['price'] <= upper)]

    # ── Constraint 4: Positive profit only ──
    # Never recommend a price that results in a loss
    df = df[df['profit'] > 0]

    remaining_count = len(df)
    filtered_count  = original_count - remaining_count

    if filtered_count > 0:
        print(f"   ⚠️  Business constraints filtered out "
              f"{filtered_count} price points")
        print(f"   ✅ {remaining_count} valid price points remain")

    return df


# ============================================================
# SECTION 7: FIND OPTIMAL PRICE (THREE STRATEGIES)
# ============================================================

def find_optimal_price(sweep_df, strategy='revenue'):
    """
    Find the single best price from the sweep results
    based on the chosen optimization strategy.

    Parameters:
        sweep_df : DataFrame from sweep_prices() (after constraints)
        strategy : One of:
                   'revenue'     → Maximize price × demand
                   'profit'      → Maximize (price-cost) × demand
                   'competitive' → Minimize distance from competitor

    Returns:
        Dictionary with optimal price and all associated metrics
    """
    if sweep_df.empty:
        raise ValueError(
            "No valid price points after applying constraints!\n"
            "Try relaxing: max_competitor_premium or max_price_change_pct"
        )

    if strategy == 'revenue':
        # Find row where revenue is maximum
        optimal_row = sweep_df.loc[sweep_df['revenue'].idxmax()]
        objective   = 'revenue'

    elif strategy == 'profit':
        # Find row where profit is maximum
        optimal_row = sweep_df.loc[sweep_df['profit'].idxmax()]
        objective   = 'profit'

    elif strategy == 'competitive':
        # Find price closest to competitor price
        # (minimum absolute difference)
        sweep_df    = sweep_df.copy()
        # This needs competitor price — passed via closure
        # We find the price with best revenue that's <= competitor price
        below_comp  = sweep_df[sweep_df['price'] <=
                                sweep_df['price'].median()]
        if not below_comp.empty:
            optimal_row = below_comp.loc[below_comp['revenue'].idxmax()]
        else:
            optimal_row = sweep_df.loc[sweep_df['revenue'].idxmax()]
        objective   = 'revenue (competitive)'

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose: 'revenue', 'profit', 'competitive'")

    return {
        'optimal_price'  : float(optimal_row['price']),
        'predicted_demand': float(optimal_row['demand']),
        'expected_revenue': float(optimal_row['revenue']),
        'expected_profit' : float(optimal_row['profit']),
        'margin_pct'      : float(optimal_row['margin_pct']),
        'strategy'        : strategy,
        'objective'       : objective,
    }


# ============================================================
# SECTION 8: FULL PRICING RECOMMENDATION REPORT
# ============================================================

def generate_pricing_report(conditions, optimal, sweep_df):
    """
    Print a professional pricing recommendation report.
    This is what you'd show to a business stakeholder.
    """
    print("\n" + "="*60)
    print("   💰 DYNAMIC PRICING ENGINE — RECOMMENDATION REPORT")
    print("="*60)

    # ── Market Conditions Summary ──
    day_names = ['Monday','Tuesday','Wednesday',
                 'Thursday','Friday','Saturday','Sunday']
    cat_name  = conditions.product_category
    day_name  = day_names[conditions.day_of_week]

    print(f"""
  📋 MARKET CONDITIONS:
  ├─ Product Category  : {cat_name}
  ├─ Competitor Price  : ${conditions.competitor_price:.2f}
  ├─ Day / Hour        : {day_name}, {conditions.hour_of_day:02d}:00
  ├─ Month             : {conditions.month} (Month #{conditions.month})
  ├─ Inventory Level   : {conditions.inventory_level} units
  ├─ Active Promotion  : {'Yes ✅' if conditions.is_promotion else 'No ❌'}
  ├─ Customer Rating   : {conditions.customer_rating:.1f} / 5.0
  └─ Cost Per Unit     : ${conditions.cost_per_unit:.2f}
""")

    # ── Optimal Price Recommendation ──
    price   = optimal['optimal_price']
    demand  = optimal['predicted_demand']
    revenue = optimal['expected_revenue']
    profit  = optimal['expected_profit']
    margin  = optimal['margin_pct']

    vs_competitor = ((price - conditions.competitor_price) /
                     conditions.competitor_price * 100)
    vs_sign = "+" if vs_competitor >= 0 else ""

    print(f"""  🎯 OPTIMAL PRICE RECOMMENDATION:
  ╔══════════════════════════════════════════════╗
  ║  Recommended Price : ${price:>8.2f}             ║
  ║  Strategy Used     : {optimal['strategy']:<28} ║
  ╚══════════════════════════════════════════════╝

  📊 EXPECTED OUTCOMES AT ${price:.2f}:
  ├─ Predicted Demand  : {demand:.0f} units
  ├─ Expected Revenue  : ${revenue:,.2f}
  ├─ Expected Profit   : ${profit:,.2f}
  ├─ Profit Margin     : {margin:.1f}%
  └─ vs Competitor     : {vs_sign}{vs_competitor:.1f}%
""")

    # ── Scenario Comparison ──
    print("  📈 SCENARIO COMPARISON (Revenue vs Price Strategy):")
    print(f"  {'Strategy':<22} {'Price':>8} {'Demand':>8} "
          f"{'Revenue':>12} {'Profit':>12}")
    print(f"  {'-'*64}")

    # Revenue-maximizing price
    rev_row  = sweep_df.loc[sweep_df['revenue'].idxmax()]
    # Profit-maximizing price
    prof_row = sweep_df.loc[sweep_df['profit'].idxmax()]
    # Lowest valid price
    low_row  = sweep_df.iloc[0]
    # Competitor-matched price (closest to competitor)
    comp_row = sweep_df.iloc[(
        sweep_df['price'] - conditions.competitor_price
    ).abs().argsort()[:1]]

    scenarios = [
        ("Max Revenue",   rev_row),
        ("Max Profit",    prof_row),
        ("Match Competitor", comp_row.iloc[0]),
        ("Lowest Price",  low_row),
    ]

    for scenario_name, row in scenarios:
        marker = " ← *" if scenario_name.lower().replace(
            " ", "_"
        ) in optimal['strategy'] else ""
        print(f"  {scenario_name:<22} "
              f"${float(row['price']):>7.2f} "
              f"{float(row['demand']):>8.0f} "
              f"${float(row['revenue']):>11,.2f} "
              f"${float(row['profit']):>11,.2f}"
              f"{marker}")

    # ── Business Insight ──
    print(f"""
  💡 BUSINESS INSIGHTS:
  ├─ Profit-maximizing price is ${float(prof_row['price']):.2f}
  │   vs Revenue-maximizing at ${float(rev_row['price']):.2f}
  │   (Profit focus costs ${float(rev_row['revenue'] - prof_row['revenue']):,.0f} in revenue
  │    but saves on unnecessary demand you can't fulfill profitably)
  ├─ At recommended price, we are {vs_sign}{vs_competitor:.1f}% vs competitor
  │   {'→ We have a price advantage! Consider slight increase.' if vs_competitor < -10 else ''}
  │   {'→ We are competitive.' if -10 <= vs_competitor <= 10 else ''}
  │   {'→ Consider matching competitor if demand drops.' if vs_competitor > 10 else ''}
  └─ Inventory at {conditions.inventory_level} units:
     {'→ Low stock! Consider premium pricing.' if conditions.inventory_level < 20 else ''}
     {'→ Healthy stock level.' if 20 <= conditions.inventory_level <= 70 else ''}
     {'→ High stock! Consider promotional pricing.' if conditions.inventory_level > 70 else ''}
""")

    print("="*60)


# ============================================================
# SECTION 9: VISUALIZATION — PRICE OPTIMIZATION CURVES
# ============================================================

def save_figure(filename):
    """Helper to save figure to reports/figures/."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    print(f"   💾 Saved → {path}")
    plt.close()


def plot_optimization_curves(sweep_df, optimal, conditions):
    """
    Plot 4 charts showing the full price optimization analysis:
    1. Revenue curve with optimal point marked
    2. Profit curve with optimal point marked
    3. Demand curve (how demand falls with price)
    4. Margin % curve (how margin changes with price)
    """
    print("\n📊 Plotting optimization curves...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'💰 Price Optimization Analysis\n'
        f'Category: {conditions.product_category} | '
        f'Competitor: ${conditions.competitor_price:.0f} | '
        f'Inventory: {conditions.inventory_level} units',
        fontsize=14, fontweight='bold'
    )

    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.4, wspace=0.35)

    opt_price   = optimal['optimal_price']
    opt_demand  = optimal['predicted_demand']
    opt_revenue = optimal['expected_revenue']
    opt_profit  = optimal['expected_profit']

    # ── Plot 1: Revenue Curve ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sweep_df['price'], sweep_df['revenue'],
             color=COLORS['success'], linewidth=2.5, label='Revenue')
    ax1.fill_between(sweep_df['price'], sweep_df['revenue'],
                     alpha=0.15, color=COLORS['success'])

    # Mark optimal point
    ax1.axvline(opt_price, color=COLORS['gold'],
                linewidth=2.5, linestyle='--',
                label=f'Optimal: ${opt_price:.0f}')
    ax1.scatter([opt_price], [opt_revenue],
                color=COLORS['gold'], s=150,
                zorder=5, marker='*')

    # Mark competitor price
    ax1.axvline(conditions.competitor_price,
                color=COLORS['secondary'], linewidth=1.5,
                linestyle=':', label=f"Competitor: ${conditions.competitor_price:.0f}")

    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Revenue ($)')
    ax1.set_title('Revenue Curve\n(Find the Peak)')
    ax1.legend(fontsize=9)

    # Annotate peak
    ax1.annotate(
        f'  Max Revenue\n  ${opt_revenue:,.0f}',
        xy=(opt_price, opt_revenue),
        xytext=(opt_price + 10, opt_revenue * 0.85),
        fontsize=9, color=COLORS['dark'],
        arrowprops=dict(arrowstyle='->', color=COLORS['dark'])
    )

    # ── Plot 2: Profit Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    # Color profit green where positive, red where negative
    ax2.fill_between(
        sweep_df['price'], sweep_df['profit'],
        where=sweep_df['profit'] >= 0,
        alpha=0.25, color=COLORS['success'],
        label='Profitable Zone'
    )
    ax2.fill_between(
        sweep_df['price'], sweep_df['profit'],
        where=sweep_df['profit'] < 0,
        alpha=0.25, color=COLORS['secondary'],
        label='Loss Zone'
    )
    ax2.plot(sweep_df['price'], sweep_df['profit'],
             color=COLORS['purple'], linewidth=2.5)
    ax2.axhline(0, color=COLORS['dark'],
                linewidth=1.5, linestyle='-', alpha=0.5)

    # Mark profit-optimal price
    prof_idx = sweep_df['profit'].idxmax()
    prof_price = sweep_df.loc[prof_idx, 'price']
    ax2.axvline(prof_price, color=COLORS['gold'],
                linewidth=2, linestyle='--',
                label=f'Max Profit: ${prof_price:.0f}')
    ax2.scatter([prof_price], [opt_profit],
                color=COLORS['gold'], s=150,
                zorder=5, marker='*')

    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Profit ($)')
    ax2.set_title('Profit Curve\n(Green = Profitable, Red = Loss)')
    ax2.legend(fontsize=9)

    # ── Plot 3: Demand Curve ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sweep_df['price'], sweep_df['demand'],
             color=COLORS['primary'], linewidth=2.5,
             label='Predicted Demand')
    ax3.fill_between(sweep_df['price'], sweep_df['demand'],
                     alpha=0.15, color=COLORS['primary'])

    # Mark demand at optimal price
    ax3.axvline(opt_price, color=COLORS['gold'],
                linewidth=2, linestyle='--',
                label=f'At Optimal: {opt_demand:.0f} units')
    ax3.scatter([opt_price], [opt_demand],
                color=COLORS['gold'], s=150,
                zorder=5, marker='*')

    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Predicted Demand (units)')
    ax3.set_title('Demand Curve\n(Law of Demand: Higher Price → Lower Demand)')
    ax3.legend(fontsize=9)

    # ── Plot 4: Margin % Curve ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(sweep_df['price'], sweep_df['margin_pct'],
             color=COLORS['accent'], linewidth=2.5,
             label='Profit Margin %')

    # Shade "good margin" zone (above 30%)
    ax4.axhline(30, color=COLORS['success'], linewidth=1.5,
                linestyle='--', alpha=0.7, label='30% Margin Target')
    ax4.fill_between(
        sweep_df['price'], sweep_df['margin_pct'],
        30, where=sweep_df['margin_pct'] >= 30,
        alpha=0.15, color=COLORS['success'],
        label='Above Target Zone'
    )

    # Mark margin at optimal price
    opt_margin = sweep_df.loc[
        (sweep_df['price'] - opt_price).abs().idxmin(),
        'margin_pct'
    ]
    ax4.axvline(opt_price, color=COLORS['gold'],
                linewidth=2, linestyle='--',
                label=f'Optimal Margin: {opt_margin:.1f}%')

    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Profit Margin (%)')
    ax4.set_title('Margin % Curve\n(Higher Price → Better Margin)')
    ax4.legend(fontsize=9)
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    save_figure('13_price_optimization.png')


def plot_sensitivity_analysis(conditions, model, scaler, feature_cols):
    """
    Sensitivity Analysis: How does the optimal price change
    when we vary ONE condition at a time?

    This is crucial for business decisions:
    'If our competitor drops price by $20, what should we do?'
    'How much does being a peak hour affect our optimal price?'

    We vary:
    1. Competitor price ($50 to $200)
    2. Inventory level (5 to 95 units)
    3. Customer rating (1.0 to 5.0)
    4. Whether promotion is active (0 vs 1)
    """
    print("📊 Plotting sensitivity analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        '🔬 Sensitivity Analysis — How Optimal Price Responds\n'
        'to Changes in Market Conditions',
        fontsize=14, fontweight='bold'
    )

    # ── Helper: find optimal price for modified conditions ──
    def get_optimal_for_sweep(modified_conditions):
        """Run full sweep and return optimal price."""
        sweep = sweep_prices(
            modified_conditions, model, scaler, feature_cols,
            n_points=100   # Fewer points for speed in sensitivity
        )
        constrained = apply_business_constraints(
            sweep, modified_conditions
        )
        if constrained.empty:
            return None, None
        opt = find_optimal_price(constrained, strategy='revenue')
        return opt['optimal_price'], opt['expected_revenue']

    import copy   # To safely copy the conditions object

    # ── Sensitivity 1: Competitor Price ──
    ax = axes[0, 0]
    comp_prices    = np.linspace(50, 180, 25)
    opt_prices_1   = []
    opt_revenues_1 = []

    for cp in comp_prices:
        cond_mod = copy.copy(conditions)
        cond_mod.competitor_price = cp
        p, r = get_optimal_for_sweep(cond_mod)
        opt_prices_1.append(p if p else np.nan)
        opt_revenues_1.append(r if r else np.nan)

    ax.plot(comp_prices, opt_prices_1,
            color=COLORS['primary'], linewidth=2.5,
            marker='o', markersize=4)
    # Add 1:1 line (if we exactly matched competitor)
    ax.plot(comp_prices, comp_prices,
            color=COLORS['secondary'], linewidth=1.5,
            linestyle='--', alpha=0.7, label='Match Competitor')
    ax.set_xlabel('Competitor Price ($)')
    ax.set_ylabel('Our Optimal Price ($)')
    ax.set_title('Sensitivity to Competitor Price\n'
                 '(Our price rises as competitor rises)')
    ax.legend(fontsize=9)

    # ── Sensitivity 2: Inventory Level ──
    ax2 = axes[0, 1]
    inventories  = np.linspace(5, 95, 25).astype(int)
    opt_prices_2 = []

    for inv in inventories:
        cond_mod = copy.copy(conditions)
        cond_mod.inventory_level = int(inv)
        p, _ = get_optimal_for_sweep(cond_mod)
        opt_prices_2.append(p if p else np.nan)

    ax2.plot(inventories, opt_prices_2,
             color=COLORS['success'], linewidth=2.5,
             marker='s', markersize=4)
    ax2.set_xlabel('Inventory Level (units)')
    ax2.set_ylabel('Our Optimal Price ($)')
    ax2.set_title('Sensitivity to Inventory Level\n'
                  '(Low stock → Higher price)')
    # Add shaded regions
    ax2.axvspan(0, 20, alpha=0.1, color=COLORS['secondary'],
                label='Low Stock Zone')
    ax2.axvspan(70, 100, alpha=0.1, color=COLORS['success'],
                label='High Stock Zone')
    ax2.legend(fontsize=9)

    # ── Sensitivity 3: Customer Rating ──
    ax3 = axes[1, 0]
    ratings      = np.linspace(1.0, 5.0, 25)
    opt_prices_3 = []

    for rat in ratings:
        cond_mod = copy.copy(conditions)
        cond_mod.customer_rating = float(rat)
        p, _ = get_optimal_for_sweep(cond_mod)
        opt_prices_3.append(p if p else np.nan)

    ax3.plot(ratings, opt_prices_3,
             color=COLORS['purple'], linewidth=2.5,
             marker='^', markersize=4)
    ax3.set_xlabel('Customer Rating (1.0 to 5.0)')
    ax3.set_ylabel('Our Optimal Price ($)')
    ax3.set_title('Sensitivity to Customer Rating\n'
                  '(Higher rating → Can charge more)')
    # Mark rating tiers
    ax3.axvline(2.5, color='gray', linestyle='--',
                alpha=0.5, label='Low/Med threshold')
    ax3.axvline(4.0, color='gray', linestyle='-.',
                alpha=0.5, label='Med/High threshold')
    ax3.legend(fontsize=9)

    # ── Sensitivity 4: Promotion vs No Promotion ──
    ax4 = axes[1, 1]
    # Compare full revenue curves: with and without promotion
    cond_promo    = copy.copy(conditions)
    cond_promo.is_promotion = 1
    cond_no_promo = copy.copy(conditions)
    cond_no_promo.is_promotion = 0

    sweep_promo    = sweep_prices(cond_promo,    model, scaler,
                                  feature_cols, n_points=100)
    sweep_no_promo = sweep_prices(cond_no_promo, model, scaler,
                                  feature_cols, n_points=100)

    ax4.plot(sweep_promo['price'], sweep_promo['revenue'],
             color=COLORS['secondary'], linewidth=2.5,
             label='With Promotion')
    ax4.plot(sweep_no_promo['price'], sweep_no_promo['revenue'],
             color=COLORS['primary'], linewidth=2.5,
             linestyle='--', label='Without Promotion')
    ax4.fill_between(
        sweep_promo['price'],
        sweep_promo['revenue'],
        sweep_no_promo['revenue'],
        alpha=0.15, color=COLORS['accent'],
        label='Promotion Lift'
    )
    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Revenue ($)')
    ax4.set_title('Promotion vs No Promotion\n'
                  '(Revenue curves comparison)')
    ax4.legend(fontsize=9)

    plt.tight_layout()
    save_figure('14_sensitivity_analysis.png')


# ============================================================
# SECTION 10: BATCH PRICING — MULTIPLE SCENARIOS AT ONCE
# ============================================================

def batch_pricing(scenarios, model, scaler, feature_cols):
    """
    Run the pricing engine for multiple scenarios simultaneously.
    This is how a real system would work — pricing hundreds of
    products or time slots in one batch run.

    Parameters:
        scenarios : List of (name, MarketConditions) tuples

    Returns:
        DataFrame with one row per scenario showing recommendations
    """
    print("\n🔄 Running Batch Pricing for Multiple Scenarios...")
    print(f"   Processing {len(scenarios)} scenarios...\n")

    results = []

    for scenario_name, conditions in scenarios:
        # Run full optimization
        sweep        = sweep_prices(conditions, model, scaler,
                                    feature_cols, n_points=150)
        constrained  = apply_business_constraints(sweep, conditions)

        if constrained.empty:
            print(f"   ⚠️  {scenario_name}: No valid prices after constraints")
            continue

        optimal = find_optimal_price(constrained, strategy='revenue')

        results.append({
            'Scenario'         : scenario_name,
            'Category'         : conditions.product_category,
            'Competitor ($)'   : conditions.competitor_price,
            'Optimal Price ($)': round(optimal['optimal_price'], 2),
            'Demand (units)'   : round(optimal['predicted_demand'], 0),
            'Revenue ($)'      : round(optimal['expected_revenue'], 2),
            'Profit ($)'       : round(optimal['expected_profit'], 2),
            'Margin (%)'       : round(optimal['margin_pct'], 1),
        })
        print(f"   ✅ {scenario_name:<30} → "
              f"Optimal: ${optimal['optimal_price']:.2f} | "
              f"Revenue: ${optimal['expected_revenue']:,.0f}")

    batch_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("   BATCH PRICING RESULTS SUMMARY")
    print(f"{'='*70}")
    print(batch_df.to_string(index=False))
    print(f"{'='*70}")

    return batch_df


# ============================================================
# MAIN — RUN THE FULL PRICING LOGIC PIPELINE
# ============================================================

def main():
    print("="*60)
    print("  💰 DYNAMIC PRICING ENGINE — PRICING LOGIC")
    print("="*60)

    # ── Step 1: Load trained model ──
    model, scaler, feature_cols = load_model_artifacts()

    # ── Step 2: Define a sample market scenario ──
    # (This simulates a real-time pricing request)
    print("\n📋 Defining market conditions for pricing decision...")

    conditions = MarketConditions(
        competitor_price = 120.0,   # Competitor charges $120
        day_of_week      = 5,       # Saturday
        hour_of_day      = 19,      # 7 PM (peak evening hour)
        month            = 12,      # December (holiday season)
        inventory_level  = 15,      # Low stock (only 15 units left!)
        is_promotion     = 1,       # Active promotion
        customer_rating  = 4.2,     # Highly rated product
        product_category = 'Electronics',
        cost_per_unit    = 40.0,    # Each unit costs us $40
        current_price    = 110.0,   # We're currently charging $110
    )

    # ── Step 3: Sweep all candidate prices ──
    print("\n🔍 Running price sweep...")
    sweep_df = sweep_prices(
        conditions, model, scaler, feature_cols, n_points=200
    )
    print(f"   ✅ Evaluated {len(sweep_df)} price points")

    # ── Step 4: Apply business constraints ──
    print("\n🛡️  Applying business constraints...")
    constrained_df = apply_business_constraints(
        sweep_df, conditions,
        max_competitor_premium = 0.25,
        max_price_change_pct   = 0.30
    )

    # ── Step 5: Find optimal price (Revenue Strategy) ──
    print("\n🎯 Finding optimal price...")
    optimal = find_optimal_price(
        constrained_df, strategy='revenue'
    )

    # ── Step 6: Generate recommendation report ──
    generate_pricing_report(conditions, optimal, constrained_df)

    # ── Step 7: Plot optimization curves ──
    print("📈 Generating optimization visualizations...")
    plot_optimization_curves(sweep_df, optimal, conditions)
    plot_sensitivity_analysis(conditions, model, scaler, feature_cols)

    # ── Step 8: Batch pricing for multiple scenarios ──
    scenarios = [
        ("Electronics - Weekend Peak",   MarketConditions(
            competitor_price=120, day_of_week=6, hour_of_day=20,
            month=12, inventory_level=10, is_promotion=1,
            customer_rating=4.5, product_category='Electronics',
            cost_per_unit=40.0)),

        ("Food - Weekday Off-Peak",      MarketConditions(
            competitor_price=25, day_of_week=2, hour_of_day=14,
            month=3, inventory_level=80, is_promotion=0,
            customer_rating=3.8, product_category='Food',
            cost_per_unit=8.0)),

        ("Books - Weekend No Promo",     MarketConditions(
            competitor_price=35, day_of_week=5, hour_of_day=11,
            month=9, inventory_level=50, is_promotion=0,
            customer_rating=4.0, product_category='Books',
            cost_per_unit=12.0)),

        ("Clothing - Summer Sale",       MarketConditions(
            competitor_price=80, day_of_week=6, hour_of_day=15,
            month=7, inventory_level=90, is_promotion=1,
            customer_rating=3.5, product_category='Clothing',
            cost_per_unit=25.0)),

        ("Toys - Holiday Season Low Stock", MarketConditions(
            competitor_price=55, day_of_week=6, hour_of_day=18,
            month=12, inventory_level=5, is_promotion=0,
            customer_rating=4.8, product_category='Toys',
            cost_per_unit=18.0)),
    ]

    batch_df = batch_pricing(scenarios, model, scaler, feature_cols)

    print("\n✅ Pricing Logic Pipeline Complete!")
    print(f"   📊 Charts saved in: {config.FIGURES_DIR}/")
    print("\n   ➡️  Ready for Step 7: Streamlit Web App!\n")

    return optimal, sweep_df, batch_df


if __name__ == "__main__":
    main()