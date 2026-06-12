# notebooks/eda.py
# ============================================================
# DYNAMIC PRICING ENGINE — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# PURPOSE : Visualize and understand the simulated dataset
#           before any machine learning is applied.
# OUTPUT  : 8 graphs saved to reports/figures/
# RUN     : python notebooks/eda.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys

# Add project root so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# SECTION 0: GLOBAL PLOT SETTINGS
# ============================================================
# Set a professional visual style for all plots
sns.set_theme(style="darkgrid")          # Dark grid background
plt.rcParams['figure.dpi'] = 120         # High resolution figures
plt.rcParams['font.size'] = 11           # Readable font size
plt.rcParams['axes.titlesize'] = 13      # Bigger title font
plt.rcParams['axes.titleweight'] = 'bold'# Bold titles

# Color palette — consistent colors throughout all graphs
COLORS = {
    'primary'    : '#2E86AB',   # Blue
    'secondary'  : '#E84855',   # Red
    'accent'     : '#F9A03F',   # Orange
    'success'    : '#3BB273',   # Green
    'purple'     : '#7B2D8B',   # Purple
    'dark'       : '#1C1C1E',   # Near black
}

PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
           COLORS['success'], COLORS['purple']]

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

def load_data():
    """Load the raw dataset from CSV."""
    print("📂 Loading dataset...")
    df = pd.read_csv(config.RAW_DATA_FILE)
    print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df

# ============================================================
# SECTION 2: BASIC DATA SUMMARY
# ============================================================

def print_data_summary(df):
    """Print a comprehensive summary of the dataset."""

    print("\n" + "="*60)
    print("        🔍 DATASET OVERVIEW")
    print("="*60)

    # --- Shape ---
    print(f"\n📐 Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Data Types ---
    print(f"\n📋 Column Info:")
    print(f"   {'Column':<22} {'Type':<12} {'Unique Values':<15} {'Missing'}")
    print(f"   {'-'*60}")
    for col in df.columns:
        dtype    = str(df[col].dtype)
        n_unique = df[col].nunique()
        missing  = df[col].isnull().sum()
        print(f"   {col:<22} {dtype:<12} {n_unique:<15} {missing}")

    # --- Numerical Summary ---
    print(f"\n📊 Statistical Summary (Numerical Columns):")
    numerical_cols = ['price', 'competitor_price', 'demand',
                      'revenue', 'customer_rating', 'inventory_level']
    print(df[numerical_cols].describe().round(2).to_string())

    # --- Key Business Insights ---
    print(f"\n💡 KEY BUSINESS INSIGHTS:")
    print(f"   Average Price          : ${df['price'].mean():.2f}")
    print(f"   Average Demand         : {df['demand'].mean():.1f} units")
    print(f"   Average Revenue        : ${df['revenue'].mean():,.2f}")
    print(f"   Highest Revenue Row    : ${df['revenue'].max():,.2f}")
    print(f"   Zero-Demand Events     : {(df['demand'] == 0).sum()} "
          f"({(df['demand']==0).mean()*100:.1f}%)")
    print(f"   Promotions Active      : {df['is_promotion'].sum():,} "
          f"({df['is_promotion'].mean()*100:.1f}%)")
    print("="*60)

# ============================================================
# SECTION 3: VISUALIZATION FUNCTIONS
# ============================================================

def save_figure(filename):
    """Helper: save figure to reports/figures/ folder."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    print(f"   💾 Saved → {filepath}")
    plt.close()   # Close figure to free memory


# ----------------------------------------------------------
# PLOT 1: DISTRIBUTION OF KEY NUMERICAL FEATURES
# ----------------------------------------------------------
def plot_distributions(df):
    """
    Plot histograms for key numerical columns.
    Helps us understand: What range of values does each feature take?
    """
    print("\n📊 Plot 1: Feature Distributions...")

    # Select the numerical columns we care about
    num_cols = ['price', 'competitor_price', 'demand',
                'revenue', 'customer_rating', 'inventory_level']

    # Create a grid of 2 rows × 3 columns of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('📊 Distribution of Key Features',
                 fontsize=16, fontweight='bold', y=1.01)

    # Flatten axes array so we can loop through it easily
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]

        # Plot histogram with KDE (smooth curve on top)
        ax.hist(df[col], bins=40,
                color=PALETTE[i % len(PALETTE)],
                alpha=0.75, edgecolor='white')

        # Add a vertical line at the mean
        mean_val = df[col].mean()
        ax.axvline(mean_val, color=COLORS['dark'],
                   linewidth=2, linestyle='--', label=f'Mean: {mean_val:.1f}')

        # Labels
        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_figure('01_feature_distributions.png')


# ----------------------------------------------------------
# PLOT 2: DEMAND vs PRICE (THE MOST IMPORTANT RELATIONSHIP)
# ----------------------------------------------------------
def plot_demand_vs_price(df):
    """
    Scatter plot of Demand vs Price.
    This validates our simulation: higher price SHOULD mean lower demand.
    This is the LAW OF DEMAND in action.
    """
    print("📊 Plot 2: Demand vs Price...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('📉 Demand vs Price — The Law of Demand',
                 fontsize=15, fontweight='bold')

    # --- Left Plot: Raw scatter ---
    ax = axes[0]
    scatter = ax.scatter(
        df['price'], df['demand'],
        c=df['is_promotion'],            # Color points by promotion status
        cmap='coolwarm',                 # Blue = no promo, Red = promo
        alpha=0.3,                       # Transparency (many overlapping points)
        s=8                              # Small dot size
    )
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Demand (Units)')
    ax.set_title('Raw Data: Each dot = 1 transaction')

    # Add colorbar legend
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Is Promotion (0=No, 1=Yes)')

    # --- Right Plot: Binned average (cleaner trend line) ---
    ax2 = axes[1]

    # Divide price into 20 equal-width bins and calculate mean demand per bin
    df['price_bin'] = pd.cut(df['price'], bins=20)
    binned = df.groupby('price_bin', observed=True)['demand'].mean().reset_index()
    binned['price_mid'] = binned['price_bin'].apply(
        lambda x: x.mid  # Get midpoint of each bin
    )

    # Plot the binned average as a smooth trend line
    ax2.plot(binned['price_mid'], binned['demand'],
             color=COLORS['primary'], linewidth=2.5,
             marker='o', markersize=6, label='Average Demand')

    # Fill area under the curve
    ax2.fill_between(binned['price_mid'], binned['demand'],
                     alpha=0.15, color=COLORS['primary'])

    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Average Demand (Units)')
    ax2.set_title('Binned Average: Clear Downward Trend')
    ax2.legend()

    # Drop temporary column
    df.drop(columns=['price_bin'], inplace=True)

    plt.tight_layout()
    save_figure('02_demand_vs_price.png')


# ----------------------------------------------------------
# PLOT 3: DEMAND BY PRODUCT CATEGORY
# ----------------------------------------------------------
def plot_demand_by_category(df):
    """
    Box plots showing demand distribution for each product category.
    Reveals which categories sell most and how variable their demand is.
    """
    print("📊 Plot 3: Demand by Category...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🏷️  Demand & Revenue by Product Category',
                 fontsize=15, fontweight='bold')

    # --- Left: Box plot of demand by category ---
    ax = axes[0]
    categories = df['product_category'].unique()
    data_by_cat = [df[df['product_category'] == cat]['demand'].values
                   for cat in categories]

    bp = ax.boxplot(data_by_cat, patch_artist=True,
                    labels=categories, notch=False)

    # Color each box differently
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Product Category')
    ax.set_ylabel('Demand (Units)')
    ax.set_title('Demand Distribution by Category')
    ax.tick_params(axis='x', rotation=15)

    # --- Right: Average revenue by category (bar chart) ---
    ax2 = axes[1]
    avg_revenue = df.groupby('product_category')['revenue'].mean().sort_values(
        ascending=False)

    bars = ax2.bar(avg_revenue.index, avg_revenue.values,
                   color=PALETTE, alpha=0.85, edgecolor='white', linewidth=1.2)

    # Add value labels on top of each bar
    for bar, val in zip(bars, avg_revenue.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f'${val:,.0f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Product Category')
    ax2.set_ylabel('Average Revenue ($)')
    ax2.set_title('Average Revenue by Category')
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    save_figure('03_demand_by_category.png')


# ----------------------------------------------------------
# PLOT 4: TIME-BASED DEMAND PATTERNS
# ----------------------------------------------------------
def plot_time_patterns(df):
    """
    Line charts showing how demand varies:
    - By hour of day (daily pattern)
    - By day of week (weekly pattern)
    - By month (seasonal pattern)
    """
    print("📊 Plot 4: Time-Based Demand Patterns...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('⏰ Time-Based Demand Patterns',
                 fontsize=15, fontweight='bold')

    # --- Hour of Day ---
    ax = axes[0]
    hourly = df.groupby('hour_of_day')['demand'].mean()
    ax.plot(hourly.index, hourly.values,
            color=COLORS['primary'], linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(hourly.index, hourly.values,
                    alpha=0.15, color=COLORS['primary'])

    # Highlight peak hours with shaded regions
    ax.axvspan(9, 12, alpha=0.12, color=COLORS['accent'],
               label='Morning Peak (9-12)')
    ax.axvspan(17, 21, alpha=0.12, color=COLORS['secondary'],
               label='Evening Peak (17-21)')
    ax.set_xlabel('Hour of Day (0=Midnight, 23=11PM)')
    ax.set_ylabel('Average Demand')
    ax.set_title('Demand by Hour of Day')
    ax.legend(fontsize=8)

    # --- Day of Week ---
    ax2 = axes[1]
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = df.groupby('day_of_week')['demand'].mean()
    bar_colors = [COLORS['secondary'] if i >= 5 else COLORS['primary']
                  for i in range(7)]   # Red for weekends

    ax2.bar(day_labels, daily.values, color=bar_colors,
            alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Demand')
    ax2.set_title('Demand by Day of Week\n(Red = Weekend)')

    # --- Month / Seasonal ---
    ax3 = axes[2]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = df.groupby('month')['demand'].mean()
    ax3.plot(month_labels, monthly.values,
             color=COLORS['success'], linewidth=2.5,
             marker='s', markersize=6, markerfacecolor=COLORS['accent'])
    ax3.fill_between(range(12), monthly.values,
                     alpha=0.15, color=COLORS['success'])
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Demand')
    ax3.set_title('Seasonal Demand Pattern')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_figure('04_time_patterns.png')


# ----------------------------------------------------------
# PLOT 5: CORRELATION HEATMAP
# ----------------------------------------------------------
def plot_correlation_heatmap(df):
    """
    Heatmap showing pairwise correlations between all numerical features.
    KEY INSIGHT: Which features are most correlated with DEMAND?
    Those will be our most important ML features.
    """
    print("📊 Plot 5: Correlation Heatmap...")

    # Select only numerical columns for correlation
    num_cols = ['price', 'competitor_price', 'demand', 'revenue',
                'day_of_week', 'is_weekend', 'hour_of_day', 'is_peak_hour',
                'month', 'season_factor', 'inventory_level',
                'is_promotion', 'customer_rating']

    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle('🔥 Feature Correlation Heatmap\n'
                 '(Focus on the "demand" row to find key predictors)',
                 fontsize=14, fontweight='bold')

    # Draw heatmap
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True   # Hide upper triangle (redundant)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,               # Show correlation values in each cell
        fmt='.2f',                # Round to 2 decimal places
        cmap='RdYlGn',            # Red=negative, Yellow=neutral, Green=positive
        center=0,                 # Center color at 0
        vmin=-1, vmax=1,          # Full correlation range
        square=True,              # Square cells
        linewidths=0.5,           # Thin lines between cells
        linecolor='white',
        ax=ax,
        annot_kws={'size': 9}
    )

    ax.set_title('')
    plt.tight_layout()
    save_figure('05_correlation_heatmap.png')


# ----------------------------------------------------------
# PLOT 6: PROMOTION EFFECT ANALYSIS
# ----------------------------------------------------------
def plot_promotion_effect(df):
    """
    Compare demand and revenue WITH vs WITHOUT promotions.
    Proves that our simulation correctly models promotion effect.
    """
    print("📊 Plot 6: Promotion Effect...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('🎯 Impact of Promotions on Demand & Revenue',
                 fontsize=15, fontweight='bold')

    promo_labels = ['No Promotion', 'Active Promotion']
    promo_colors = [COLORS['primary'], COLORS['secondary']]

    # --- Average Demand ---
    ax = axes[0]
    avg_demand = df.groupby('is_promotion')['demand'].mean()
    bars = ax.bar(promo_labels, avg_demand.values,
                  color=promo_colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, avg_demand.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Demand (Units)')
    ax.set_title('Average Demand')

    # --- Average Revenue ---
    ax2 = axes[1]
    avg_rev = df.groupby('is_promotion')['revenue'].mean()
    bars2 = ax2.bar(promo_labels, avg_rev.values,
                    color=promo_colors, alpha=0.85,
                    edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars2, avg_rev.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f'${val:,.0f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=12)
    ax2.set_ylabel('Average Revenue ($)')
    ax2.set_title('Average Revenue')

    # --- Demand Distribution Comparison ---
    ax3 = axes[2]
    for promo_val, label, color in zip([0, 1], promo_labels, promo_colors):
        subset = df[df['is_promotion'] == promo_val]['demand']
        ax3.hist(subset, bins=30, alpha=0.6, label=label,
                 color=color, edgecolor='white')
    ax3.set_xlabel('Demand (Units)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Demand Distribution Comparison')
    ax3.legend()

    plt.tight_layout()
    save_figure('06_promotion_effect.png')


# ----------------------------------------------------------
# PLOT 7: PRICE vs REVENUE (PROFIT SWEET SPOT)
# ----------------------------------------------------------
def plot_price_vs_revenue(df):
    """
    Find the OPTIMAL PRICE that maximizes revenue.
    This is the business heart of dynamic pricing!

    Key insight: Revenue = Price × Demand
    - Too low price → High demand but low revenue per unit
    - Too high price → Low demand, few units sold
    - SWEET SPOT in the middle maximizes total revenue
    """
    print("📊 Plot 7: Price vs Revenue (Optimal Pricing Zone)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('💰 Price vs Revenue — Finding the Optimal Price',
                 fontsize=15, fontweight='bold')

    # --- Binned Revenue by Price ---
    ax = axes[0]
    df['price_bin'] = pd.cut(df['price'], bins=25)
    binned = df.groupby('price_bin', observed=True).agg(
        avg_revenue=('revenue', 'mean'),
        avg_demand=('demand', 'mean')
    ).reset_index()
    binned['price_mid'] = binned['price_bin'].apply(lambda x: x.mid)

    # Revenue curve
    ax.plot(binned['price_mid'], binned['avg_revenue'],
            color=COLORS['success'], linewidth=2.5,
            marker='o', markersize=5, label='Avg Revenue')
    ax.fill_between(binned['price_mid'], binned['avg_revenue'],
                    alpha=0.15, color=COLORS['success'])

    # Mark the peak (optimal price)
    peak_idx = binned['avg_revenue'].idxmax()
    peak_price = binned.loc[peak_idx, 'price_mid']
    peak_rev = binned.loc[peak_idx, 'avg_revenue']
    ax.axvline(peak_price, color=COLORS['secondary'],
               linewidth=2, linestyle='--',
               label=f'Optimal ≈ ${peak_price:.0f}')
    ax.scatter([peak_price], [peak_rev],
               color=COLORS['secondary'], s=120, zorder=5)

    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Average Revenue ($)')
    ax.set_title('Revenue Curve — Optimal Price Point')
    ax.legend()

    # --- Heatmap: Revenue by Category and Price Bucket ---
    ax2 = axes[1]

    # Create price buckets: Low / Medium / High
    df['price_bucket'] = pd.qcut(
        df['price'],
        q=3,
        labels=['Low\n($10-70)', 'Medium\n($70-140)', 'High\n($140-200)']
    )

    pivot = df.groupby(
        ['product_category', 'price_bucket'], observed=True
    )['revenue'].mean().unstack()

    sns.heatmap(pivot, annot=True, fmt=',.0f',
                cmap='YlOrRd', ax=ax2,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10})
    ax2.set_title('Avg Revenue by Category × Price Tier')
    ax2.set_xlabel('Price Tier')
    ax2.set_ylabel('Product Category')

    df.drop(columns=['price_bin', 'price_bucket'], inplace=True)

    plt.tight_layout()
    save_figure('07_price_vs_revenue.png')


# ----------------------------------------------------------
# PLOT 8: FEATURE IMPORTANCE (CORRELATION WITH DEMAND)
# ----------------------------------------------------------
def plot_feature_correlation_with_demand(df):
    """
    Bar chart showing which features are most correlated with demand.
    This tells us BEFORE training which features will matter most to our ML model.
    Features with higher absolute correlation = better predictors.
    """
    print("📊 Plot 8: Feature Importance (Pre-ML Correlation Analysis)...")

    # Calculate correlation of every numerical column with demand
    num_cols = ['price', 'competitor_price', 'day_of_week', 'is_weekend',
                'hour_of_day', 'is_peak_hour', 'month', 'season_factor',
                'inventory_level', 'is_promotion', 'customer_rating']

    correlations = df[num_cols].corrwith(df['demand']).sort_values(
        key=abs, ascending=False   # Sort by absolute value
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('🎯 Feature Correlation with Demand\n'
                 '(Pre-ML Importance Estimate)',
                 fontsize=14, fontweight='bold')

    # Color bars: green if positive correlation, red if negative
    bar_colors = [COLORS['success'] if v >= 0
                  else COLORS['secondary']
                  for v in correlations.values]

    bars = ax.barh(
        correlations.index,
        correlations.values,
        color=bar_colors,
        alpha=0.85,
        edgecolor='white',
        linewidth=1.2
    )

    # Add value labels at end of each bar
    for bar, val in zip(bars, correlations.values):
        x_pos = val + 0.005 if val >= 0 else val - 0.005
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha=ha,
                fontsize=10, fontweight='bold')

    ax.axvline(0, color=COLORS['dark'], linewidth=1.2)
    ax.set_xlabel('Pearson Correlation with Demand')
    ax.set_title('')
    ax.set_xlim(-0.8, 0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='Positive Effect on Demand'),
        Patch(facecolor=COLORS['secondary'], label='Negative Effect on Demand')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_figure('08_feature_importance_pre_ml.png')

    # Print the findings
    print("\n  🔍 CORRELATION FINDINGS:")
    for feature, corr in correlations.items():
        direction = "↑ Increases" if corr > 0 else "↓ Decreases"
        print(f"     {feature:<22} : {corr:>7.3f}  → {direction} demand")


# ============================================================
# MAIN — RUN ALL EDA STEPS IN ORDER
# ============================================================

def main():
    print("="*60)
    print("   🔬 DYNAMIC PRICING ENGINE — EDA PIPELINE")
    print("="*60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Print summary
    print_data_summary(df)

    # Step 3: Generate all plots
    print("\n📈 Generating visualizations...")
    plot_distributions(df)
    plot_demand_vs_price(df)
    plot_demand_by_category(df)
    plot_time_patterns(df)
    plot_correlation_heatmap(df)
    plot_promotion_effect(df)
    plot_price_vs_revenue(df)
    plot_feature_correlation_with_demand(df)

    # Final summary
    print("\n" + "="*60)
    print("  ✅ EDA COMPLETE!")
    print(f"  📁 All 8 plots saved to: {config.FIGURES_DIR}")
    print("="*60)
    print("\n  📌 KEY TAKEAWAYS FROM EDA:")
    print("  1. Price has a STRONG NEGATIVE correlation with demand")
    print("  2. Promotions significantly BOOST demand")
    print("  3. Peak hours show HIGHER demand than off-peak")
    print("  4. Customer rating POSITIVELY impacts demand")
    print("  5. There's an optimal price point that maximizes revenue")
    print("  6. Electronics generates highest avg revenue per transaction")
    print("\n  ➡️  These insights will guide our ML feature selection!\n")


if __name__ == "__main__":
    main()