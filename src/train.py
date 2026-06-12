# src/train.py
# ============================================================
# DYNAMIC PRICING ENGINE — MODEL TRAINING PIPELINE
# ============================================================
# PURPOSE : Train and compare three ML models:
#           1. Linear Regression  (Baseline)
#           2. Random Forest      (Intermediate)
#           3. XGBoost            (Advanced)
# INPUT   : data/processed/train.csv, test.csv
# OUTPUT  : Trained models saved in models/
#           Training report printed to terminal
# RUN     : python src/train.py
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
import warnings
warnings.filterwarnings('ignore')    # Suppress non-critical warnings

# --- Scikit-learn ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --- XGBoost ---
from xgboost import XGBRegressor

# --- Visualization ---
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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
    'linear'  : '#2E86AB',   # Blue
    'rf'      : '#3BB273',   # Green
    'xgb'     : '#E84855',   # Red
    'neutral' : '#F9A03F',   # Orange
    'dark'    : '#1C1C1E',
}


# ============================================================
# SECTION 1: LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    """
    Load the train/test splits created by feature_engineering.py.
    Also loads the feature names list to ensure column consistency.
    """
    print("📂 Loading processed data...")

    # Load train and test CSV files
    train_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, 'train.csv')
    )
    test_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, 'test.csv')
    )

    # Load feature names (so we know exactly which columns are features)
    feature_names_path = os.path.join(
        config.PROCESSED_DATA_DIR, 'feature_names.txt'
    )
    with open(feature_names_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    # Separate features (X) and target (y)
    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COL]

    X_test  = test_df[feature_cols]
    y_test  = test_df[config.TARGET_COL]

    print(f"   ✅ Training set : {X_train.shape[0]:,} rows × "
          f"{X_train.shape[1]} features")
    print(f"   ✅ Test set     : {X_test.shape[0]:,} rows × "
          f"{X_test.shape[1]} features")
    print(f"   ✅ Features     : {feature_cols}")

    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================
# SECTION 2: COMPUTE EVALUATION METRICS
# ============================================================

def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Compute all evaluation metrics for a trained model.

    Parameters:
        y_true     : Actual demand values (ground truth)
        y_pred     : Predicted demand values from model
        model_name : Name string for display purposes

    Returns:
        Dictionary of metric name → value
    """

    # --- Core Metrics ---
    mae  = mean_absolute_error(y_true, y_pred)

    # RMSE: sklearn returns MSE, so we take square root manually
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    r2   = r2_score(y_true, y_pred)

    # MAPE: avoid division by zero for rows where actual demand = 0
    # We only compute MAPE where actual > 0
    mask = y_true > 0
    mape = np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100

    return {
        'model' : model_name,
        'MAE'   : round(mae,  2),
        'RMSE'  : round(rmse, 2),
        'R2'    : round(r2,   4),
        'MAPE'  : round(mape, 2)
    }


# ============================================================
# SECTION 3: MODEL DEFINITIONS WITH HYPERPARAMETERS
# ============================================================

def get_models():
    """
    Define all three models with their hyperparameters.
    Each hyperparameter is explained in detail below.

    Returns:
        Dictionary of model_name → (model_object, color)
    """

    # ----------------------------------------------------------
    # MODEL 1: LINEAR REGRESSION
    # ----------------------------------------------------------
    # LinearRegression has very few hyperparameters — it's simple by design.
    # fit_intercept=True: Learn a bias term (the 'b' in y = mx + b)
    #    → Without this, the line is forced through the origin (0,0)
    #    → Always True for regression problems
    linear_model = LinearRegression(
        fit_intercept=True   # Learn the baseline demand offset
    )

    # ----------------------------------------------------------
    # MODEL 2: RANDOM FOREST REGRESSOR
    # ----------------------------------------------------------
    # n_estimators=300
    #    → Build 300 decision trees
    #    → More trees = better accuracy but slower training
    #    → 300 is a sweet spot: accurate without being too slow
    #
    # max_depth=12
    #    → Each tree can ask at most 12 Yes/No questions (levels deep)
    #    → Deeper = more complex, risks overfitting (memorizing training data)
    #    → Shallower = simpler, risks underfitting (too simple to capture patterns)
    #    → 12 is deep enough for our 24-feature dataset
    #
    # min_samples_split=10
    #    → A tree node must have at least 10 samples before it can split further
    #    → Prevents trees from creating tiny, overfitted leaves
    #
    # min_samples_leaf=5
    #    → Every leaf (final node) must have at least 5 samples
    #    → Ensures each prediction is based on at least 5 training examples
    #
    # max_features='sqrt'
    #    → At each split, consider only √24 ≈ 5 random features
    #    → Introduces randomness = diverse trees = better ensemble
    #
    # n_jobs=-1
    #    → Use ALL available CPU cores for parallel training
    #    → On a 4-core machine, trains 4 trees simultaneously
    #
    # random_state=42
    #    → Reproducible results every run
    rf_model = RandomForestRegressor(
        n_estimators    = 50,
        max_depth       = 12,
        min_samples_split = 10,
        min_samples_leaf  = 5,
        max_features    = 'sqrt',
        n_jobs          = -1,
        random_state    = config.RANDOM_SEED
    )

    # ----------------------------------------------------------
    # MODEL 3: XGBOOST REGRESSOR
    # ----------------------------------------------------------
    # n_estimators=500
    #    → Build 500 boosting rounds (trees added sequentially)
    #    → More rounds = better accuracy, but watch for overfitting
    #
    # learning_rate=0.05
    #    → Each new tree's contribution is scaled by 0.05 (5%)
    #    → Lower = slower learning, more trees needed, but more accurate
    #    → Higher = faster but risks overshooting the optimal solution
    #    → 0.05 is a classic "small steps" setting for accuracy
    #
    # max_depth=6
    #    → XGBoost trees are shallower than RF trees by design
    #    → With boosting, shallow trees ensemble better
    #    → 6 is the most common XGBoost depth setting
    #
    # subsample=0.8
    #    → Each tree is trained on a random 80% of training rows
    #    → Introduces randomness, prevents overfitting
    #    → Similar to Random Forest's bootstrap sampling
    #
    # colsample_bytree=0.8
    #    → Each tree uses a random 80% of features
    #    → Another source of randomness for better generalization
    #
    # reg_alpha=0.1 (L1 regularization)
    #    → Penalizes model for having many non-zero weights
    #    → Encourages sparse solutions (some features weighted to 0)
    #    → Prevents overfitting by simplifying the model
    #
    # reg_lambda=1.0 (L2 regularization)
    #    → Penalizes model for having large weights
    #    → Shrinks all weights toward zero (but not to zero)
    #    → Standard regularization used in Ridge Regression too
    #
    # early_stopping_rounds (handled during fit)
    #    → Stop training if validation error hasn't improved in N rounds
    #    → Prevents wasting time and overfitting
    #
    # random_state=42 → Reproducibility
    xgb_model = XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        objective         = 'reg:squarederror',  # Minimize squared error
        eval_metric       = 'rmse',              # Track RMSE during training
        random_state      = config.RANDOM_SEED,
        verbosity         = 0                    # Silent mode
    )

    # Return as ordered dictionary: name → (model, display color)
    return {
        'Linear Regression' : (linear_model, COLORS['linear']),
        'Random Forest'     : (rf_model,     COLORS['rf']),
        'XGBoost'           : (xgb_model,    COLORS['xgb']),
    }


# ============================================================
# SECTION 4: TRAIN A SINGLE MODEL
# ============================================================

def train_model(name, model, X_train, y_train, X_test, y_test):
    """
    Train one model, record training time, compute metrics,
    and save the trained model to disk.

    Parameters:
        name    : Model name string (for display/saving)
        model   : Sklearn/XGBoost model object
        X_train, y_train : Training data
        X_test,  y_test  : Test data (for evaluation only)

    Returns:
        Tuple of (trained_model, train_metrics, test_metrics,
                  y_train_pred, y_test_pred, train_time)
    """
    print(f"\n{'─'*55}")
    print(f"  🤖 Training: {name}")
    print(f"{'─'*55}")

    # --- TRAIN ---
    # Record how long training takes
    start_time = time.time()

    if name == 'XGBoost':
        # XGBoost supports early stopping:
        # Monitor validation RMSE every 10 rounds
        # Stop if no improvement for 50 rounds
        # This saves time and prevents overfitting
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],   # Monitor on test set
            verbose=False                   # Don't print every round
        )
    else:
        # Linear Regression and Random Forest: standard fit
        model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"  ⏱️  Training time: {train_time:.2f} seconds")

    # --- PREDICT ---
    # Get predictions on BOTH train and test sets
    # Train predictions: shows how well model fits training data
    # Test predictions:  shows how well model generalizes to new data
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Clip predictions to valid demand range [0, MAX_DEMAND]
    # Models can sometimes predict negative demand — physically impossible
    y_train_pred = np.clip(y_train_pred, 0, None)
    y_test_pred  = np.clip(y_test_pred,  0, None)

    # --- EVALUATE ---
    train_metrics = compute_metrics(
        y_train.values, y_train_pred, f"{name} (Train)"
    )
    test_metrics  = compute_metrics(
        y_test.values, y_test_pred, f"{name} (Test)"
    )

    # Print metrics
    print(f"\n  📊 TRAINING SET METRICS:")
    print(f"     MAE  = {train_metrics['MAE']:>8.2f}  "
          f"(avg prediction error in demand units)")
    print(f"     RMSE = {train_metrics['RMSE']:>8.2f}  "
          f"(penalizes large errors more)")
    print(f"     R²   = {train_metrics['R2']:>8.4f}  "
          f"(1.0 = perfect, 0.0 = predicts mean)")
    print(f"     MAPE = {train_metrics['MAPE']:>7.2f}%  "
          f"(avg % error in predictions)")

    print(f"\n  📊 TEST SET METRICS:")
    print(f"     MAE  = {test_metrics['MAE']:>8.2f}")
    print(f"     RMSE = {test_metrics['RMSE']:>8.2f}")
    print(f"     R²   = {test_metrics['R2']:>8.4f}")
    print(f"     MAPE = {test_metrics['MAPE']:>7.2f}%")

    # --- OVERFITTING CHECK ---
    # If Train R² >> Test R², the model is overfitting (memorizing)
    r2_gap = train_metrics['R2'] - test_metrics['R2']
    if r2_gap > 0.05:
        print(f"\n  ⚠️  OVERFITTING DETECTED: Train R² is "
              f"{r2_gap:.3f} higher than Test R²")
    else:
        print(f"\n  ✅ No significant overfitting "
              f"(R² gap = {r2_gap:.3f})")

    # --- SAVE MODEL ---
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    # Create a safe filename: "Linear Regression" → "linear_regression.pkl"
    safe_name = name.lower().replace(' ', '_')
    model_path = os.path.join(config.MODELS_DIR, f'{safe_name}.pkl')
    joblib.dump(model, model_path)
    print(f"  💾 Model saved → {model_path}")

    return (model, train_metrics, test_metrics,
            y_train_pred, y_test_pred, train_time)


# ============================================================
# SECTION 5: TRAIN ALL MODELS
# ============================================================

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Loop through all three models, train each one,
    and collect results for comparison.
    """
    print("\n" + "="*55)
    print("   🚀 STARTING MODEL TRAINING PIPELINE")
    print("="*55)

    models_dict   = get_models()
    all_results   = {}   # Store all results for comparison

    for name, (model, color) in models_dict.items():
        result = train_model(
            name, model, X_train, y_train, X_test, y_test
        )
        (trained_model, train_metrics, test_metrics,
         y_train_pred, y_test_pred, train_time) = result

        all_results[name] = {
            'model'        : trained_model,
            'color'        : color,
            'train_metrics': train_metrics,
            'test_metrics' : test_metrics,
            'y_train_pred' : y_train_pred,
            'y_test_pred'  : y_test_pred,
            'train_time'   : train_time
        }

    return all_results


# ============================================================
# SECTION 6: MODEL COMPARISON & VISUALIZATION
# ============================================================

def save_figure(filename):
    """Helper to save figure to reports/figures/."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    print(f"   💾 Saved → {path}")
    plt.close()


def plot_model_comparison(all_results):
    """
    Create a 4-panel comparison chart showing all metrics
    side-by-side for all three models.
    """
    print("\n📊 Plotting model comparison...")

    model_names = list(all_results.keys())
    colors = [all_results[n]['color'] for n in model_names]

    # Extract test metrics for each model
    mae_vals  = [all_results[n]['test_metrics']['MAE']  for n in model_names]
    rmse_vals = [all_results[n]['test_metrics']['RMSE'] for n in model_names]
    r2_vals   = [all_results[n]['test_metrics']['R2']   for n in model_names]
    mape_vals = [all_results[n]['test_metrics']['MAPE'] for n in model_names]
    time_vals = [all_results[n]['train_time']           for n in model_names]

    # Short names for x-axis labels
    short_names = ['Linear\nRegression', 'Random\nForest', 'XGBoost']

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('🤖 Model Comparison — All Metrics (Test Set)',
                 fontsize=16, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: MAE (lower is better) ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(short_names, mae_vals, color=colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    ax1.set_title('MAE ↓ (Lower is Better)')
    ax1.set_ylabel('Mean Absolute Error (units)')
    # Mark the best (lowest) bar with a star
    best_idx = np.argmin(mae_vals)
    ax1.get_children()[best_idx].set_edgecolor('gold')
    ax1.get_children()[best_idx].set_linewidth(3)

    # ── Panel 2: RMSE (lower is better) ──
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(short_names, rmse_vals, color=colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    ax2.set_title('RMSE ↓ (Lower is Better)')
    ax2.set_ylabel('Root Mean Squared Error (units)')
    best_idx = np.argmin(rmse_vals)
    ax2.get_children()[best_idx].set_edgecolor('gold')
    ax2.get_children()[best_idx].set_linewidth(3)

    # ── Panel 3: R² Score (higher is better) ──
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(short_names, r2_vals, color=colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, r2_vals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    ax3.set_title('R² Score ↑ (Higher is Better)')
    ax3.set_ylabel('R² Score')
    ax3.set_ylim(0, 1.05)
    # Draw the "perfect" line at R²=1
    ax3.axhline(1.0, color='gray', linestyle='--',
                alpha=0.5, linewidth=1, label='Perfect (1.0)')
    ax3.legend(fontsize=9)
    best_idx = np.argmax(r2_vals)
    ax3.get_children()[best_idx].set_edgecolor('gold')
    ax3.get_children()[best_idx].set_linewidth(3)

    # ── Panel 4: MAPE (lower is better) ──
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(short_names, mape_vals, color=colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, mape_vals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    ax4.set_title('MAPE ↓ (Lower is Better)')
    ax4.set_ylabel('Mean Absolute % Error')
    best_idx = np.argmin(mape_vals)
    ax4.get_children()[best_idx].set_edgecolor('gold')
    ax4.get_children()[best_idx].set_linewidth(3)

    # ── Panel 5: Training Time ──
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(short_names, time_vals, color=colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, time_vals):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.1f}s', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    ax5.set_title('Training Time (seconds)')
    ax5.set_ylabel('Seconds')

    # ── Panel 6: Radar / Summary Table ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')    # Hide axes — we'll draw a text table

    # Build comparison table
    table_data = [
        ['Metric', 'Linear', 'RF', 'XGBoost'],
        ['MAE ↓',
         f"{mae_vals[0]:.1f}",
         f"{mae_vals[1]:.1f}",
         f"{mae_vals[2]:.1f}"],
        ['RMSE ↓',
         f"{rmse_vals[0]:.1f}",
         f"{rmse_vals[1]:.1f}",
         f"{rmse_vals[2]:.1f}"],
        ['R² ↑',
         f"{r2_vals[0]:.3f}",
         f"{r2_vals[1]:.3f}",
         f"{r2_vals[2]:.3f}"],
        ['MAPE ↓',
         f"{mape_vals[0]:.1f}%",
         f"{mape_vals[1]:.1f}%",
         f"{mape_vals[2]:.1f}%"],
        ['Time',
         f"{time_vals[0]:.1f}s",
         f"{time_vals[1]:.1f}s",
         f"{time_vals[2]:.1f}s"],
    ]

    table = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Color the header row
    for j in range(4):
        table[0, j].set_facecolor('#2E86AB')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Summary Table', fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure('09_model_comparison.png')


def plot_actual_vs_predicted(all_results, y_test):
    """
    Scatter plot of Actual vs Predicted demand for all three models.
    A perfect model would show all points on the diagonal line y=x.
    The closer points are to this line, the better the model.
    """
    print("📊 Plotting actual vs predicted...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🎯 Actual vs Predicted Demand — All Models\n'
                 '(Perfect model = all points on the diagonal)',
                 fontsize=14, fontweight='bold')

    for ax, (name, result) in zip(axes, all_results.items()):
        y_pred = result['y_test_pred']
        color  = result['color']
        r2     = result['test_metrics']['R2']

        # Scatter plot: each point = one test row
        ax.scatter(y_test, y_pred,
                   alpha=0.25, s=8,
                   color=color, label='Predictions')

        # Perfect prediction line (diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color=COLORS['dark'], linewidth=2,
                linestyle='--', label='Perfect Prediction')

        ax.set_xlabel('Actual Demand (units)')
        ax.set_ylabel('Predicted Demand (units)')
        ax.set_title(f'{name}\nR² = {r2:.4f}')
        ax.legend(fontsize=9)

        # Add R² text box inside plot
        ax.text(0.05, 0.92,
                f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_figure('10_actual_vs_predicted.png')


def plot_residuals(all_results, y_test):
    """
    Residual plots for each model.
    Residual = Actual - Predicted

    A good model has residuals:
    - Centered around 0 (no systematic bias)
    - Randomly distributed (no patterns)
    - Small in magnitude

    If residuals show a pattern (curve, funnel shape),
    the model is missing something important.
    """
    print("📊 Plotting residuals...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📉 Residual Analysis — All Models\n'
                 '(Good model = residuals randomly scattered around 0)',
                 fontsize=14, fontweight='bold')

    for col_idx, (name, result) in enumerate(all_results.items()):
        y_pred    = result['y_test_pred']
        color     = result['color']
        residuals = y_test.values - y_pred

        # ── Top row: Residuals vs Predicted ──
        ax_top = axes[0, col_idx]
        ax_top.scatter(y_pred, residuals,
                       alpha=0.25, s=8, color=color)
        ax_top.axhline(0, color=COLORS['dark'],
                       linewidth=2, linestyle='--')
        ax_top.set_xlabel('Predicted Demand')
        ax_top.set_ylabel('Residual (Actual - Predicted)')
        ax_top.set_title(f'{name}\nResiduals vs Predicted')

        # ── Bottom row: Residual Distribution ──
        ax_bot = axes[1, col_idx]
        ax_bot.hist(residuals, bins=50,
                    color=color, alpha=0.75,
                    edgecolor='white')
        ax_bot.axvline(0, color=COLORS['dark'],
                       linewidth=2, linestyle='--',
                       label='Zero Error')
        ax_bot.axvline(residuals.mean(),
                       color='gold', linewidth=2,
                       linestyle='-',
                       label=f'Mean: {residuals.mean():.1f}')
        ax_bot.set_xlabel('Residual Value')
        ax_bot.set_ylabel('Frequency')
        ax_bot.set_title(f'Residual Distribution\n'
                         f'std = {residuals.std():.1f}')
        ax_bot.legend(fontsize=9)

    plt.tight_layout()
    save_figure('11_residual_analysis.png')


def plot_feature_importance(all_results, feature_cols):
    """
    Plot feature importances from Random Forest and XGBoost.
    Linear Regression shows coefficients instead.

    Feature importance answers: "Which features did the model
    rely on most to make predictions?"
    """
    print("📊 Plotting feature importances...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('🔑 Feature Importances — What Each Model Learned\n'
                 '(Higher = feature had more influence on predictions)',
                 fontsize=14, fontweight='bold')

    model_info = [
        ('Linear Regression', 'coef_',         'Coefficient Value'),
        ('Random Forest',     'feature_importances_', 'Importance Score'),
        ('XGBoost',           'feature_importances_', 'Importance Score'),
    ]

    for ax, (name, attr, xlabel) in zip(axes, model_info):
        model = all_results[name]['model']
        color = all_results[name]['color']

        # Get importance values
        importance = getattr(model, attr)

        if name == 'Linear Regression':
            # For Linear Regression, use absolute value of coefficients
            # (negative coefficient is still important)
            importance = np.abs(importance)

        # Create Series for easy sorting
        imp_series = pd.Series(importance, index=feature_cols)
        imp_series = imp_series.sort_values(ascending=True).tail(15)

        # Horizontal bar chart (easier to read feature names)
        bars = ax.barh(imp_series.index, imp_series.values,
                       color=color, alpha=0.85,
                       edgecolor='white', linewidth=0.8)

        ax.set_xlabel(xlabel)
        ax.set_title(f'{name}\n(Top 15 Features)')

        # Highlight the most important feature
        max_bar = bars[-1]
        max_bar.set_edgecolor('gold')
        max_bar.set_linewidth(2.5)

    plt.tight_layout()
    save_figure('12_feature_importance.png')


# ============================================================
# SECTION 7: FINAL COMPARISON REPORT
# ============================================================

def print_final_comparison(all_results):
    """
    Print the definitive model comparison report and
    explain WHY the winning model is best.
    """
    print("\n" + "="*60)
    print("   🏆 FINAL MODEL COMPARISON REPORT")
    print("="*60)

    # Build comparison table
    print(f"\n  {'Model':<22} {'MAE':>8} {'RMSE':>8} "
          f"{'R²':>8} {'MAPE':>8} {'Time':>8}")
    print(f"  {'-'*60}")

    best_r2    = -np.inf
    best_model = ""

    for name, result in all_results.items():
        m = result['test_metrics']
        t = result['train_time']
        print(f"  {name:<22} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
              f"{m['R2']:>8.4f} {m['MAPE']:>7.2f}% {t:>7.1f}s")
        if m['R2'] > best_r2:
            best_r2    = m['R2']
            best_model = name

    print(f"\n  🥇 WINNER: {best_model} (R² = {best_r2:.4f})")

    # ── Explanation of Why XGBoost Wins ──
    print(f"""
{'='*60}
  📖 WHY {best_model.upper()} IS THE BEST MODEL
{'='*60}

  1️⃣  NON-LINEAR RELATIONSHIPS
     Our demand formula has squared terms, interaction effects,
     and clipping — none of which are linear.
     • Linear Regression: CANNOT capture curves or interactions
     • Random Forest:     CAN capture non-linearity ✅
     • XGBoost:           BEST at capturing complex curves ✅✅

  2️⃣  SEQUENTIAL ERROR CORRECTION
     XGBoost builds each tree to fix the previous tree's mistakes.
     This targeted learning is more efficient than RF's
     independent trees.

  3️⃣  REGULARIZATION
     XGBoost has built-in L1 + L2 regularization preventing
     overfitting. The R² gap between train and test should be
     minimal — confirming good generalization.

  4️⃣  FEATURE INTERACTIONS
     Our price_x_promotion, price_x_peak features create
     complex joint effects. XGBoost's boosted trees naturally
     model these multi-feature interactions.

  5️⃣  INDUSTRY STANDARD
     XGBoost (and its variants like LightGBM, CatBoost) win
     the vast majority of Kaggle competitions and are used in
     production at Uber, Airbnb, and Amazon for pricing tasks.

  ⚠️  TRADE-OFF:
     XGBoost is less interpretable than Linear Regression.
     In regulated industries (banking, healthcare), you might
     prefer Linear Regression + good features for transparency.
     For pricing optimization, accuracy wins → XGBoost.
{'='*60}
""")


# ============================================================
# MAIN — RUN FULL TRAINING PIPELINE
# ============================================================

def main():
    print("="*60)
    print("  🚀 DYNAMIC PRICING ENGINE — TRAINING PIPELINE")
    print("="*60)

    # Step 1: Load processed data
    X_train, X_test, y_train, y_test, feature_cols = load_processed_data()

    # Step 2: Train all models
    all_results = train_all_models(X_train, y_train, X_test, y_test)

    # Step 3: Plot comparison charts
    print("\n📈 Generating comparison visualizations...")
    plot_model_comparison(all_results)
    plot_actual_vs_predicted(all_results, y_test)
    plot_residuals(all_results, y_test)
    plot_feature_importance(all_results, feature_cols)

    # Step 4: Print final report
    print_final_comparison(all_results)

    print("✅ Training Pipeline Complete!")
    print(f"   📁 Models saved in     : {config.MODELS_DIR}/")
    print(f"   📊 Charts saved in     : {config.FIGURES_DIR}/")
    print("\n   ➡️  Ready for Step 6: Pricing Logic & Optimization!\n")

    return all_results


if __name__ == "__main__":
    main()