# startup.py
# ============================================================
# DYNAMIC PRICING ENGINE — STARTUP SCRIPT
# ============================================================
# PURPOSE : Run when the app first launches on Streamlit Cloud.
#           Checks if all required files exist.
#           If not, generates them automatically.
# ============================================================

import os
import sys

# Add root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import config

def check_file(path, name):
    """Check if a required file exists."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {path}")
    return exists

def run_pipeline_if_needed():
    """
    Check all required files.
    If any are missing, run the generation pipeline.
    """
    print("\n🔍 Checking required files for Streamlit Cloud...\n")

    # Files that must exist for the app to work
    required_files = {
        "Raw Dataset"       : config.RAW_DATA_FILE,
        "Train Data"        : os.path.join(
                                config.PROCESSED_DATA_DIR, 'train.csv'),
        "Test Data"         : os.path.join(
                                config.PROCESSED_DATA_DIR, 'test.csv'),
        "Feature Names"     : os.path.join(
                                config.PROCESSED_DATA_DIR,
                                'feature_names.txt'),
        "Scaler"            : os.path.join(
                                config.MODELS_DIR, 'scaler.pkl'),
        "XGBoost Model"     : os.path.join(
                                config.MODELS_DIR, 'xgboost.pkl'),
        "Random Forest"     : os.path.join(
                                config.MODELS_DIR, 'random_forest.pkl'),
        "Linear Regression" : os.path.join(
                                config.MODELS_DIR,
                                'linear_regression.pkl'),
    }

    all_exist = all(
        check_file(path, name)
        for name, path in required_files.items()
    )

    if all_exist:
        print("\n✅ All files found — App is ready!\n")
        return True

    print("\n⚠️  Some files missing — Running generation pipeline...\n")

    # Create required directories
    os.makedirs(config.RAW_DATA_DIR,       exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,         exist_ok=True)
    os.makedirs(config.FIGURES_DIR,        exist_ok=True)

    steps = [
        ("src/data_generator.py",       "📊 Generating dataset..."),
        ("src/feature_engineering.py",  "⚙️  Engineering features..."),
        ("src/train.py",                "🤖 Training models..."),
    ]

    import subprocess
    for script, message in steps:
        print(message)
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ❌ Failed: {result.stderr}")
            return False
        print(f"  ✅ Done")

    print("\n✅ Pipeline complete — App is ready!\n")
    return True


if __name__ == "__main__":
    run_pipeline_if_needed()