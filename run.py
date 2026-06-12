# run.py
# ============================================================
# DYNAMIC PRICING ENGINE — ONE-COMMAND PIPELINE RUNNER
# ============================================================
# PURPOSE : Run the entire ML pipeline from scratch in one go.
# USAGE   : python run.py
#           python run.py --skip-eda        (skip EDA charts)
#           python run.py --app-only        (only launch app)
# ============================================================

import subprocess
import sys
import os
import time
import argparse

# ── Color codes for terminal output ──
GREEN  = '\033[92m'
BLUE   = '\033[94m'
YELLOW = '\033[93m'
RED    = '\033[91m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def print_step(step_num, text):
    print(f"\n{BOLD}{YELLOW}▶ Step {step_num}: {text}{RESET}")

def print_success(text):
    print(f"{GREEN}  ✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}  ❌ {text}{RESET}")

def run_script(script_path, step_name):
    """Run a Python script and report success/failure."""
    print_step("", step_name)
    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # Show output in real time
        text=True
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print_success(f"{step_name} completed in {elapsed:.1f}s")
        return True
    else:
        print_error(f"{step_name} FAILED after {elapsed:.1f}s")
        return False


def main():
    # ── Parse arguments ──
    parser = argparse.ArgumentParser(
        description='Dynamic Pricing Engine Pipeline Runner'
    )
    parser.add_argument(
        '--skip-eda', action='store_true',
        help='Skip EDA visualizations (faster)'
    )
    parser.add_argument(
        '--app-only', action='store_true',
        help='Skip training, only launch the Streamlit app'
    )
    args = parser.parse_args()

    print_header("🏷️  DYNAMIC PRICING ENGINE — FULL PIPELINE")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {os.getcwd()}")

    if args.app_only:
        print(f"\n{YELLOW}  ℹ️  App-only mode: Skipping training pipeline{RESET}")
        print(f"\n{BOLD}🌐 Launching Streamlit App...{RESET}")
        os.system("streamlit run app/app.py")
        return

    total_start = time.time()
    steps = []

    # ── Step 1: Data Generation ──
    steps.append(run_script(
        'src/data_generator.py',
        '📊 Data Generation (10,000 transactions)'
    ))

    # ── Step 2: EDA (optional) ──
    if not args.skip_eda:
        steps.append(run_script(
            'notebooks/eda.py',
            '🔍 Exploratory Data Analysis (8 charts)'
        ))
    else:
        print(f"\n{YELLOW}  ⏭️  Skipping EDA (--skip-eda flag){RESET}")

    # ── Step 3: Feature Engineering ──
    steps.append(run_script(
        'src/feature_engineering.py',
        '⚙️  Feature Engineering (24 features)'
    ))

    # ── Step 4: Model Training ──
    steps.append(run_script(
        'src/train.py',
        '🤖 Model Training (Linear + RF + XGBoost)'
    ))

    # ── Step 5: Pricing Logic ──
    steps.append(run_script(
        'src/pricing_logic.py',
        '💰 Pricing Optimization Engine'
    ))

    # ── Summary ──
    total_time = time.time() - total_start
    success_count = sum(steps)
    total_count   = len(steps)

    print_header("📋 PIPELINE SUMMARY")
    print(f"  Steps completed : {success_count}/{total_count}")
    print(f"  Total time      : {total_time:.1f} seconds")

    if success_count == total_count:
        print(f"\n{GREEN}{BOLD}  ✅ ALL STEPS COMPLETED SUCCESSFULLY!{RESET}")
        print(f"\n{BOLD}🌐 Launching Streamlit App...{RESET}")
        print(f"  Open: http://localhost:8501\n")
        os.system("streamlit run app/app.py")
    else:
        print(f"\n{RED}{BOLD}  ❌ {total_count - success_count} STEP(S) FAILED{RESET}")
        print(f"  Check the error messages above and fix before retrying.")
        sys.exit(1)


if __name__ == "__main__":
    main()