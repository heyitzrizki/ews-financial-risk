from pathlib import Path
import subprocess
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PIPELINE_SCRIPTS = [
    BASE_DIR / "src" / "data_loader.py",
    BASE_DIR / "src" / "preprocessing.py",
    BASE_DIR / "src" / "tvp_var.py",
    BASE_DIR / "src" / "regime_detection.py",
    BASE_DIR / "src" / "predictive_model.py",
]

LATEST_SIGNAL_PATH = BASE_DIR / "data" / "processed" / "predictive" / "latest_signal.csv"


def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print(f"\n[RUN] {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"Script failed: {script_path.name}")


def print_latest_signal(path: Path) -> None:
    if not path.exists():
        print("\nNo latest signal file found.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("\nLatest signal file exists but is empty.")
        return

    latest = df.iloc[-1]
    print("\n=== Final EWS Signal ===")
    print(f"Date      : {latest['Date']}")
    print(f"Model     : {latest['model']}")
    print(f"Horizon   : {int(latest['horizon'])} days")
    print(f"Risk Prob : {float(latest['y_prob']):.4f}")
    print(f"Alert     : {int(latest['y_pred'])}")
    print(f"Threshold : {float(latest['threshold']):.4f}")


def main() -> None:
    print("Running end-to-end EWS pipeline...")
    for script_path in PIPELINE_SCRIPTS:
        run_script(script_path)

    print_latest_signal(LATEST_SIGNAL_PATH)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
