from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "merged" / "market_close_2001_2026.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETURN_VARS = ["JCI", "SP500", "SSE", "USDIDR", "WTI", "Gold"]


def load_merged_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Input merged file must contain 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    expected = RETURN_VARS + ["VIX"]
    missing_cols = [col for col in expected if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in merged file: {missing_cols}")

    return df


def handle_missing_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    print("Missing values before fill:")
    print(df.isnull().sum())

    # Same handling as TVP_VAR_clean.ipynb
    df = df.ffill().bfill()

    print(f"\nClean data: {df.shape[0]} observations")
    print(f"Missing values after fill: {df.isnull().sum().sum()}")
    return df


def compute_returns_and_rv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_ret = pd.DataFrame(index=df.index)

    for var in RETURN_VARS:
        df_ret[f"{var}_ret"] = np.log(df[var] / df[var].shift(1))

    # Keep VIX return for transformation completeness
    df_ret["VIX_ret"] = np.log(df["VIX"] / df["VIX"].shift(1))

    # Realized Volatility (5-day rolling std of percentage returns, annualized)
    df_rv = pd.DataFrame(index=df.index)
    for var in RETURN_VARS:
        daily_pct_ret = df_ret[f"{var}_ret"] * 100
        df_rv[f"{var}_RV"] = daily_pct_ret.rolling(window=5).std() * np.sqrt(252)

    # VIX level as implied volatility level
    df_rv["VIX_Level"] = df["VIX"]

    df_transformed = pd.concat([df_ret, df_rv], axis=1).dropna()
    return df_ret, df_rv, df_transformed


def main() -> None:
    print(f"Loading merged data from: {INPUT_FILE}")
    df = load_merged_data(INPUT_FILE)

    df_clean = handle_missing_like_notebook(df)
    df_clean.to_csv(OUTPUT_DIR / "raw_data_filled.csv")

    df_ret, df_rv, df_transformed = compute_returns_and_rv(df_clean)

    df_ret_clean = df_transformed.filter(like="_ret")
    df_rv_clean = df_transformed.filter(like="_RV").join(df_transformed["VIX_Level"])

    print(f"\nTransformed return data shape: {df_ret_clean.shape}")
    print(f"Transformed volatility data shape: {df_rv_clean.shape}")
    print(f"Period: {df_transformed.index.min().date()} to {df_transformed.index.max().date()}")

    df_ret_clean.to_csv(OUTPUT_DIR / "returns.csv")
    df_rv_clean.to_csv(OUTPUT_DIR / "volatility_system.csv")
    df_transformed.to_csv(OUTPUT_DIR / "transformed_full.csv")

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'raw_data_filled.csv'}")
    print(f"- {OUTPUT_DIR / 'returns.csv'}")
    print(f"- {OUTPUT_DIR / 'volatility_system.csv'}")
    print(f"- {OUTPUT_DIR / 'transformed_full.csv'}")


if __name__ == "__main__":
    main()
