from pathlib import Path
import pandas as pd
import yfinance as yf

START = "2001-12-31"
END = "2026-04-03"

TICKERS = {
    "^JKSE": "JCI",
    "^GSPC": "SP500",
    "000001.SS": "SSE",
    "^VIX": "VIX",
    "CL=F": "WTI",
    "GC=F": "Gold",
    "IDR=X": "USDIDR",
}

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
MERGED_DIR = BASE / "data" / "merged"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def download_one_ticker(ticker: str, name: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START,
        end=END,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )

    if df.empty and name == "USDIDR":
        print("IDR=X empty, trying USDIDR=X ...")
        df = yf.download(
            "USDIDR=X",
            start=START,
            end=END,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )

    return df


def standardize_price_table(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Date").reset_index()

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    out = df.loc[:, keep].copy()
    out["Date"] = pd.to_datetime(out["Date"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            col = out.loc[:, c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            out[c] = pd.to_numeric(col, errors="coerce")

    for c in ["Open", "High", "Low", "Close"]:
        if c in out.columns:
            out[c] = out[c].round(4)

    return out


def main() -> None:
    print("BASE:", BASE)
    series_map = {}

    for ticker, name in TICKERS.items():
        print(f"Downloading {name} ({ticker}) ...")
        raw_df = download_one_ticker(ticker, name)

        if raw_df.empty:
            print(f"[WARN] No data for {name}")
            continue

        out = standardize_price_table(raw_df)
        out.to_csv(RAW_DIR / f"{name}.csv", index=False)
        series_map[name] = out.set_index("Date")["Close"]
        print(f"Saved: {RAW_DIR / f'{name}.csv'}")

    if not series_map:
        raise ValueError("No data downloaded. Check internet or ticker symbols.")

    merged = pd.concat(series_map, axis=1).sort_index()
    merged.columns.name = None
    merged = merged.reset_index()

    merged.to_csv(MERGED_DIR / "market_close_2001_2026.csv", index=False)

    print("\nDone.")
    print("Saved:", MERGED_DIR / "market_close_2001_2026.csv")
    print("Columns:", list(merged.columns))
    print("Shape:", merged.shape)


if __name__ == "__main__":
    main()