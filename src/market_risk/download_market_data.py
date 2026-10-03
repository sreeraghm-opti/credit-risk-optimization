from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "nifty50_prices.csv"

TICKER = "^NSEI"
START_DATE = "2015-01-01"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading NIFTY 50 historical market data...")

    data = yf.download(
        TICKER,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError(
            "No market data returned. Check your internet connection "
            "and Yahoo Finance availability."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_localize(None)

    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = [
        column for column in
        ["Date", "Open", "High", "Low", "Close", "Volume"]
        if column in data.columns
    ]

    data = (
        data[keep]
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .dropna(subset=["Close"])
    )

    if (data["Close"] <= 0).any():
        raise ValueError("Non-positive closing prices detected.")

    data.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    print("\nDOWNLOAD COMPLETE")
    print(f"Observations: {len(data):,}")
    print(f"First date: {data['Date'].min().date()}")
    print(f"Last date: {data['Date'].max().date()}")
    print(f"Missing closing prices: {data['Close'].isna().sum()}")
    print(f"Duplicate dates: {data['Date'].duplicated().sum()}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nMost recent observations:")
    print(data.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
