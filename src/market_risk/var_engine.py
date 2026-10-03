from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "nifty50_prices.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "market" / "var_summary.csv"

CONFIDENCE_LEVELS = (0.95, 0.99)
WINDOW = 250
SIMULATIONS = 100_000
PORTFOLIO_VALUE = 10_000_000  # Illustrative ₹1 crore exposure
RANDOM_SEED = 42


def historical_var_es(returns, confidence):
    """Return positive loss VaR and Expected Shortfall."""
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, confidence)
    tail_losses = losses[losses >= var]
    es = tail_losses.mean() if len(tail_losses) else var
    return float(var), float(es)


def parametric_var_es(returns, confidence):
    """Normal-distribution VaR and ES, expressed as positive losses."""
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if sigma <= 0:
        return max(0.0, float(-mu)), max(0.0, float(-mu))

    z = norm.ppf(confidence)
    var = -(mu - z * sigma)
    es = -mu + sigma * norm.pdf(z) / (1 - confidence)

    return float(max(0.0, var)), float(max(0.0, es))


def monte_carlo_var_es(returns, confidence, rng):
    """Simulate normally distributed returns fitted to historical data."""
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean()
    sigma = returns.std(ddof=1)

    simulated_returns = rng.normal(mu, sigma, size=SIMULATIONS)
    losses = -simulated_returns

    var = np.quantile(losses, confidence)
    tail_losses = losses[losses >= var]
    es = tail_losses.mean() if len(tail_losses) else var

    return float(max(0.0, var)), float(max(0.0, es))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Market data not found: {INPUT_FILE}")

    prices = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    prices = prices.sort_values("Date").drop_duplicates("Date")

    if prices["Close"].isna().any() or (prices["Close"] <= 0).any():
        raise ValueError("Closing prices contain invalid values.")

    prices["return"] = prices["Close"].pct_change()
    returns = prices.set_index("Date")["return"].dropna()

    if len(returns) < WINDOW:
        raise ValueError(f"At least {WINDOW} returns are required.")

    estimation_returns = returns.tail(WINDOW)

    print("=" * 68)
    print("NIFTY 50 — ONE-DAY VaR & EXPECTED SHORTFALL")
    print("=" * 68)
    print(f"Return observations available: {len(returns):,}")
    print(f"Estimation window: {len(estimation_returns)} trading days")
    print(
        f"Estimation period: {estimation_returns.index.min().date()} "
        f"to {estimation_returns.index.max().date()}"
    )
    print(f"Illustrative exposure: ₹{PORTFOLIO_VALUE:,.0f}")
    print("Convention: positive figures represent losses.")
    print("\nRisk estimates:")

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for confidence in CONFIDENCE_LEVELS:
        methods = {
            "Historical": historical_var_es(
                estimation_returns, confidence
            ),
            "Parametric normal": parametric_var_es(
                estimation_returns, confidence
            ),
            "Monte Carlo normal": monte_carlo_var_es(
                estimation_returns, confidence, rng
            ),
        }

        for method, (var, es) in methods.items():
            row = {
                "confidence": confidence,
                "method": method,
                "var_return": var,
                "expected_shortfall_return": es,
                "var_amount": var * PORTFOLIO_VALUE,
                "expected_shortfall_amount": es * PORTFOLIO_VALUE,
                "window_observations": len(estimation_returns),
                "portfolio_value_assumption": PORTFOLIO_VALUE,
            }
            results.append(row)

            print(
                f"{confidence:.0%} | {method:<20} "
                f"VaR={var:.3%} (₹{var * PORTFOLIO_VALUE:,.0f}) | "
                f"ES={es:.3%} (₹{es * PORTFOLIO_VALUE:,.0f})"
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved summary: {OUTPUT_FILE}")
    print(
        "\nCaution: estimates are based on historical index returns and "
        "a normal-distribution model. They are not forecasts or regulatory "
        "capital estimates. The ₹1 crore exposure is illustrative."
    )


if __name__ == "__main__":
    main()
