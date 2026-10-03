from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "market" / "nifty50_prices.csv"
OUTPUT_FILE = ROOT / "reports" / "market" / "var_backtest.csv"

WINDOW = 250
CONFIDENCE_LEVELS = (0.95, 0.99)


def kupiec_test(exceptions, observations, confidence):
    """Kupiec unconditional coverage likelihood-ratio test."""
    x = int(exceptions)
    n = int(observations)
    p = 1.0 - confidence

    if n == 0:
        return np.nan, np.nan

    phat = x / n

    def log_likelihood(probability):
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        return (
            x * np.log(probability)
            + (n - x) * np.log(1 - probability)
        )

    lr_stat = max(
        0.0,
        -2 * (log_likelihood(p) - log_likelihood(phat)),
    )
    p_value = chi2.sf(lr_stat, df=1)

    return float(lr_stat), float(p_value)


def calculate_var(training_returns, confidence, method):
    losses = -np.asarray(training_returns, dtype=float)

    if method == "Historical":
        return float(np.quantile(losses, confidence))

    if method == "Parametric normal":
        mu = np.mean(training_returns)
        sigma = np.std(training_returns, ddof=1)
        return float(max(0.0, -(mu - norm.ppf(confidence) * sigma)))

    raise ValueError(f"Unknown method: {method}")


def main():
    data = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates("Date")
    data["return"] = data["Close"].pct_change()
    data = data.dropna(subset=["return"]).reset_index(drop=True)

    returns = data["return"].to_numpy()
    dates = data["Date"]

    if len(returns) <= WINDOW:
        raise ValueError("Not enough returns for the chosen estimation window.")

    # At each date, estimate VaR only from preceding observations.
    results = []

    for method in ("Historical", "Parametric normal"):
        for confidence in CONFIDENCE_LEVELS:
            for i in range(WINDOW, len(returns)):
                training = returns[i - WINDOW:i]
                realised_return = returns[i]
                var = calculate_var(training, confidence, method)
                realised_loss = -realised_return
                exception = realised_loss > var

                results.append({
                    "date": dates.iloc[i],
                    "method": method,
                    "confidence": confidence,
                    "var_return": var,
                    "realised_return": realised_return,
                    "realised_loss": realised_loss,
                    "exception": int(exception),
                })

    detail = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_FILE, index=False)

    summary_rows = []

    for (method, confidence), group in detail.groupby(
        ["method", "confidence"], sort=False
    ):
        n = len(group)
        exceptions = int(group["exception"].sum())
        expected = n * (1 - confidence)
        rate = exceptions / n

        lr_stat, p_value = kupiec_test(
            exceptions, n, confidence
        )

        summary_rows.append({
            "method": method,
            "confidence": confidence,
            "backtest_observations": n,
            "exceptions": exceptions,
            "expected_exceptions": expected,
            "exception_rate": rate,
            "kupiec_lr_stat": lr_stat,
            "kupiec_p_value": p_value,
            "coverage_test_reject_5pct": (
                bool(p_value < 0.05) if np.isfinite(p_value) else None
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary_file = ROOT / "reports" / "market" / "var_backtest_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("=" * 72)
    print("CHRONOLOGICAL VaR BACKTEST")
    print("=" * 72)
    print(f"Estimation window: {WINDOW} trading days")
    print(f"Backtest observations per model: {len(returns) - WINDOW}")
    print("Exception: realised loss strictly exceeds the forecast VaR.")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved detailed results:", OUTPUT_FILE)
    print("Saved summary:", summary_file)
    print(
        "\nInterpretation: the Kupiec test evaluates unconditional "
        "exception frequency, not exception clustering or all aspects "
        "of model adequacy. A non-rejection does not prove the model is valid."
    )


if __name__ == "__main__":
    main()
