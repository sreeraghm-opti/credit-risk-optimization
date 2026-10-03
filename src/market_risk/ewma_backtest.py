from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "market" / "nifty50_prices.csv"
DETAIL_FILE = ROOT / "reports" / "market" / "ewma_backtest.csv"
SUMMARY_FILE = ROOT / "reports" / "market" / "ewma_backtest_summary.csv"

WINDOW = 250
LAMBDA = 0.94
CONFIDENCE_LEVELS = (0.95, 0.99)


def kupiec_test(flags, confidence):
    flags = np.asarray(flags, dtype=int)
    n = len(flags)
    x = int(flags.sum())
    p = 1 - confidence

    if n == 0:
        return np.nan, np.nan

    observed = x / n

    def ll(successes, trials, probability):
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        failures = trials - successes
        return (
            successes * np.log(probability)
            + failures * np.log(1 - probability)
        )

    lr = max(0.0, -2 * (ll(x, n, p) - ll(x, n, observed)))
    return float(lr), float(chi2.sf(lr, 1))


def independence_test(flags):
    flags = np.asarray(flags, dtype=int)
    previous, current = flags[:-1], flags[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    def ll(successes, trials, p):
        if trials == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return (
            successes * np.log(p)
            + (trials - successes) * np.log(1 - p)
        )

    total = n00 + n01 + n10 + n11
    if total == 0:
        return np.nan, np.nan

    pooled = (n01 + n11) / total
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    ll_null = ll(n01 + n11, total, pooled)
    ll_alt = (
        ll(n01, n00 + n01, p01)
        + ll(n11, n10 + n11, p11)
    )

    lr = max(0.0, 2 * (ll_alt - ll_null))
    return float(lr), float(chi2.sf(lr, 1))


def main():
    data = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates("Date")
    data["return"] = data["Close"].pct_change()
    data = data.dropna(subset=["return"]).reset_index(drop=True)

    returns = data["return"].to_numpy()
    dates = data["Date"]

    if len(returns) <= WINDOW:
        raise ValueError("Insufficient return observations.")

    # Initialize variance from the first estimation window.
    variance = float(np.var(returns[:WINDOW], ddof=1))
    forecasts = []

    # At each date, forecast from information available before that return.
    for i in range(WINDOW, len(returns)):
        if i > WINDOW:
            variance = (
                LAMBDA * variance
                + (1 - LAMBDA) * returns[i - 1] ** 2
            )

        sigma = np.sqrt(max(variance, 0.0))

        for confidence in CONFIDENCE_LEVELS:
            var = float(norm.ppf(confidence) * sigma)
            realised_loss = -returns[i]

            forecasts.append({
                "date": dates.iloc[i],
                "method": "EWMA normal",
                "confidence": confidence,
                "lambda": LAMBDA,
                "var_return": var,
                "realised_return": returns[i],
                "realised_loss": realised_loss,
                "exception": int(realised_loss > var),
            })

    detail = pd.DataFrame(forecasts)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_FILE, index=False)

    summary_rows = []

    for confidence, group in detail.groupby("confidence", sort=True):
        flags = group.sort_values("date")["exception"].to_numpy()
        n = len(flags)
        exceptions = int(flags.sum())

        kupiec_lr, kupiec_p = kupiec_test(flags, confidence)
        ind_lr, ind_p = independence_test(flags)

        cc_lr = kupiec_lr + ind_lr
        cc_p = float(chi2.sf(cc_lr, df=2))

        summary_rows.append({
            "method": "EWMA normal",
            "confidence": confidence,
            "observations": n,
            "exceptions": exceptions,
            "expected_exceptions": n * (1 - confidence),
            "exception_rate": exceptions / n,
            "kupiec_lr_stat": kupiec_lr,
            "kupiec_p_value": kupiec_p,
            "independence_lr_stat": ind_lr,
            "independence_p_value": ind_p,
            "conditional_coverage_lr_stat": cc_lr,
            "conditional_coverage_p_value": cc_p,
            "conditional_coverage_reject_5pct": bool(cc_p < 0.05),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("=" * 72)
    print("EWMA VOLATILITY — CHRONOLOGICAL VaR BACKTEST")
    print("=" * 72)
    print(f"Estimation window: {WINDOW} trading days")
    print(f"Decay factor: {LAMBDA}")
    print(f"Backtest observations: {len(returns) - WINDOW}")
    print("Expected return assumption: zero")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved detailed results:", DETAIL_FILE)
    print("Saved summary:", SUMMARY_FILE)
    print(
        "\nCaution: EWMA is a volatility model, not a guarantee of calibrated "
        "VaR. Test results are sample-dependent; non-rejection does not "
        "establish that the model is adequate."
    )


if __name__ == "__main__":
    main()
