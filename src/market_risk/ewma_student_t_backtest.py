from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, t


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "market" / "nifty50_prices.csv"
DETAIL_FILE = ROOT / "reports" / "market" / "ewma_student_t_backtest.csv"
SUMMARY_FILE = ROOT / "reports" / "market" / "ewma_student_t_backtest_summary.csv"

WINDOW = 250
LAMBDA = 0.94
CONFIDENCE_LEVELS = (0.95, 0.99)

# Bound degrees of freedom so the fitted Student-t distribution
# has finite variance and numerically stable tail quantiles.
MIN_DF = 2.1
MAX_DF = 100.0


def kupiec_test(flags, confidence):
    flags = np.asarray(flags, dtype=int)
    n = len(flags)
    x = int(flags.sum())
    expected_probability = 1.0 - confidence

    if n == 0:
        return np.nan, np.nan

    observed_probability = x / n

    def log_likelihood(successes, trials, probability):
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        return (
            successes * np.log(probability)
            + (trials - successes) * np.log(1 - probability)
        )

    lr = -2 * (
        log_likelihood(x, n, expected_probability)
        - log_likelihood(x, n, observed_probability)
    )

    lr = max(0.0, float(lr))
    return lr, float(chi2.sf(lr, df=1))


def independence_test(flags):
    """Christoffersen first-order Markov independence test."""
    flags = np.asarray(flags, dtype=int)
    previous = flags[:-1]
    current = flags[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    def log_likelihood(successes, trials, probability):
        if trials == 0:
            return 0.0

        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        return (
            successes * np.log(probability)
            + (trials - successes) * np.log(1 - probability)
        )

    total = n00 + n01 + n10 + n11
    if total == 0:
        return np.nan, np.nan

    pooled_probability = (n01 + n11) / total
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    ll_null = log_likelihood(
        n01 + n11, total, pooled_probability
    )
    ll_markov = (
        log_likelihood(n01, n00 + n01, p01)
        + log_likelihood(n11, n10 + n11, p11)
    )

    lr = max(0.0, 2 * (ll_markov - ll_null))
    return float(lr), float(chi2.sf(lr, df=1))


def fit_student_t_df(training_returns):
    """
    Estimate Student-t degrees of freedom from the training window.

    Standardisation and fitting use training data only. The fitted
    distribution's location and scale are not used as the forecast
    volatility; EWMA supplies the time-varying volatility instead.
    """
    values = np.asarray(training_returns, dtype=float)
    sample_sd = np.std(values, ddof=1)

    if not np.isfinite(sample_sd) or sample_sd <= 0:
        return 10.0

    standardized = (values - np.mean(values)) / sample_sd

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_df, _, _ = t.fit(standardized, floc=0)

        if not np.isfinite(fitted_df):
            return 10.0

        return float(np.clip(fitted_df, MIN_DF, MAX_DF))

    except (ValueError, RuntimeError, FloatingPointError):
        return 10.0


def main():
    data = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    data = (
        data.sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    data["return"] = data["Close"].pct_change()
    data = data.dropna(subset=["return"]).reset_index(drop=True)

    returns = data["return"].to_numpy()
    dates = data["Date"]

    if len(returns) <= WINDOW:
        raise ValueError("Insufficient returns for the estimation window.")

    # Initial variance uses only the first 250 observations.
    variance = float(np.var(returns[:WINDOW], ddof=1))
    forecasts = []

    for i in range(WINDOW, len(returns)):
        # Update using the previous day's return, before forecasting i.
        if i > WINDOW:
            variance = (
                LAMBDA * variance
                + (1.0 - LAMBDA) * returns[i - 1] ** 2
            )

        sigma = float(np.sqrt(max(variance, 0.0)))

        # Fit tail thickness using only the trailing estimation window.
        training = returns[i - WINDOW:i]
        degrees_freedom = fit_student_t_df(training)

        for confidence in CONFIDENCE_LEVELS:
            # Standardise Student-t innovations to unit variance:
            # Var(T_df) = df / (df - 2), for df > 2.
            raw_quantile = t.ppf(confidence, df=degrees_freedom)
            standardized_quantile = (
                raw_quantile
                * np.sqrt((degrees_freedom - 2.0) / degrees_freedom)
            )

            var_forecast = max(0.0, sigma * standardized_quantile)
            realised_loss = -returns[i]

            forecasts.append({
                "date": dates.iloc[i],
                "method": "EWMA Student-t",
                "confidence": confidence,
                "lambda": LAMBDA,
                "student_t_df": degrees_freedom,
                "forecast_volatility": sigma,
                "var_return": var_forecast,
                "realised_return": returns[i],
                "realised_loss": realised_loss,
                "exception": int(realised_loss > var_forecast),
            })

    detail = pd.DataFrame(forecasts)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_FILE, index=False)

    summary_rows = []

    for confidence, group in detail.groupby("confidence", sort=True):
        group = group.sort_values("date")
        flags = group["exception"].to_numpy(dtype=int)
        n = len(flags)
        exceptions = int(flags.sum())

        kupiec_lr, kupiec_p = kupiec_test(flags, confidence)
        independence_lr, independence_p = independence_test(flags)

        conditional_lr = kupiec_lr + independence_lr
        conditional_p = float(chi2.sf(conditional_lr, df=2))

        summary_rows.append({
            "method": "EWMA Student-t",
            "confidence": confidence,
            "observations": n,
            "exceptions": exceptions,
            "expected_exceptions": n * (1 - confidence),
            "exception_rate": exceptions / n,
            "mean_student_t_df": group["student_t_df"].mean(),
            "median_student_t_df": group["student_t_df"].median(),
            "kupiec_lr_stat": kupiec_lr,
            "kupiec_p_value": kupiec_p,
            "independence_lr_stat": independence_lr,
            "independence_p_value": independence_p,
            "conditional_coverage_lr_stat": conditional_lr,
            "conditional_coverage_p_value": conditional_p,
            "conditional_coverage_reject_5pct": bool(conditional_p < 0.05),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("=" * 76)
    print("EWMA STUDENT-t — CHRONOLOGICAL VaR BACKTEST")
    print("=" * 76)
    print(f"Initial estimation window: {WINDOW} trading days")
    print(f"EWMA decay factor: {LAMBDA}")
    print(f"Backtest observations: {len(returns) - WINDOW}")
    print("Student-t degrees of freedom: refitted on each trailing window")
    print("Expected return assumption: zero")
    print("\nSummary:")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    print("\nSaved detailed results:", DETAIL_FILE)
    print("Saved summary:", SUMMARY_FILE)
    print(
        "\nCaution: Student-t innovations allow heavier tails, but this does "
        "not guarantee better forecasts. Degrees of freedom are bounded, "
        "and results remain dependent on the sample and model assumptions."
    )


if __name__ == "__main__":
    main()
