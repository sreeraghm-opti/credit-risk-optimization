from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/raw/market/nifty50_prices.csv"
DETAIL_FILE = ROOT / "reports/market/ewma_student_t_corrected.csv"
SUMMARY_FILE = ROOT / "reports/market/ewma_student_t_corrected_summary.csv"

WINDOW = 250
LAMBDA = 0.94
MIN_DF = 2.1
MAX_DF = 100.0
CONFIDENCE_LEVELS = (0.95, 0.99)


def kupiec_test(flags, confidence):
    flags = np.asarray(flags, dtype=int)
    n = len(flags)
    x = int(flags.sum())
    p0 = 1.0 - confidence
    phat = x / n

    def ll(k, total, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return k * np.log(p) + (total - k) * np.log(1 - p)

    lr = max(0.0, -2 * (ll(x, n, p0) - ll(x, n, phat)))
    return lr, float(chi2.sf(lr, 1))


def independence_test(flags):
    flags = np.asarray(flags, dtype=int)
    prev, curr = flags[:-1], flags[1:]

    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    def ll(k, total, p):
        if total == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return k * np.log(p) + (total - k) * np.log(1 - p)

    total = n00 + n01 + n10 + n11
    if total == 0:
        return np.nan, np.nan

    p_pool = (n01 + n11) / total
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    ll_null = ll(n01 + n11, total, p_pool)
    ll_markov = ll(n01, n00 + n01, p01) + ll(n11, n10 + n11, p11)

    lr = max(0.0, 2 * (ll_markov - ll_null))
    return lr, float(chi2.sf(lr, 1))


def ewma_standardized_residuals(training):
    """
    Construct historical one-step-ahead standardized residuals.

    The first 50 observations initialize variance. Each subsequent
    observation is scaled using variance calculated before that return.
    """
    training = np.asarray(training, dtype=float)
    burn_in = 50

    variance = float(np.var(training[:burn_in], ddof=1))
    residuals = []

    for j in range(burn_in, len(training)):
        sigma = np.sqrt(max(variance, 1e-16))
        residuals.append(training[j] / sigma)

        variance = (
            LAMBDA * variance
            + (1.0 - LAMBDA) * training[j] ** 2
        )

    return np.asarray(residuals)


def fit_unit_variance_student_t_df(residuals):
    """
    Fit df by MLE under a Student-t distribution standardised to
    unit variance. The Student-t scale is not independently fitted.
    """
    z = np.asarray(residuals, dtype=float)
    z = z[np.isfinite(z)]

    if len(z) < 30:
        return 10.0, False

    def negative_log_likelihood(df):
        scale = np.sqrt((df - 2.0) / df)
        # If T ~ t_df, then Z = scale*T has variance one.
        return -np.sum(
            t.logpdf(z / scale, df=df) - np.log(scale)
        )

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(MIN_DF, MAX_DF),
        method="bounded",
        options={"xatol": 1e-4},
    )

    if not result.success or not np.isfinite(result.fun):
        return 10.0, False

    fitted_df = float(result.x)
    hit_bound = (
        fitted_df <= MIN_DF + 0.01
        or fitted_df >= MAX_DF - 0.01
    )

    return fitted_df, hit_bound


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
        raise ValueError("Not enough observations for the estimation window.")

    variance = float(np.var(returns[:WINDOW], ddof=1))
    rows = []

    for i in range(WINDOW, len(returns)):
        # Update with the previous day's return before forecasting today.
        if i > WINDOW:
            variance = (
                LAMBDA * variance
                + (1.0 - LAMBDA) * returns[i - 1] ** 2
            )

        sigma = float(np.sqrt(max(variance, 1e-16)))
        training = returns[i - WINDOW:i]

        standardized_residuals = ewma_standardized_residuals(training)
        df, hit_bound = fit_unit_variance_student_t_df(
            standardized_residuals
        )

        for confidence in CONFIDENCE_LEVELS:
            normal_var = sigma * norm.ppf(confidence)

            student_quantile = (
                np.sqrt((df - 2.0) / df)
                * t.ppf(confidence, df=df)
            )
            student_var = sigma * student_quantile

            rows.append({
                "date": dates.iloc[i],
                "confidence": confidence,
                "lambda": LAMBDA,
                "student_t_df": df,
                "df_hit_bound": hit_bound,
                "forecast_volatility": sigma,
                "normal_var_return": normal_var,
                "student_t_var_return": student_var,
                "realised_return": returns[i],
                "realised_loss": -returns[i],
                "normal_exception": int(-returns[i] > normal_var),
                "student_t_exception": int(-returns[i] > student_var),
            })

    detail = pd.DataFrame(rows)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_FILE, index=False)

    summary_rows = []

    for confidence, group in detail.groupby("confidence", sort=True):
        for model, var_col, exception_col in [
            ("EWMA normal", "normal_var_return", "normal_exception"),
            ("EWMA Student-t corrected",
             "student_t_var_return", "student_t_exception"),
        ]:
            group = group.sort_values("date")
            flags = group[exception_col].to_numpy(dtype=int)

            kupiec_lr, kupiec_p = kupiec_test(flags, confidence)
            ind_lr, ind_p = independence_test(flags)
            cc_lr = kupiec_lr + ind_lr
            cc_p = float(chi2.sf(cc_lr, df=2))

            summary_rows.append({
                "method": model,
                "confidence": confidence,
                "observations": len(flags),
                "exceptions": int(flags.sum()),
                "expected_exceptions": len(flags) * (1 - confidence),
                "exception_rate": float(flags.mean()),
                "mean_var_return": float(group[var_col].mean()),
                "median_student_t_df": float(group["student_t_df"].median()),
                "mean_student_t_df": float(group["student_t_df"].mean()),
                "df_bound_hit_share": float(group["df_hit_bound"].mean()),
                "kupiec_p_value": kupiec_p,
                "independence_p_value": ind_p,
                "conditional_coverage_p_value": cc_p,
                "conditional_coverage_reject_5pct": bool(cc_p < 0.05),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("=" * 76)
    print("CORRECTED EWMA STUDENT-t — CHRONOLOGICAL VaR BACKTEST")
    print("=" * 76)
    print(f"Initial estimation window: {WINDOW} trading days")
    print(f"EWMA decay factor: {LAMBDA}")
    print(f"Backtest observations: {len(returns) - WINDOW}")
    print("Student-t df fitted to EWMA-standardised training residuals")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved detailed results:", DETAIL_FILE)
    print("Saved summary:", SUMMARY_FILE)
    print(
        "\nInterpretation caution: passing a coverage test is not proof "
        "of model adequacy. This is a one-index historical backtest, "
        "not a validated bank-wide market-risk model."
    )


if __name__ == "__main__":
    main()
