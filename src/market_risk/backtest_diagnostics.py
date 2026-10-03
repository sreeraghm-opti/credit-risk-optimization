from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "reports" / "market" / "var_backtest.csv"
SUMMARY_FILE = ROOT / "reports" / "market" / "var_backtest_diagnostics.csv"
EXCEPTIONS_FILE = ROOT / "reports" / "market" / "var_99_exceptions.csv"


def bernoulli_log_likelihood(successes, trials, probability):
    if trials == 0:
        return 0.0

    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    failures = trials - successes

    return (
        successes * np.log(probability)
        + failures * np.log(1 - probability)
    )


def independence_test(exception_flags):
    """Christoffersen first-order Markov independence test."""
    flags = np.asarray(exception_flags, dtype=int)

    if len(flags) < 2:
        return np.nan, np.nan

    previous = flags[:-1]
    current = flags[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pooled = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_independent = (
        bernoulli_log_likelihood(n01 + n11, n00 + n01 + n10 + n11, pooled)
    )
    ll_markov = (
        bernoulli_log_likelihood(n01, n00 + n01, p01)
        + bernoulli_log_likelihood(n11, n10 + n11, p11)
    )

    statistic = max(0.0, 2 * (ll_markov - ll_independent))
    return float(statistic), float(chi2.sf(statistic, df=1))


def main():
    data = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    summary_rows = []
    exception_rows = []

    for (method, confidence), group in data.groupby(
        ["method", "confidence"], sort=False
    ):
        group = group.sort_values("date").reset_index(drop=True)
        flags = group["exception"].to_numpy(dtype=int)

        kupiec_p = pd.read_csv(
            ROOT / "reports" / "market" / "var_backtest_summary.csv"
        )
        match = kupiec_p[
            (kupiec_p["method"] == method)
            & np.isclose(kupiec_p["confidence"], confidence)
        ]

        coverage_p = float(match["kupiec_p_value"].iloc[0])
        ind_stat, ind_p = independence_test(flags)

        # Conditional coverage LR is the sum of the two LR statistics.
        kupiec_lr = float(match["kupiec_lr_stat"].iloc[0])
        cc_lr = kupiec_lr + ind_stat if np.isfinite(ind_stat) else np.nan
        cc_p = float(chi2.sf(cc_lr, df=2)) if np.isfinite(cc_lr) else np.nan

        summary_rows.append({
            "method": method,
            "confidence": confidence,
            "observations": len(group),
            "exceptions": int(flags.sum()),
            "exception_rate": float(flags.mean()),
            "kupiec_p_value": coverage_p,
            "independence_lr_stat": ind_stat,
            "independence_p_value": ind_p,
            "conditional_coverage_lr_stat": cc_lr,
            "conditional_coverage_p_value": cc_p,
            "conditional_coverage_reject_5pct": (
                bool(cc_p < 0.05) if np.isfinite(cc_p) else None
            ),
        })

        if np.isclose(confidence, 0.99):
            exceptions = group[group["exception"] == 1].copy()
            exceptions["method"] = method
            exception_rows.append(exceptions)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    if exception_rows:
        pd.concat(exception_rows, ignore_index=True).to_csv(
            EXCEPTIONS_FILE, index=False
        )

    print("=" * 72)
    print("VaR EXCEPTION INDEPENDENCE & CONDITIONAL COVERAGE")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved diagnostics:", SUMMARY_FILE)
    print("Saved 99% exception dates:", EXCEPTIONS_FILE)
    print(
        "\nInterpretation: rejection indicates a statistical model failure "
        "under the test assumptions. Non-rejection does not prove adequacy."
    )


if __name__ == "__main__":
    main()
