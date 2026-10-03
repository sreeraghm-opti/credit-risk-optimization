from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "market"
REPORTS.mkdir(parents=True, exist_ok=True)

FILES = {
    "backtest": REPORTS / "var_backtest.csv",
    "ewma": REPORTS / "ewma_backtest.csv",
    "student": REPORTS / "ewma_student_t_backtest.csv",
    "corrected": REPORTS / "ewma_student_t_corrected.csv",
}


def read_csv_checked(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the relevant model backtest first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def canonicalise(df, model, confidence_col="confidence",
                 var_col="var_return", exception_col="exception"):
    out = df.copy()
    out["model"] = model
    out["confidence"] = out[confidence_col].astype(float)
    out["var_return"] = out[var_col].astype(float)
    out["exception"] = out[exception_col].astype(int)
    return out[[
        "date", "model", "confidence", "var_return", "exception"
    ]]


def kupiec(flags, confidence):
    flags = np.asarray(flags, dtype=int)
    n = len(flags)
    x = int(flags.sum())
    p0 = 1.0 - confidence
    phat = x / n

    def ll(k, total, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return k * np.log(p) + (total - k) * np.log(1 - p)

    lr = max(0.0, -2 * (ll(x, n, p0) - ll(x, n, phat)))
    return float(lr), float(chi2.sf(lr, 1))


def independence(flags):
    flags = np.asarray(flags, dtype=int)
    prev, curr = flags[:-1], flags[1:]

    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total = n00 + n01 + n10 + n11

    if total == 0:
        return np.nan, np.nan

    def ll(k, trials, p):
        if trials == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return k * np.log(p) + (trials - k) * np.log(1 - p)

    pooled = (n01 + n11) / total
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    ll0 = ll(n01 + n11, total, pooled)
    ll1 = ll(n01, n00 + n01, p01) + ll(n11, n10 + n11, p11)
    lr = max(0.0, 2 * (ll1 - ll0))
    return float(lr), float(chi2.sf(lr, 1))


# 1. Historical and parametric normal models
base = read_csv_checked(FILES["backtest"])
base["method"] = base["method"].astype(str)

historical = base[base["method"].str.lower().str.contains("historical")]
parametric = base[
    base["method"].str.lower().str.contains("parametric|normal", regex=True)
]

if historical.empty or parametric.empty:
    raise ValueError(
        "Could not identify historical and parametric models in "
        "var_backtest.csv. Inspect its 'method' column."
    )

historical = canonicalise(historical, "Historical VaR")
parametric = canonicalise(parametric, "Parametric normal VaR")

# 2. EWMA normal
ewma_df = read_csv_checked(FILES["ewma"])
ewma = canonicalise(ewma_df, "EWMA normal")

# 3. Original EWMA Student-t
student_df = read_csv_checked(FILES["student"])
student = canonicalise(student_df, "EWMA Student-t (original)")

# 4. Corrected EWMA Student-t
corrected_df = read_csv_checked(FILES["corrected"])
corrected = canonicalise(
    corrected_df,
    "EWMA Student-t (corrected)",
    var_col="student_t_var_return",
    exception_col="student_t_exception",
)

# Combine the five models.
all_models = pd.concat(
    [historical, parametric, ewma, student, corrected],
    ignore_index=True,
)

# Retain the common evaluation dates for a fair comparison.
date_sets = [
    set(g["date"].unique())
    for _, g in all_models.groupby(["model", "confidence"])
]
common_dates = set.intersection(*date_sets)

all_models = all_models[all_models["date"].isin(common_dates)].copy()
all_models = all_models.sort_values(
    ["confidence", "model", "date"]
).reset_index(drop=True)

if all_models.empty:
    raise ValueError("No common dates across all models.")

# 5. Summary statistics and coverage tests.
summary_rows = []

for (model, confidence), group in all_models.groupby(
    ["model", "confidence"], sort=True
):
    group = group.sort_values("date")
    flags = group["exception"].to_numpy(dtype=int)
    n = len(flags)
    count = int(flags.sum())

    kupiec_lr, kupiec_p = kupiec(flags, confidence)
    ind_lr, ind_p = independence(flags)
    cc_lr = kupiec_lr + ind_lr
    cc_p = float(chi2.sf(cc_lr, 2))

    summary_rows.append({
        "model": model,
        "confidence": confidence,
        "observations": n,
        "exceptions": count,
        "expected_exceptions": n * (1 - confidence),
        "exception_rate": count / n,
        "mean_var_return": group["var_return"].mean(),
        "median_var_return": group["var_return"].median(),
        "kupiec_lr": kupiec_lr,
        "kupiec_p_value": kupiec_p,
        "independence_lr": ind_lr,
        "independence_p_value": ind_p,
        "conditional_coverage_lr": cc_lr,
        "conditional_coverage_p_value": cc_p,
        "conditional_coverage_reject_5pct": cc_p < 0.05,
    })

summary = pd.DataFrame(summary_rows)
summary_path = REPORTS / "var_model_comparison_summary.csv"
detail_path = REPORTS / "var_model_comparison_daily.csv"
markdown_path = REPORTS / "var_model_comparison_report.md"
chart_path = REPORTS / "var_model_comparison.png"

summary.to_csv(summary_path, index=False)
all_models.to_csv(detail_path, index=False)

# 6. Comparison chart: exception rate against the target rate.
for confidence in [0.95, 0.99]:
    subset = summary[summary["confidence"] == confidence].copy()
    if subset.empty:
        continue

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(subset))
    ax.bar(x, subset["exception_rate"] * 100)
    ax.axhline(
        (1 - confidence) * 100,
        linestyle="--",
        label="Target exception rate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"], rotation=25, ha="right")
    ax.set_ylabel("Exception rate (%)")
    ax.set_title(f"VaR Backtest Comparison — {confidence:.0%} Confidence")
    ax.legend()
    fig.tight_layout()

    suffix = str(int(confidence * 100))
    fig.savefig(
        REPORTS / f"var_model_comparison_{suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

# 7. Human-readable report.
lines = [
    "# VaR Model Comparison",
    "",
    "## Methodology",
    "",
    f"- Common evaluation dates: {len(common_dates)}",
    "- Models: historical, parametric normal, EWMA normal, "
    "original EWMA Student-t, corrected EWMA Student-t.",
    "- Exception: realised loss strictly exceeds the forecast VaR.",
    "- Kupiec tests unconditional coverage; independence tests "
    "first-order exception dependence; conditional coverage combines them.",
    "- P-values below 0.05 indicate rejection at the 5% level.",
    "",
    "These are historical NIFTY 50 index backtests, not a regulatory "
    "validation of a bank trading-book model.",
    "",
    "## Results",
    "",
]

for confidence in [0.95, 0.99]:
    lines.extend([
        f"### {confidence:.0%} confidence",
        "",
        "| Model | Exceptions | Expected | Exception rate | Mean VaR | Kupiec p | Independence p | Conditional coverage p | Reject at 5%? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])

    subset = summary[summary["confidence"] == confidence]
    for _, r in subset.iterrows():
        lines.append(
            f"| {r['model']} "
            f"| {int(r['exceptions'])} "
            f"| {r['expected_exceptions']:.2f} "
            f"| {r['exception_rate']:.2%} "
            f"| {r['mean_var_return']:.2%} "
            f"| {r['kupiec_p_value']:.4f} "
            f"| {r['independence_p_value']:.4f} "
            f"| {r['conditional_coverage_p_value']:.4f} "
            f"| {'Yes' if r['conditional_coverage_reject_5pct'] else 'No'} |"
        )
    lines.append("")

lines.extend([
    "## Interpretation",
    "",
    "A lower exception rate is not automatically better: the objective is "
    "calibration to the intended exception probability, alongside adequate "
    "independence and risk sensitivity. Failure to reject a test is not "
    "proof that a model is correct. The 99% backtest has relatively few "
    "expected exceptions, so test power is limited.",
    "",
    "Mean VaR is shown as a percentage of index value, not as a portfolio "
    "loss estimate. Index returns exclude dividends and do not include "
    "transaction costs.",
    "",
])

markdown_path.write_text("\n".join(lines), encoding="utf-8")

print("=" * 88)
print("CONSOLIDATED VaR MODEL COMPARISON")
print("=" * 88)
print("Common evaluation dates:", len(common_dates))
print("\nSummary:")
print(summary.to_string(
    index=False, float_format=lambda x: f"{x:.4f}"
))
print("\nSaved files:")
print(summary_path)
print(detail_path)
print(markdown_path)
print(REPORTS / "var_model_comparison_95.png")
print(REPORTS / "var_model_comparison_99.png")
print("\nDone. Existing model scripts and result files were not modified.")
