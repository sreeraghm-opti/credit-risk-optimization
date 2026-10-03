from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths and data
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "default_payment_next_month"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

categorical_features = [
    "sex",
    "education",
    "marriage",
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]


# --------------------------------------------------
# Helper: create stable bins from training data
# --------------------------------------------------

def make_bins(train_series, test_series, feature):

    if feature in categorical_features:
        train_binned = train_series.astype(str)
        test_binned = test_series.astype(str)

        return train_binned, test_binned

    # Quantile bins learned from TRAINING data only.
    try:
        _, edges = pd.qcut(
            train_series,
            q=10,
            retbins=True,
            duplicates="drop",
        )

        edges = np.unique(edges)

        if len(edges) < 3:
            return (
                train_series.astype(str),
                test_series.astype(str),
            )

        # Extend boundaries so test observations outside the
        # training range are still assigned to a bin.
        edges[0] = -np.inf
        edges[-1] = np.inf

        train_binned = pd.cut(
            train_series,
            bins=edges,
            include_lowest=True,
        ).astype(str)

        test_binned = pd.cut(
            test_series,
            bins=edges,
            include_lowest=True,
        ).astype(str)

        return train_binned, test_binned

    except ValueError:
        return (
            train_series.astype(str),
            test_series.astype(str),
        )


# --------------------------------------------------
# WoE / IV
# --------------------------------------------------

def calculate_woe_iv(train_feature, train_target, feature):

    binned, _ = make_bins(
        train_feature,
        train_feature,
        feature,
    )

    temp = pd.DataFrame(
        {
            "bin": binned,
            "target": train_target.to_numpy(),
        }
    )

    grouped = temp.groupby("bin", observed=False)["target"].agg(
        total="count",
        bad="sum",
    )

    grouped["good"] = grouped["total"] - grouped["bad"]

    # Additive smoothing avoids division by zero.
    smoothing = 0.5

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["dist_good"] = (
        (grouped["good"] + smoothing)
        / (total_good + smoothing * len(grouped))
    )

    grouped["dist_bad"] = (
        (grouped["bad"] + smoothing)
        / (total_bad + smoothing * len(grouped))
    )

    grouped["woe"] = np.log(
        grouped["dist_good"] / grouped["dist_bad"]
    )

    grouped["iv_component"] = (
        grouped["dist_good"] - grouped["dist_bad"]
    ) * grouped["woe"]

    iv = grouped["iv_component"].sum()

    grouped = grouped.reset_index()
    grouped.insert(0, "feature", feature)
    grouped["iv"] = iv

    return grouped, iv


woe_tables = []
iv_results = []

for feature in X.columns:

    table, iv = calculate_woe_iv(
        X_train[feature],
        y_train,
        feature,
    )

    woe_tables.append(table)

    iv_results.append(
        {
            "feature": feature,
            "information_value": iv,
        }
    )

woe_iv_detail = pd.concat(woe_tables, ignore_index=True)

iv_summary = (
    pd.DataFrame(iv_results)
    .sort_values(
        "information_value",
        ascending=False,
    )
)

woe_iv_detail.to_csv(
    REPORT_DIR / "woe_iv_detail.csv",
    index=False,
)

iv_summary.to_csv(
    REPORT_DIR / "information_value.csv",
    index=False,
)


# --------------------------------------------------
# PSI
# --------------------------------------------------

def calculate_psi(train_binned, test_binned):

    train_dist = train_binned.value_counts(
        normalize=True,
        dropna=False,
    )

    test_dist = test_binned.value_counts(
        normalize=True,
        dropna=False,
    )

    categories = train_dist.index.union(test_dist.index)

    train_dist = train_dist.reindex(categories, fill_value=0)
    test_dist = test_dist.reindex(categories, fill_value=0)

    # Small floor prevents log(0).
    train_dist = train_dist.clip(lower=1e-6)
    test_dist = test_dist.clip(lower=1e-6)

    return float(
        ((test_dist - train_dist)
         * np.log(test_dist / train_dist)).sum()
    )


psi_results = []

for feature in X.columns:

    train_binned, test_binned = make_bins(
        X_train[feature],
        X_test[feature],
        feature,
    )

    psi = calculate_psi(
        train_binned,
        test_binned,
    )

    psi_results.append(
        {
            "feature": feature,
            "psi": psi,
        }
    )

psi_summary = (
    pd.DataFrame(psi_results)
    .sort_values(
        "psi",
        ascending=False,
    )
)

psi_summary.to_csv(
    REPORT_DIR / "population_stability_index.csv",
    index=False,
)


# --------------------------------------------------
# PSI chart
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plot_df = psi_summary.sort_values("psi")

plt.barh(
    plot_df["feature"],
    plot_df["psi"],
)

plt.axvline(
    0.10,
    linestyle="--",
    label="0.10 reference",
)

plt.axvline(
    0.25,
    linestyle="--",
    label="0.25 reference",
)

plt.xlabel("Population Stability Index")
plt.ylabel("Feature")
plt.title("Train vs Test Population Stability")
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "population_stability_index.png",
    dpi=180,
)

plt.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 72)
print("CREDIT RISK: SCORECARD DIAGNOSTICS")
print("=" * 72)

print(f"Training observations: {len(X_train):,}")
print(f"Test observations:     {len(X_test):,}")

print("\nTOP INFORMATION VALUE FEATURES")
print("-" * 72)

print(
    iv_summary.head(15).to_string(
        index=False,
        formatters={
            "information_value": "{:.4f}".format,
        },
    )
)

print("\nPOPULATION STABILITY INDEX")
print("-" * 72)

print(
    psi_summary.to_string(
        index=False,
        formatters={
            "psi": "{:.4f}".format,
        },
    )
)

print("\nSaved:")
print(REPORT_DIR / "information_value.csv")
print(REPORT_DIR / "woe_iv_detail.csv")
print(REPORT_DIR / "population_stability_index.csv")
print(REPORT_DIR / "population_stability_index.png")

print("\nScorecard diagnostics complete.")
