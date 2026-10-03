
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# MACROECONOMIC CREDIT RISK STRESS TESTING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

PREDICTIONS_FILE = REPORTS / "threshold_test_predictions.csv"
RAW_DATA_FILE = ROOT / "data" / "raw" / "credit_card_default.csv"

LGD = 0.45

# Illustrative stress assumptions, not empirically estimated.
STRESS_SCENARIOS = {
    "Baseline": 1.00,
    "Adverse": 1.50,
    "Severe adverse": 2.00,
}


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

predictions = pd.read_csv(PREDICTIONS_FILE)
raw_data = pd.read_csv(RAW_DATA_FILE)

required_columns = {"row_index", "actual_default", "predicted_pd"}
missing = required_columns - set(predictions.columns)

if missing:
    raise ValueError(f"Missing prediction columns: {sorted(missing)}")

predictions["row_index"] = predictions["row_index"].astype(int)

if predictions["row_index"].duplicated().any():
    raise ValueError("Duplicate row_index values found.")

if not predictions["row_index"].isin(raw_data.index).all():
    raise ValueError("Some prediction rows do not match the raw dataset.")

predictions = predictions.set_index("row_index")

# Align exposures using the original dataset row indices.
predictions["credit_limit"] = raw_data.loc[
    predictions.index, "credit_limit"
]

predictions["ead_proxy"] = predictions["credit_limit"].clip(lower=0)

pd_values = predictions["predicted_pd"].to_numpy(dtype=float)

if not np.isfinite(pd_values).all():
    raise ValueError("Predicted PD contains non-finite values.")

if ((pd_values < 0) | (pd_values > 1)).any():
    raise ValueError("Predicted PD values must lie between 0 and 1.")

if not 0 <= LGD <= 1:
    raise ValueError("LGD must lie between 0 and 1.")


# ============================================================
# 2. CALCULATE STRESSED EXPECTED LOSS PROXIES
# ============================================================

scenario_rows = []
customer_results = predictions[
    ["actual_default", "predicted_pd", "ead_proxy"]
].copy()

for scenario, multiplier in STRESS_SCENARIOS.items():

    # Apply a transparent PD multiplier and cap probabilities at 100%.
    stressed_pd = np.clip(pd_values * multiplier, 0, 1)

    # Expected loss proxy = stressed PD × LGD × exposure proxy.
    customer_ecl = stressed_pd * LGD * customer_results["ead_proxy"].to_numpy()

    customer_results[f"{scenario}_pd"] = stressed_pd
    customer_results[f"{scenario}_ecl_proxy"] = customer_ecl

    total_ecl = customer_ecl.sum()
    total_exposure = customer_results["ead_proxy"].sum()

    scenario_rows.append({
        "scenario": scenario,
        "pd_multiplier": multiplier,
        "mean_stressed_pd": stressed_pd.mean(),
        "total_ecl_proxy": total_ecl,
        "mean_ecl_per_customer": customer_ecl.mean(),
        "ecl_to_exposure_proxy": (
            total_ecl / total_exposure if total_exposure > 0 else np.nan
        ),
        "change_vs_baseline_pct": np.nan,
    })


# ============================================================
# 3. COMPARE SCENARIOS
# ============================================================

summary = pd.DataFrame(scenario_rows)

baseline_ecl = summary.loc[
    summary["scenario"] == "Baseline", "total_ecl_proxy"
].iloc[0]

if baseline_ecl > 0:
    summary["change_vs_baseline_pct"] = (
        (summary["total_ecl_proxy"] / baseline_ecl) - 1
    ) * 100

print("=" * 76)
print("MACROECONOMIC CREDIT RISK STRESS TEST")
print("=" * 76)
print(f"Test-set customers:        {len(predictions):,}")
print(f"LGD assumption:             {LGD:.0%}")
print(f"Total exposure proxy:      {customer_results['ead_proxy'].sum():,.2f}")
print("Stress multipliers:         illustrative, not estimated")
print()
print(
    summary.to_string(
        index=False,
        formatters={
            "mean_stressed_pd": "{:.2%}".format,
            "total_ecl_proxy": "{:,.2f}".format,
            "mean_ecl_per_customer": "{:,.2f}".format,
            "ecl_to_exposure_proxy": "{:.2%}".format,
            "change_vs_baseline_pct": "{:+.2f}%".format,
        },
    )
)


# ============================================================
# 4. IDENTIFY CUSTOMERS CONTRIBUTING MOST TO SEVERE STRESS
# ============================================================

severe_column = "Severe adverse_ecl_proxy"

top_customers = customer_results.nlargest(
    20, severe_column
).copy()

top_customers.insert(
    0, "row_index", top_customers.index
)

top_customers = top_customers[
    [
        "row_index",
        "actual_default",
        "predicted_pd",
        "Severe adverse_pd",
        "ead_proxy",
        "Severe adverse_ecl_proxy",
    ]
]


# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

summary.to_csv(
    REPORTS / "stress_testing_summary.csv",
    index=False,
)

customer_results.to_csv(
    REPORTS / "customer_stress_results.csv",
    index_label="row_index",
)

top_customers.to_csv(
    REPORTS / "top_stressed_customers.csv",
    index=False,
)


# ============================================================
# 6. VISUALISE STRESS SCENARIOS
# ============================================================

plt.figure(figsize=(9, 5))

bars = plt.bar(
    summary["scenario"],
    summary["total_ecl_proxy"] / 1_000_000,
)

plt.ylabel("Expected loss proxy (millions of dataset currency units)")
plt.xlabel("Stress scenario")
plt.title("Portfolio Expected Loss Under PD Stress Scenarios")

for bar, value in zip(bars, summary["total_ecl_proxy"] / 1_000_000):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(
    REPORTS / "stress_testing_comparison.png",
    dpi=160,
)
plt.close()


print("\nSaved:")
print(REPORTS / "stress_testing_summary.csv")
print(REPORTS / "customer_stress_results.csv")
print(REPORTS / "top_stressed_customers.csv")
print(REPORTS / "stress_testing_comparison.png")

print("\nStress testing complete.")
print(
    "Caution: these are illustrative expected-loss proxies, "
    "not forecasts of actual bank losses."
)