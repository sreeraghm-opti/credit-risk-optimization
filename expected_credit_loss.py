
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Paths and assumptions
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "threshold_test_predictions.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default_payment_next_month"

# Illustrative LGD assumptions; not estimated from observed recoveries.
LGD_SCENARIOS = [0.30, 0.45, 0.60]


# --------------------------------------------------
# 2. Load data and align predictions with observations
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

# The prediction file stores original row indices.
if "row_index" not in predictions.columns:
    raise ValueError("Prediction file is missing row_index.")

if predictions["row_index"].duplicated().any():
    raise ValueError("Duplicate row indices found in predictions.")

if not predictions["row_index"].isin(df.index).all():
    raise ValueError("Some prediction indices do not match the dataset.")

test_df = (
    df.loc[predictions["row_index"]]
    .reset_index(drop=True)
)

predictions = predictions.reset_index(drop=True)

if len(test_df) != len(predictions):
    raise ValueError("Test observations and predictions are misaligned.")

# Confirm that actual outcomes match the original dataset.
if not np.array_equal(
    test_df[TARGET].to_numpy(),
    predictions["actual_default"].to_numpy(),
):
    raise ValueError("Actual outcomes do not match prediction rows.")

# Align each predicted probability with the correct test observation.
results = test_df.copy()
results["predicted_pd"] = predictions["predicted_pd"].to_numpy()


# --------------------------------------------------
# 3. Define the EAD proxy
# --------------------------------------------------

# Credit limit is NOT actual exposure at default.
# This proxy is used only for an illustrative scenario analysis.
results["ead_proxy"] = results["credit_limit"].clip(lower=0)


# --------------------------------------------------
# 4. Calculate expected loss under each LGD scenario
# --------------------------------------------------

for lgd in LGD_SCENARIOS:
    column = f"ecl_lgd_{int(lgd * 100)}pct"
    results[column] = (
        results["predicted_pd"]
        * lgd
        * results["ead_proxy"]
    )


# --------------------------------------------------
# 5. Portfolio summary
# --------------------------------------------------

portfolio_exposure = results["ead_proxy"].sum()
mean_pd = results["predicted_pd"].mean()
actual_default_rate = results[TARGET].mean()

print("=" * 65)
print("CREDIT RISK — EXPECTED CREDIT LOSS SCENARIO ANALYSIS")
print("=" * 65)

print(f"Test-set customers:       {len(results):,}")
print(f"Observed default rate:    {actual_default_rate:.2%}")
print(f"Mean predicted PD:        {mean_pd:.2%}")
print(f"Total credit-limit proxy: {portfolio_exposure:,.2f}")

summary_rows = []

for lgd in LGD_SCENARIOS:
    ecl_column = f"ecl_lgd_{int(lgd * 100)}pct"

    total_ecl = results[ecl_column].sum()
    ecl_per_customer = results[ecl_column].mean()
    exposure_weighted_loss_rate = (
        total_ecl / portfolio_exposure
        if portfolio_exposure > 0
        else np.nan
    )

    summary_rows.append(
        {
            "lgd_assumption": lgd,
            "total_ecl_proxy": total_ecl,
            "mean_ecl_per_customer": ecl_per_customer,
            "ecl_to_total_credit_limit_proxy": (
                exposure_weighted_loss_rate
            ),
        }
    )

summary = pd.DataFrame(summary_rows)

print("\nPORTFOLIO ECL SCENARIOS")
print("-" * 65)
print(summary.to_string(index=False, formatters={
    "lgd_assumption": "{:.0%}".format,
    "total_ecl_proxy": "{:,.2f}".format,
    "mean_ecl_per_customer": "{:,.2f}".format,
    "ecl_to_total_credit_limit_proxy": "{:.2%}".format,
}))


# --------------------------------------------------
# 6. Compare predicted risk with observed outcomes
# --------------------------------------------------

# This is a descriptive backtest, not an LGD validation.
# It compares predicted PD with actual default labels.
results["predicted_default_amount_proxy"] = (
    results["predicted_pd"] * results["ead_proxy"]
)

results["observed_default_amount_proxy"] = (
    results[TARGET] * results["ead_proxy"]
)

comparison = pd.DataFrame(
    {
        "measure": [
            "Predicted default-weighted exposure proxy",
            "Observed default-weighted exposure proxy",
        ],
        "amount": [
            results["predicted_default_amount_proxy"].sum(),
            results["observed_default_amount_proxy"].sum(),
        ],
    }
)

print("\nPREDICTED VS OBSERVED DEFAULT-WEIGHTED PROXY")
print("-" * 65)
print(comparison.to_string(index=False, formatters={
    "amount": "{:,.2f}".format,
}))


# --------------------------------------------------
# 7. Save outputs
# --------------------------------------------------

customer_output_path = REPORT_DIR / "customer_ecl_scenarios.csv"
summary_output_path = REPORT_DIR / "portfolio_ecl_summary.csv"

results.to_csv(customer_output_path, index=False)
summary.to_csv(summary_output_path, index=False)

print("\nSaved:")
print(customer_output_path)
print(summary_output_path)

print("\nECL scenario analysis complete.")
print(
    "Reminder: EAD is proxied by credit limit, and LGD values "
    "are assumptions. Results are illustrative, not realised "
    "or validated credit losses."
)