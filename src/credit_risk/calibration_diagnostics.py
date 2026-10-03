from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "cross_validation_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


# --------------------------------------------------
# Load out-of-fold predictions
# --------------------------------------------------

pred = pd.read_csv(PREDICTIONS_PATH)

# Use the Random Forest OOF predictions for the main
# calibration assessment.
rf = pred[pred["model"] == "Random forest"].copy()

if rf.empty:
    raise RuntimeError("Random Forest predictions were not found.")

y = rf["actual_default"].to_numpy()
p = rf["oof_predicted_pd"].to_numpy()


# --------------------------------------------------
# Calibration intercept and slope
# --------------------------------------------------

eps = 1e-6
p_clipped = np.clip(p, eps, 1 - eps)

logit_p = np.log(p_clipped / (1 - p_clipped))

calibration_model = LogisticRegression(
    fit_intercept=True,
    solver="lbfgs",
    max_iter=2000,
)

calibration_model.fit(
    logit_p.reshape(-1, 1),
    y,
)

intercept = float(calibration_model.intercept_[0])
slope = float(calibration_model.coef_[0, 0])


# --------------------------------------------------
# PD decile calibration table
# --------------------------------------------------

calibration_df = pd.DataFrame(
    {
        "actual_default": y,
        "predicted_pd": p,
    }
)

calibration_df["pd_decile"] = pd.qcut(
    calibration_df["predicted_pd"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1

decile_table = (
    calibration_df
    .groupby("pd_decile", observed=False)
    .agg(
        observations=("actual_default", "size"),
        mean_predicted_pd=("predicted_pd", "mean"),
        observed_default_rate=("actual_default", "mean"),
    )
    .reset_index()
)

decile_table["calibration_gap"] = (
    decile_table["observed_default_rate"]
    - decile_table["mean_predicted_pd"]
)

decile_table.to_csv(
    REPORT_DIR / "pd_decile_calibration.csv",
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = pd.DataFrame(
    [
        {
            "model": "Random forest",
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
    ]
)

summary.to_csv(
    REPORT_DIR / "calibration_metrics.csv",
    index=False,
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("=" * 72)
print("CREDIT RISK: CALIBRATION DIAGNOSTICS")
print("=" * 72)

print("Model: Random forest")
print(f"Observations: {len(y):,}")
print(f"Calibration intercept: {intercept:.4f}")
print(f"Calibration slope:     {slope:.4f}")

print("\nPD DECILE CALIBRATION")
print("-" * 72)

print(
    decile_table.to_string(
        index=False,
        formatters={
            "mean_predicted_pd": "{:.4f}".format,
            "observed_default_rate": "{:.4f}".format,
            "calibration_gap": "{:.4f}".format,
        },
    )
)

print("\nSaved:")
print(REPORT_DIR / "calibration_metrics.csv")
print(REPORT_DIR / "pd_decile_calibration.csv")

print("\nCalibration diagnostics complete.")
