from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


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

print("=" * 72)
print("CREDIT RISK: MODEL EXPLAINABILITY")
print("=" * 72)

print(f"Dataset rows:       {len(df):,}")
print(f"Training rows:      {len(X_train):,}")
print(f"Test rows:          {len(X_test):,}")
print(f"Number of features: {X.shape[1]}")


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Random Forest training complete.")


# ------------------------------------------------------------
# Native feature importance
# ------------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

feature_importance.to_csv(
    REPORT_DIR / "random_forest_feature_importance.csv",
    index=False,
)

print("\nTOP RANDOM FOREST FEATURES")
print("-" * 72)

print(
    feature_importance.head(10).to_string(
        index=False,
        formatters={"importance": "{:.5f}".format},
    )
)


# ------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------

print("\nCalculating permutation importance...")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

permutation_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

permutation_df.to_csv(
    REPORT_DIR / "permutation_importance.csv",
    index=False,
)

print("\nTOP PERMUTATION FEATURES")
print("-" * 72)

print(
    permutation_df.head(10).to_string(
        index=False,
        formatters={
            "importance_mean": "{:.5f}".format,
            "importance_std": "{:.5f}".format,
        },
    )
)


# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

print("\nCalculating SHAP values...")

X_shap = X_test.sample(
    min(3000, len(X_test)),
    random_state=42,
)

explainer = shap.TreeExplainer(model)

shap_result = explainer(X_shap)

shap_values = shap_result.values

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

if shap_values.shape[1] != X_shap.shape[1]:
    raise RuntimeError(
        "SHAP output does not match the number of input features."
    )

shap_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)

shap_importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\nTOP SHAP FEATURES")
print("-" * 72)

print(
    shap_importance.head(10).to_string(
        index=False,
        formatters={
            "mean_abs_shap": "{:.5f}".format,
        },
    )
)


# ------------------------------------------------------------
# SHAP summary plot
# ------------------------------------------------------------

shap.summary_plot(
    shap_values,
    X_shap,
    show=False,
    max_display=15,
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Consolidated explainability report
# ------------------------------------------------------------

comparison = (
    feature_importance.rename(
        columns={"importance": "rf_importance"}
    )
    .merge(
        permutation_df,
        on="feature",
        how="outer",
    )
    .merge(
        shap_importance,
        on="feature",
        how="outer",
    )
)

comparison["rf_rank"] = comparison["rf_importance"].rank(
    ascending=False,
    method="min",
)

comparison["permutation_rank"] = comparison[
    "importance_mean"
].rank(
    ascending=False,
    method="min",
)

comparison["shap_rank"] = comparison[
    "mean_abs_shap"
].rank(
    ascending=False,
    method="min",
)

comparison = comparison.sort_values("shap_rank")

comparison.to_csv(
    REPORT_DIR / "explainability_summary.csv",
    index=False,
)


print("\nSaved:")
print(REPORT_DIR / "random_forest_feature_importance.csv")
print(REPORT_DIR / "permutation_importance.csv")
print(REPORT_DIR / "shap_feature_importance.csv")
print(REPORT_DIR / "shap_summary.png")
print(REPORT_DIR / "explainability_summary.csv")

print("\nExplainability analysis complete.")
