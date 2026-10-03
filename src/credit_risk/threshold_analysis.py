
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "default_payment_next_month"
X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_features = ["sex", "education", "marriage"]
numerical_features = [
    col for col in X.columns if col not in categorical_features
]


# --------------------------------------------------
# 2. Split: 60% train, 20% validation, 20% test
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("=" * 65)
print("CREDIT RISK — VALIDATION AND THRESHOLD ANALYSIS")
print("=" * 65)

for name, target in [
    ("Training", y_train),
    ("Validation", y_validation),
    ("Test", y_test),
]:
    print(
        f"{name:12s}: {len(target):,} observations | "
        f"default rate = {target.mean():.2%}"
    )


# --------------------------------------------------
# 3. Preprocessing and model
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", StandardScaler(), numerical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=2000, random_state=42),
        ),
    ]
)

model.fit(X_train, y_train)

# Use validation data for threshold selection.
validation_pd = model.predict_proba(X_validation)[:, 1]


# --------------------------------------------------
# 4. Select a threshold using validation data only
# --------------------------------------------------

TARGET_RECALL = 0.50

threshold_rows = []

for threshold in np.arange(0.05, 0.951, 0.01):
    prediction = (validation_pd >= threshold).astype(int)

    threshold_rows.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_validation, prediction, zero_division=0
            ),
            "recall": recall_score(
                y_validation, prediction, zero_division=0
            ),
            "accuracy": accuracy_score(y_validation, prediction),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

eligible = threshold_df[
    threshold_df["recall"] >= TARGET_RECALL
]

if eligible.empty:
    # Fallback: choose the threshold with the highest validation recall.
    # If several thresholds tie, choose the one with the highest precision.
    best_recall = threshold_df["recall"].max()
    eligible = threshold_df[threshold_df["recall"] == best_recall]

selected_row = eligible.sort_values(
    ["precision", "threshold"],
    ascending=[False, False],
).iloc[0]

selected_threshold = float(selected_row["threshold"])

threshold_df.to_csv(
    REPORT_DIR / "validation_threshold_analysis.csv",
    index=False,
)

print("\nTHRESHOLD SELECTION")
print("-" * 40)
print(f"Target validation recall: {TARGET_RECALL:.0%}")
print(f"Selected threshold:       {selected_threshold:.2f}")
print(f"Validation precision:     {selected_row['precision']:.4f}")
print(f"Validation recall:        {selected_row['recall']:.4f}")


# --------------------------------------------------
# 5. Evaluate once on the untouched test set
# --------------------------------------------------

test_pd = model.predict_proba(X_test)[:, 1]

test_predictions_50 = (test_pd >= 0.50).astype(int)
test_predictions_selected = (
    test_pd >= selected_threshold
).astype(int)

print("\nTEST SET — PROBABILITY METRICS")
print("-" * 40)
print(f"ROC-AUC:           {roc_auc_score(y_test, test_pd):.4f}")
print(
    f"Average precision: {average_precision_score(y_test, test_pd):.4f}"
)
print(f"Brier score:       {brier_score_loss(y_test, test_pd):.4f}")


def report_threshold(name, predictions):
    print(f"\n{name}")
    print("-" * 40)
    print(f"Threshold accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Default precision:  "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Default recall:     "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


report_threshold("TEST RESULTS — THRESHOLD 0.50", test_predictions_50)
report_threshold(
    f"TEST RESULTS — SELECTED THRESHOLD {selected_threshold:.2f}",
    test_predictions_selected,
)


# --------------------------------------------------
# 6. Save test predictions
# --------------------------------------------------

output = pd.DataFrame(
    {
        "actual_default": y_test,
        "predicted_pd": test_pd,
        "prediction_at_050": test_predictions_50,
        "prediction_at_selected_threshold": test_predictions_selected,
    },
    index=y_test.index,
)

output.to_csv(
    REPORT_DIR / "threshold_test_predictions.csv",
    index_label="row_index",
)

print("\nSaved:")
print(REPORT_DIR / "validation_threshold_analysis.csv")
print(REPORT_DIR / "threshold_test_predictions.csv")
print("\nThreshold analysis complete.")