
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and data
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "default_payment_next_month"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 2. Define feature types
# --------------------------------------------------

categorical_features = [
    "sex",
    "education",
    "marriage",
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# This is a research baseline, not a lending decision model.
# We will revisit the use of sex and other sensitive attributes
# during the fairness analysis.


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ]
)


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("=" * 60)
print("CREDIT RISK — LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print(f"Training observations: {len(X_train):,}")
print(f"Test observations:     {len(X_test):,}")
print(f"Training default rate: {y_train.mean():.2%}")
print(f"Test default rate:     {y_test.mean():.2%}")


# --------------------------------------------------
# 5. Fit model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Predict probabilities and classes
# --------------------------------------------------

y_probability = model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)
brier = brier_score_loss(y_test, y_probability)
accuracy = accuracy_score(y_test, y_prediction)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print(f"ROC-AUC:               {roc_auc:.4f}")
print(f"Average precision:     {pr_auc:.4f}")
print(f"Brier score:           {brier:.4f}")
print(f"Accuracy at 0.50:      {accuracy:.4f}")

print("\nCONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, y_prediction))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_prediction,
        target_names=["No default", "Default"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 8. Save test predictions
# --------------------------------------------------

predictions = pd.DataFrame(
    {
        "actual_default": y_test.to_numpy(),
        "predicted_pd": y_probability,
        "predicted_default_at_50": y_prediction,
    },
    index=y_test.index,
)

predictions.to_csv(
    REPORT_DIR / "baseline_test_predictions.csv",
    index_label="row_index",
)

print("\nSaved predictions to:")
print(REPORT_DIR / "baseline_test_predictions.csv")
print("\nBaseline modelling complete.")