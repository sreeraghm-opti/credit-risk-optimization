"""
Fairness and sensitive-feature audit for the credit-risk project.

Run from the project root:
    .venv/bin/python src/credit_risk/fairness_audit.py

Outputs:
    reports/fairness_group_metrics.csv
    reports/fairness_model_comparison.csv
    reports/fairness_group_rates.png

This is an educational diagnostic, not a legal compliance assessment.
Group comparisons are descriptive and do not establish causation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "credit_card_default.csv"
REPORTS = ROOT / "reports"
TARGET = "default_payment_next_month"
SENSITIVE_FEATURE = "sex"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def build_logistic_pipeline(feature_columns, categorical_columns):
    numeric_columns = [
        col for col in feature_columns if col not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def calculate_group_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y_true)),
        "observed_default_rate": float(y_true.mean()),
        "mean_predicted_pd": float(probabilities.mean()),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "average_precision": (
            float(average_precision_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
        "precision_at_threshold": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_at_threshold": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": safe_rate(fp, fp + tn),
        "false_negative_rate": safe_rate(fn, fn + tp),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check data/raw/credit_card_default.csv."
        )

    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    if SENSITIVE_FEATURE not in df.columns:
        raise ValueError(
            f"Expected sensitive-feature column '{SENSITIVE_FEATURE}' "
            f"not found. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Retain the original source-data row references for auditability.
    row_indices = pd.Series(df.index, index=df.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    categorical_base = [
        col for col in ["education", "marriage"] if col in X.columns
    ]

    feature_sets = {
        "Logistic regression (with sex)": list(X.columns),
        "Logistic regression (without sex)": [
            col for col in X.columns if col != SENSITIVE_FEATURE
        ],
    }

    probabilities_by_model = {}
    comparison_rows = []

    print("=" * 72)
    print("CREDIT RISK FAIRNESS & SENSITIVE-FEATURE AUDIT")
    print("=" * 72)
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Default rate in test set: {y_test.mean():.2%}")
    print(f"Threshold used for diagnostic rates: {THRESHOLD:.2f}")
    print(
        "Important: sex is used here only for model auditing. "
        "This is not a lending-policy recommendation."
    )
    print()

    for model_name, feature_columns in feature_sets.items():
        categorical_columns = [
            col for col in categorical_base if col in feature_columns
        ]
        if SENSITIVE_FEATURE in feature_columns:
            categorical_columns.append(SENSITIVE_FEATURE)

        model = build_logistic_pipeline(
            feature_columns, categorical_columns
        )
        model.fit(X_train[feature_columns], y_train)
        probabilities = model.predict_proba(
            X_test[feature_columns]
        )[:, 1]

        probabilities_by_model[model_name] = probabilities
        metrics = calculate_group_metrics(
            y_test, probabilities, THRESHOLD
        )
        metrics["model"] = model_name
        comparison_rows.append(metrics)

        print(
            f"{model_name}: "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}, "
            f"recall={metrics['recall_at_threshold']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}, "
            f"FNR={metrics['false_negative_rate']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = REPORTS / "fairness_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Group-level diagnostics, using the same test customers for both models.
    group_rows = []
    test_sex = X_test[SENSITIVE_FEATURE].to_numpy()

    for group_code in sorted(pd.Series(test_sex).dropna().unique()):
        mask = test_sex == group_code
        group_y = y_test.to_numpy()[mask]

        for model_name, probabilities in probabilities_by_model.items():
            group_metrics = calculate_group_metrics(
                group_y, probabilities[mask], THRESHOLD
            )
            group_metrics["model"] = model_name
            group_metrics["sex_code"] = str(group_code)
            group_rows.append(group_metrics)

    group_df = pd.DataFrame(group_rows)
    group_path = REPORTS / "fairness_group_metrics.csv"
    group_df.to_csv(group_path, index=False)

    # Plot group default rates and model positive-prediction rates.
    plot_rows = []
    for group_code in sorted(pd.Series(test_sex).dropna().unique()):
        mask = test_sex == group_code
        group_y = y_test.to_numpy()[mask]

        plot_rows.append(
            {
                "group": f"Sex code {group_code}",
                "rate_type": "Observed default rate",
                "rate": float(group_y.mean()),
            }
        )

        for model_name, probabilities in probabilities_by_model.items():
            plot_rows.append(
                {
                    "group": f"Sex code {group_code}",
                    "rate_type": model_name,
                    "rate": float((probabilities[mask] >= THRESHOLD).mean()),
                }
            )

    plot_df = pd.DataFrame(plot_rows)
    pivot = plot_df.pivot(
        index="group", columns="rate_type", values="rate"
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Dataset sex code")
    ax.set_title("Observed Default and Model Positive-Prediction Rates")
    ax.set_ylim(0, max(0.05, float(np.nanmax(pivot.to_numpy())) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = REPORTS / "fairness_group_rates.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Group-level metrics:")
    print(
        group_df[
            [
                "model",
                "sex_code",
                "n",
                "observed_default_rate",
                "mean_predicted_pd",
                "predicted_positive_rate",
                "precision_at_threshold",
                "recall_at_threshold",
                "false_positive_rate",
                "false_negative_rate",
                "brier_score",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print()
    print("Saved outputs:")
    print(f"  {comparison_path}")
    print(f"  {group_path}")
    print(f"  {plot_path}")
    print()
    print(
        "Interpretation cautions: group differences can reflect differences "
        "in observed outcomes, sample sizes, feature distributions, and model "
        "errors. Removing sex does not guarantee fairness because other "
        "features may act as proxies. These diagnostics do not establish "
        "legal compliance or justify a lending decision."
    )


if __name__ == "__main__":
    main()
