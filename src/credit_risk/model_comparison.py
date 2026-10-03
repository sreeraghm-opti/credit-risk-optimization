"""
Model comparison and probability calibration for the credit-risk project.

Run from the project root:
    .venv/bin/python src/credit_risk/model_comparison.py

Outputs are written to reports/:
    model_comparison_metrics.csv
    model_comparison_predictions.csv
    calibration_comparison.png
    roc_comparison.png

Important:
- This is a research exercise using historical UCI data.
- The test set is a random holdout, not out-of-time validation.
- Calibration does not make the model suitable for production lending.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
N_CALIBRATION_BINS = 10


def make_preprocessor(feature_columns, categorical_columns, scale_numeric):
    """Create preprocessing that is fitted only on training folds."""
    numeric_columns = [
        col for col in feature_columns if col not in categorical_columns
    ]

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
    )


def make_models(feature_columns, categorical_columns):
    """Return comparable logistic-regression and random-forest pipelines."""
    logistic = Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(
                    feature_columns, categorical_columns, scale_numeric=True
                ),
            ),
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

    random_forest = Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(
                    feature_columns, categorical_columns, scale_numeric=False
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=10,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic regression": logistic,
        "Logistic regression (calibrated)": CalibratedClassifierCV(
            estimator=logistic,
            method="sigmoid",
            cv=5,
        ),
        "Random forest": random_forest,
        "Random forest (calibrated)": CalibratedClassifierCV(
            estimator=random_forest,
            method="sigmoid",
            cv=5,
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Check that data/raw/credit_card_default.csv exists."
        )

    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # These are coded categorical variables in the UCI dataset.
    categorical_columns = [
        col for col in ["sex", "education", "marriage"] if col in X.columns
    ]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 72)
    print("CREDIT RISK MODEL COMPARISON & CALIBRATION")
    print("=" * 72)
    print(f"Dataset rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Training default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    print(f"Categorical features: {categorical_columns}")
    print("Test set is a random stratified holdout, not out-of-time validation.")
    print()

    models = make_models(feature_columns, categorical_columns)
    results = []
    prediction_data = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual_default": y_test.to_numpy(),
        }
    )

    probability_store = {}

    for name, model in models.items():
        print(f"Fitting: {name} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions_at_50 = (probabilities >= DEFAULT_THRESHOLD).astype(int)

        metrics = {
            "model": name,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "brier_score": brier_score_loss(y_test, probabilities),
            "precision_at_0_50": precision_score(
                y_test, predictions_at_50, zero_division=0
            ),
            "recall_at_0_50": recall_score(
                y_test, predictions_at_50, zero_division=0
            ),
            "test_default_rate": float(y_test.mean()),
            "mean_predicted_pd": float(probabilities.mean()),
            "n_test": int(len(y_test)),
        }
        results.append(metrics)
        probability_store[name] = probabilities
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        prediction_data[f"pd_{safe_name}"] = probabilities

        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f} | "
            f"AP={metrics['average_precision']:.4f} | "
            f"Brier={metrics['brier_score']:.4f} | "
            f"Recall@0.50={metrics['recall_at_0_50']:.4f}"
        )

    metrics_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )
    metrics_path = REPORTS / "model_comparison_metrics.csv"
    predictions_path = REPORTS / "model_comparison_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    prediction_data.to_csv(predictions_path, index=False)

    # Reliability diagram: points closer to the diagonal are better calibrated.
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for name, probabilities in probability_store.items():
        observed_rate, mean_predicted = calibration_curve(
            y_test,
            probabilities,
            n_bins=N_CALIBRATION_BINS,
            strategy="quantile",
        )
        ax.plot(
            mean_predicted,
            observed_rate,
            marker="o",
            linewidth=1.8,
            label=name,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Probability Calibration on the Test Set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    calibration_path = REPORTS / "calibration_comparison.png"
    fig.savefig(calibration_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ROC comparison.
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name, probabilities in probability_store.items():
        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=name,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill reference")
    ax.set_title("ROC Curve Comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    roc_path = REPORTS / "roc_comparison.png"
    fig.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Saved outputs:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {calibration_path}")
    print(f"  {roc_path}")
    print()
    print("Metrics (lower Brier score is better; higher ROC-AUC/AP is better):")
    print(
        metrics_df[
            [
                "model",
                "roc_auc",
                "average_precision",
                "brier_score",
                "precision_at_0_50",
                "recall_at_0_50",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print(
        "Interpretation caution: this is a random holdout on historical data. "
        "Use a time-based validation design before making claims about "
        "future performance or production use."
    )


if __name__ == "__main__":
    main()
