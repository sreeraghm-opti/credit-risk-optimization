"""
Cross-validation and uncertainty analysis for the credit-risk project.

Run from the project root:
    .venv/bin/python src/credit_risk/model_validation.py

Outputs:
    reports/cross_validation_metrics.csv
    reports/cross_validation_predictions.csv
    reports/cross_validation_calibration.png

This uses stratified 5-fold cross-validation on the historical dataset.
It is NOT out-of-time validation and does not establish production readiness.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "credit_card_default.csv"
REPORTS = ROOT / "reports"

TARGET = "default_payment_next_month"
CATEGORICAL_FEATURES = ["sex", "education", "marriage"]
N_SPLITS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def make_preprocessor(numeric_features):
    """Build preprocessing inside each CV fold to avoid preprocessing leakage."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def metric_values(y_true, probabilities):
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def bootstrap_intervals(y_true, probabilities, n_bootstrap, seed):
    """Percentile bootstrap intervals for metrics from out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    y_array = np.asarray(y_true)
    p_array = np.asarray(probabilities)

    values = {"roc_auc": [], "average_precision": [], "brier_score": []}
    n = len(y_array)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_sample = y_array[indices]
        p_sample = p_array[indices]

        # AUC is undefined if a bootstrap sample contains only one class.
        if np.unique(y_sample).size < 2:
            continue

        values["roc_auc"].append(roc_auc_score(y_sample, p_sample))
        values["average_precision"].append(
            average_precision_score(y_sample, p_sample)
        )
        values["brier_score"].append(brier_score_loss(y_sample, p_sample))

    intervals = {}
    for metric, samples in values.items():
        low, high = np.percentile(samples, [2.5, 97.5])
        intervals[f"{metric}_ci95_low"] = float(low)
        intervals[f"{metric}_ci95_high"] = float(high)

    return intervals


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Check that data/raw/credit_card_default.csv exists."
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET])

    categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    if set(categorical) != set(CATEGORICAL_FEATURES):
        missing = sorted(set(CATEGORICAL_FEATURES) - set(categorical))
        raise ValueError(f"Expected categorical columns are missing: {missing}")

    numeric_features = [c for c in X.columns if c not in categorical]

    # Check that the target is binary and contains both classes.
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary target values 0 and 1; found {y.unique()}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    print("=" * 72)
    print("CREDIT RISK: STRATIFIED CROSS-VALIDATION & UNCERTAINTY")
    print("=" * 72)
    print(f"Dataset rows: {len(data):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Default rate: {y.mean():.2%}")
    print(f"Cross-validation: {N_SPLITS}-fold stratified")
    print("Caution: random folds are not out-of-time validation.")
    print("Sensitive-feature note: sex is included in this model comparison;")
    print("review the separate fairness audit before interpreting its use.")
    print()

    prediction_frames = []
    metric_rows = []
    calibration_series = []

    for model_name, estimator in models.items():
        print(f"Cross-validating: {model_name} ...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features)),
                ("model", estimator),
            ]
        )

        # Every row receives a probability from a fold whose training data
        # excluded that row.
        probabilities = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        metrics = metric_values(y, probabilities)
        intervals = bootstrap_intervals(
            y,
            probabilities,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_STATE,
        )

        row = {
            "model": model_name,
            "n_rows": len(y),
            "n_folds": N_SPLITS,
            **metrics,
            **intervals,
        }
        metric_rows.append(row)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": data.index,
                    "actual_default": y.to_numpy(),
                    "model": model_name,
                    "oof_predicted_pd": probabilities,
                }
            )
        )

        fraction_positive, mean_predicted = calibration_curve(
            y,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        calibration_series.append(
            (model_name, mean_predicted, fraction_positive)
        )

        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f} "
            f"(95% CI {intervals['roc_auc_ci95_low']:.4f}–"
            f"{intervals['roc_auc_ci95_high']:.4f})"
        )
        print(
            f"  AP={metrics['average_precision']:.4f} "
            f"(95% CI {intervals['average_precision_ci95_low']:.4f}–"
            f"{intervals['average_precision_ci95_high']:.4f})"
        )
        print(
            f"  Brier={metrics['brier_score']:.4f} "
            f"(95% CI {intervals['brier_score_ci95_low']:.4f}–"
            f"{intervals['brier_score_ci95_high']:.4f})"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        "roc_auc", ascending=False
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_path = REPORTS / "cross_validation_metrics.csv"
    predictions_path = REPORTS / "cross_validation_predictions.csv"
    calibration_path = REPORTS / "cross_validation_calibration.png"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for model_name, mean_predicted, fraction_positive in calibration_series:
        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=model_name,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title(f"Calibration from {N_SPLITS}-Fold Out-of-Fold Predictions")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Saved outputs:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {calibration_path}")
    print()
    print("Summary (95% intervals are percentile bootstrap intervals):")
    print(
        metrics_df[
            [
                "model",
                "roc_auc",
                "roc_auc_ci95_low",
                "roc_auc_ci95_high",
                "average_precision",
                "average_precision_ci95_low",
                "average_precision_ci95_high",
                "brier_score",
                "brier_score_ci95_low",
                "brier_score_ci95_high",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print(
        "Interpretation: these intervals quantify resampling uncertainty in "
        "the pooled out-of-fold predictions. They are not a substitute for "
        "external or time-based validation, and they do not account for every "
        "source of model-selection uncertainty."
    )


if __name__ == "__main__":
    main()
