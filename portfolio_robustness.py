
"""
Portfolio Robustness & Sensitivity Analysis
===========================================

Run from the project root:
    .venv/bin/python portfolio_robustness.py

Tests the sensitivity of constrained portfolio selection to:
- LGD assumptions
- Expected-loss budgets
- Exposure limits

Uses out-of-fold random-forest PD estimates.
Actual default outcomes are used only for retrospective diagnostics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "credit_card_default.csv"
PREDICTIONS_PATH = ROOT / "reports" / "cross_validation_predictions.csv"
REPORTS = ROOT / "reports"

TARGET = "default_payment_next_month"
MODEL_NAME = "Random forest"

# Keep consistent with portfolio_optimization.py
MAX_SELECTED_ACCOUNTS = 1_000
EAD_PROXY_CAP = 600_000

# Baseline assumptions
BASELINE_LGD = 0.45
BASELINE_EXPOSURE_LIMIT = 100_000_000
BASELINE_LOSS_BUDGET = 5_000_000

# One-at-a-time sensitivity scenarios
SCENARIOS = [
    {
        "scenario": "Baseline",
        "lgd": 0.45,
        "exposure_limit": 100_000_000,
        "loss_budget": 5_000_000,
    },
    {
        "scenario": "LGD 30%",
        "lgd": 0.30,
        "exposure_limit": 100_000_000,
        "loss_budget": 5_000_000,
    },
    {
        "scenario": "LGD 60%",
        "lgd": 0.60,
        "exposure_limit": 100_000_000,
        "loss_budget": 5_000_000,
    },
    {
        "scenario": "Loss budget 4m",
        "lgd": 0.45,
        "exposure_limit": 100_000_000,
        "loss_budget": 4_000_000,
    },
    {
        "scenario": "Loss budget 6m",
        "lgd": 0.45,
        "exposure_limit": 100_000_000,
        "loss_budget": 6_000_000,
    },
    {
        "scenario": "Exposure limit 80m",
        "lgd": 0.45,
        "exposure_limit": 80_000_000,
        "loss_budget": 5_000_000,
    },
    {
        "scenario": "Exposure limit 120m",
        "lgd": 0.45,
        "exposure_limit": 120_000_000,
        "loss_budget": 5_000_000,
    },
]


def load_candidates():
    """Load OOF predictions and construct the eligible candidate pool."""

    for path in (DATA_PATH, PREDICTIONS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run model_validation.py first."
            )

    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_data = {TARGET, "credit_limit"}
    required_predictions = {
        "row_index",
        "model",
        "oof_predicted_pd",
    }

    missing_data = required_data - set(data.columns)
    missing_predictions = required_predictions - set(predictions.columns)

    if missing_data:
        raise ValueError(f"Missing dataset columns: {sorted(missing_data)}")

    if missing_predictions:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_predictions)}"
        )

    predictions = predictions.loc[
        predictions["model"].eq(MODEL_NAME)
    ].copy()

    if predictions.empty:
        raise ValueError(
            f"No out-of-fold predictions found for {MODEL_NAME!r}."
        )

    data = data.copy()
    data["row_index"] = data.index

    candidates = predictions.merge(
        data[["row_index", "credit_limit", TARGET]],
        on="row_index",
        how="inner",
        validate="one_to_one",
    )

    if candidates.empty:
        raise ValueError("No matching rows between data and predictions.")

    # Exclude exact duplicate source records, consistent with the
    # existing optimisation script. No customer ID is available.
    duplicate_mask = data.drop(columns=["row_index"]).duplicated(
        keep="first"
    )
    unique_indices = set(data.loc[~duplicate_mask, "row_index"])

    before_dedup = len(candidates)
    candidates = candidates.loc[
        candidates["row_index"].isin(unique_indices)
    ].copy()

    duplicates_excluded = before_dedup - len(candidates)

    candidates["pd_estimate"] = pd.to_numeric(
        candidates["oof_predicted_pd"], errors="coerce"
    ).clip(0.0, 1.0)

    candidates["ead_proxy"] = pd.to_numeric(
        candidates["credit_limit"], errors="coerce"
    ).fillna(0).clip(lower=0, upper=EAD_PROXY_CAP)

    candidates = candidates.dropna(subset=["pd_estimate"])
    candidates = candidates.loc[
        candidates["ead_proxy"] > 0
    ].reset_index(drop=True)

    if candidates.empty:
        raise ValueError("No eligible positive-exposure candidates remain.")

    return candidates, duplicates_excluded


def optimise_portfolio(candidates, lgd, exposure_limit, loss_budget):
    """Maximise selected exposure under the scenario constraints."""

    ead = candidates["ead_proxy"].to_numpy(dtype=float)
    pd_estimate = candidates["pd_estimate"].to_numpy(dtype=float)
    expected_loss = pd_estimate * lgd * ead

    n = len(candidates)

    # Binary decision variable: 1 = select account, 0 = exclude.
    # Negating exposure converts maximisation into minimisation.
    objective = -ead

    matrix = np.vstack([
        ead,
        expected_loss,
        np.ones(n, dtype=float),
    ])

    constraints = LinearConstraint(
        matrix,
        lb=np.array([-np.inf, -np.inf, -np.inf]),
        ub=np.array([
            exposure_limit,
            loss_budget,
            MAX_SELECTED_ACCOUNTS,
        ], dtype=float),
    )

    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": 120},
    )

    if result.x is None:
        raise RuntimeError(
            f"Optimisation failed for scenario with LGD={lgd:.0%}, "
            f"exposure limit={exposure_limit:,.0f}, "
            f"loss budget={loss_budget:,.0f}. "
            f"Solver message: {result.message}"
        )

    selected_mask = result.x > 0.5
    selected = candidates.loc[selected_mask].copy()

    if selected.empty:
        raise RuntimeError("The optimiser returned an empty portfolio.")

    selected_ead = selected["ead_proxy"].to_numpy(dtype=float)
    selected_pd = selected["pd_estimate"].to_numpy(dtype=float)

    total_exposure = float(selected_ead.sum())
    total_expected_loss = float(
        np.sum(selected_pd * lgd * selected_ead)
    )

    weighted_mean_pd = float(
        np.average(selected_pd, weights=selected_ead)
    )

    # Retrospective diagnostic only. This is not an optimisation input.
    observed_default_rate = float(selected[TARGET].mean())

    return {
        "selected": selected,
        "solver_status": result.message,
        "selected_accounts": len(selected),
        "selected_exposure_proxy": total_exposure,
        "expected_loss_proxy": total_expected_loss,
        "exposure_weighted_mean_pd": weighted_mean_pd,
        "expected_loss_to_selected_exposure": (
            total_expected_loss / total_exposure
            if total_exposure > 0 else np.nan
        ),
        "retrospective_observed_default_rate": observed_default_rate,
        "exposure_limit_utilisation": total_exposure / exposure_limit,
        "loss_budget_utilisation": total_expected_loss / loss_budget,
        "account_limit_utilisation": (
            len(selected) / MAX_SELECTED_ACCOUNTS
        ),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    candidates, duplicates_excluded = load_candidates()

    print("=" * 72)
    print("CREDIT PORTFOLIO ROBUSTNESS & SENSITIVITY ANALYSIS")
    print("=" * 72)
    print(f"Model: {MODEL_NAME} out-of-fold predictions")
    print(f"Eligible candidate rows: {len(candidates):,}")
    print(f"Exact duplicate rows excluded: {duplicates_excluded:,}")
    print(f"Scenarios to solve: {len(SCENARIOS)}")
    print("\nSolving each scenario...\n")

    results = []
    selected_sets = {}

    for scenario in SCENARIOS:
        outcome = optimise_portfolio(
            candidates=candidates,
            lgd=scenario["lgd"],
            exposure_limit=scenario["exposure_limit"],
            loss_budget=scenario["loss_budget"],
        )

        selected = outcome.pop("selected")
        selected_sets[scenario["scenario"]] = set(
            selected["row_index"].astype(int)
        )

        row = {
            "scenario": scenario["scenario"],
            "lgd_assumption": scenario["lgd"],
            "exposure_limit": scenario["exposure_limit"],
            "expected_loss_budget": scenario["loss_budget"],
            **outcome,
        }
        results.append(row)

        print(
            f"{scenario['scenario']:<24} "
            f"Accounts={row['selected_accounts']:>4,} | "
            f"Exposure={row['selected_exposure_proxy']:>14,.0f} | "
            f"Expected loss={row['expected_loss_proxy']:>12,.0f} | "
            f"Observed default rate="
            f"{row['retrospective_observed_default_rate']:.2%}"
        )

    summary = pd.DataFrame(results)

    # Compare every portfolio with the baseline portfolio.
    baseline_set = selected_sets["Baseline"]

    def jaccard_similarity(selected_set):
        union = baseline_set | selected_set
        if not union:
            return np.nan
        return len(baseline_set & selected_set) / len(union)

    summary["baseline_overlap_accounts"] = summary["scenario"].map(
        lambda name: len(baseline_set & selected_sets[name])
    )
    summary["baseline_jaccard_similarity"] = summary["scenario"].map(
        lambda name: jaccard_similarity(selected_sets[name])
    )

    # Change in expected loss relative to the baseline result.
    baseline_loss = float(
        summary.loc[
            summary["scenario"].eq("Baseline"),
            "expected_loss_proxy",
        ].iloc[0]
    )

    summary["expected_loss_change_vs_baseline"] = (
        summary["expected_loss_proxy"] - baseline_loss
    )

    summary["expected_loss_change_vs_baseline_pct"] = (
        summary["expected_loss_change_vs_baseline"] / baseline_loss
        if baseline_loss else np.nan
    )

    # Save scenario summary.
    summary_path = REPORTS / "portfolio_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Save account selections for auditability and further analysis.
    selection_rows = []

    for scenario in SCENARIOS:
        scenario_name = scenario["scenario"]
        chosen_indices = selected_sets[scenario_name]

        chosen = candidates.loc[
            candidates["row_index"].isin(chosen_indices)
        ].copy()

        chosen["scenario"] = scenario_name
        chosen["lgd_assumption"] = scenario["lgd"]
        chosen["expected_loss_proxy"] = (
            chosen["pd_estimate"]
            * scenario["lgd"]
            * chosen["ead_proxy"]
        )

        selection_rows.append(
            chosen[
                [
                    "scenario",
                    "row_index",
                    "pd_estimate",
                    "ead_proxy",
                    "expected_loss_proxy",
                    TARGET,
                    "lgd_assumption",
                ]
            ]
        )

    selections = pd.concat(selection_rows, ignore_index=True)
    selections = selections.rename(
        columns={TARGET: "observed_default"}
    )

    selections_path = REPORTS / "portfolio_robustness_selections.csv"
    selections.to_csv(selections_path, index=False)

    print("\n" + "=" * 72)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Summary: {summary_path}")
    print(f"Account selections: {selections_path}")
    print("\nBaseline portfolio overlap:")
    print(
        summary[
            [
                "scenario",
                "baseline_overlap_accounts",
                "baseline_jaccard_similarity",
            ]
        ].to_string(index=False)
    )

    print(
        "\nCaution: retrospective observed default rates are diagnostics, "
        "not future performance estimates. Scenario assumptions are "
        "illustrative, and random/out-of-fold validation is not "
        "out-of-time validation."
    )


if __name__ == "__main__":
    main()