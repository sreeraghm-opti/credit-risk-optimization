from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# Credit Risk & Portfolio Analytics Engine
# Educational research prototype; not a production credit model.
# ============================================================

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Credit Risk Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Credit Risk & Portfolio Analytics Engine")
st.caption(
    "Probability of default · Expected credit loss · Portfolio tail risk · "
    "Stress testing · Model validation · Fairness diagnostics"
)

st.warning(
    "Research and educational prototype. The dataset is historical Taiwanese "
    "credit-card data. Credit limit is used as an exposure proxy; LGD and "
    "Monte Carlo correlation are assumptions. Results are not validated bank "
    "loss estimates, regulatory capital figures, IFRS 9 provisions, or a "
    "basis for real lending decisions."
)


# ---------------------------- Utilities ----------------------------

def load_report(filename):
    """Load a CSV report, returning None when it has not been generated."""
    path = REPORTS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


def show_image(filename, caption=None):
    """Display a saved report image if available."""
    path = REPORTS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: reports/{filename}")


def format_rate(value):
    return f"{value:.2%}" if pd.notna(value) else "—"


def format_number(value):
    return f"{value:,.2f}" if pd.notna(value) else "—"


def style_model_metrics(df):
    """Prepare model-comparison metrics for readable display."""
    out = df.copy()
    percent_cols = [
        col for col in ["precision_at_0_50", "recall_at_0_50"]
        if col in out.columns
    ]
    for col in percent_cols:
        out[col] = out[col].map(format_rate)

    for col in ["roc_auc", "average_precision", "brier_score"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
    return out


def normalize_lgd_percent(series):
    """
    Handle LGD values stored as '45%', 45, or 0.45.
    Return numeric percentage values.
    """
    numeric = pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )
    if numeric.dropna().empty:
        return numeric
    if numeric.dropna().max() <= 1:
        numeric = numeric * 100
    return numeric


def metric_column(df, possible_names):
    """Return the first matching column name, case-insensitively."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in possible_names:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


# ---------------------------- Sidebar ----------------------------

st.sidebar.header("Dashboard controls")
lgd_percent = st.sidebar.slider(
    "LGD assumption (%)",
    min_value=10,
    max_value=90,
    value=45,
    step=5,
)
lgd = lgd_percent / 100

st.sidebar.caption(
    "The selected LGD rescales the ECL proxy. Other model and simulation "
    "assumptions remain unchanged."
)

st.sidebar.divider()
st.sidebar.caption("Project status")
required_reports = {
    "Predictions": "threshold_test_predictions.csv",
    "ECL summary": "portfolio_ecl_summary.csv",
    "Monte Carlo": "monte_carlo_risk_summary.csv",
    "Stress testing": "stress_testing_summary.csv",
    "Fairness audit": "fairness_group_metrics.csv",
    "Model comparison": "model_comparison_metrics.csv",
}
for label, filename in required_reports.items():
    status = "Ready" if (REPORTS / filename).exists() else "Not generated"
    st.sidebar.write(f"{'🟢' if status == 'Ready' else '⚪'} {label}: {status}")


# ---------------------------- Load reports ----------------------------

predictions = load_report("threshold_test_predictions.csv")
ecl = load_report("portfolio_ecl_summary.csv")
monte_carlo = load_report("monte_carlo_risk_summary.csv")
stress = load_report("stress_testing_summary.csv")
top_customers = load_report("top_stressed_customers.csv")
model_metrics = load_report("model_comparison_metrics.csv")
fairness = load_report("fairness_group_metrics.csv")
fairness_comparison = load_report("fairness_model_comparison.csv")


if predictions is None:
    st.error(
        "Missing reports/threshold_test_predictions.csv. "
        "Run threshold_analysis.py first."
    )
    st.stop()

required_prediction_cols = {"actual_default", "predicted_pd"}
missing_cols = required_prediction_cols - set(predictions.columns)
if missing_cols:
    st.error(
        "The prediction report is missing required columns: "
        + ", ".join(sorted(missing_cols))
    )
    st.stop()


# ---------------------------- Overview ----------------------------

st.header("Portfolio overview")

n_customers = len(predictions)
default_rate = predictions["actual_default"].mean()
mean_pd = predictions["predicted_pd"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Test-set customers", f"{n_customers:,}")
col2.metric("Observed default rate", format_rate(default_rate))
col3.metric("Mean predicted PD", format_rate(mean_pd))

st.caption(
    "Evaluation uses a random stratified holdout from historical data, "
    "not out-of-time validation."
)
st.divider()


# ---------------------------- Model comparison ----------------------------

st.header("1. Model comparison and validation")

if model_metrics is not None:
    st.markdown(
        "Compare discrimination, ranking quality, probability error and "
        "default recall at a 0.50 classification threshold."
    )
    st.dataframe(
        style_model_metrics(model_metrics),
        use_container_width=True,
        hide_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        show_image(
            "roc_comparison.png",
            "ROC curves: discrimination across classification thresholds",
        )
    with c2:
        show_image(
            "calibration_comparison.png",
            "Calibration curves: predicted probabilities versus observed frequencies",
        )
else:
    st.info(
        "Model comparison report not found. Run "
        "`.venv/bin/python model_comparison.py`."
    )

with st.expander("How to interpret these metrics"):
    st.markdown(
        "- **ROC-AUC:** ability to rank defaulters above non-defaulters.\n"
        "- **Average precision:** precision-recall performance, useful when "
        "defaults are the minority class.\n"
        "- **Brier score:** mean squared error of predicted probabilities; "
        "lower is better.\n"
        "- **Recall at 0.50:** fraction of observed defaults classified as "
        "positive at a 0.50 threshold.\n\n"
        "No single metric establishes that a model is suitable for lending. "
        "Thresholds should be chosen using validation data and an explicit "
        "decision-cost framework."
    )

st.divider()


# ---------------------------- Expected credit loss ----------------------------

st.header("2. Expected credit loss")

if ecl is not None:
    lgd_col = metric_column(ecl, ["lgd_assumption", "lgd"])
    ecl_col = metric_column(ecl, ["total_ecl_proxy", "total_ecl", "ecl"])
    exposure_col = metric_column(
        ecl, ["total_exposure_proxy", "exposure_proxy_total", "total_ead_proxy"]
    )

    if lgd_col and ecl_col:
        lgd_values = normalize_lgd_percent(ecl[lgd_col])
        ecl_values = pd.to_numeric(ecl[ecl_col], errors="coerce")
        valid = lgd_values.notna() & ecl_values.notna()
        ecl_plot = pd.DataFrame(
            {"LGD (%)": lgd_values[valid], "ECL proxy": ecl_values[valid]}
        ).sort_values("LGD (%)")

        # Scale from the 45% reference scenario for the interactive LGD.
        reference = ecl_plot.loc[
            (ecl_plot["LGD (%)"] - 45).abs().idxmin()
        ] if not ecl_plot.empty else None

        if reference is not None and reference["LGD (%)"] == 45:
            selected_ecl = float(reference["ECL proxy"]) * lgd / 0.45
            st.metric(
                f"Expected credit loss proxy at {lgd_percent}% LGD",
                f"{selected_ecl:,.0f}",
            )
            if exposure_col:
                exposure_values = pd.to_numeric(
                    ecl[exposure_col], errors="coerce"
                ).dropna()
                if not exposure_values.empty:
                    st.metric(
                        "Exposure proxy",
                        f"{float(exposure_values.iloc[0]):,.0f}",
                    )
        else:
            st.info(
                "Could not identify the 45% reference scenario. "
                "Showing the available LGD scenarios below."
            )

        if not ecl_plot.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(
                ecl_plot["LGD (%)"],
                ecl_plot["ECL proxy"] / 1_000_000,
                marker="o",
                label="Reported scenarios",
            )
            ax.axvline(
                lgd_percent,
                linestyle="--",
                label=f"Selected LGD: {lgd_percent}%",
            )
            ax.set_xlabel("Loss given default (LGD, %)")
            ax.set_ylabel("Expected credit loss proxy (millions)")
            ax.set_title("Expected loss sensitivity to LGD")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning(
            "The ECL report does not contain recognizable LGD and total-ECL "
            "columns. Inspect reports/portfolio_ecl_summary.csv."
        )
else:
    st.info("Run expected_credit_loss.py to generate the ECL report.")

st.caption(
    "ECL proxy = predicted probability of default × exposure proxy × assumed LGD. "
    "The exposure proxy is credit limit, not verified outstanding exposure."
)
st.divider()


# ---------------------------- Monte Carlo risk ----------------------------

st.header("3. Monte Carlo portfolio risk")

if monte_carlo is not None:
    st.dataframe(monte_carlo, use_container_width=True, hide_index=True)

    loss_scenarios = load_report("monte_carlo_scenario_losses.csv")
    if loss_scenarios is not None:
        independent_col = metric_column(
            loss_scenarios, ["independent_loss_proxy"]
        )
        correlated_col = metric_column(
            loss_scenarios, ["correlated_loss_proxy"]
        )

        if independent_col or correlated_col:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            if independent_col:
                values = pd.to_numeric(
                    loss_scenarios[independent_col], errors="coerce"
                ).dropna()
                ax.hist(
                    values / 1_000_000,
                    bins=60,
                    density=True,
                    alpha=0.5,
                    label="Independent defaults",
                )
            if correlated_col:
                values = pd.to_numeric(
                    loss_scenarios[correlated_col], errors="coerce"
                ).dropna()
                ax.hist(
                    values / 1_000_000,
                    bins=60,
                    density=True,
                    alpha=0.5,
                    label="Correlated defaults",
                )
            ax.set_xlabel("Simulated portfolio loss proxy (millions)")
            ax.set_ylabel("Density")
            ax.set_title("Monte Carlo portfolio loss distribution")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("Run monte_carlo_risk.py to generate Monte Carlo results.")

st.caption(
    "The common-factor correlation assumption is illustrative and has not "
    "been estimated or validated against a bank portfolio."
)
st.divider()


# ---------------------------- Stress testing ----------------------------

st.header("4. Macroeconomic stress testing")

if stress is not None:
    stress_display = stress.copy()

    pd_col = metric_column(
        stress_display, ["mean_stressed_pd", "mean_pd", "stressed_pd"]
    )
    ecl_stress_col = metric_column(
        stress_display, ["total_ecl_proxy", "total_ecl", "ecl"]
    )
    change_col = metric_column(
        stress_display, ["change_vs_baseline_pct", "change_vs_baseline"]
    )

    if pd_col:
        stress_display[pd_col] = pd.to_numeric(
            stress_display[pd_col], errors="coerce"
        ).map(format_rate)
    if ecl_stress_col:
        stress_display[ecl_stress_col] = pd.to_numeric(
            stress_display[ecl_stress_col], errors="coerce"
        ).map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    if change_col:
        stress_display[change_col] = pd.to_numeric(
            stress_display[change_col], errors="coerce"
        ).map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")

    st.dataframe(stress_display, use_container_width=True, hide_index=True)

    scenario_col = metric_column(stress, ["scenario"])
    if scenario_col and ecl_stress_col:
        plot_values = stress[[scenario_col, ecl_stress_col]].copy()
        plot_values[ecl_stress_col] = pd.to_numeric(
            plot_values[ecl_stress_col], errors="coerce"
        )
        plot_values = plot_values.dropna()
        if not plot_values.empty:
            chart = plot_values.set_index(scenario_col)[ecl_stress_col] / 1_000_000
            st.bar_chart(chart, y_label="ECL proxy (millions)")
else:
    st.info("Run stress_testing.py to generate stress-testing results.")

st.caption(
    "Stress scenarios apply illustrative PD multipliers; they are not "
    "estimated causal relationships between macroeconomic variables and defaults."
)
st.divider()


# ---------------------------- Fairness diagnostics ----------------------------

st.header("5. Fairness and sensitive-feature diagnostics")

st.markdown(
    "These group-level comparisons audit model behaviour across the dataset's "
    "coded `sex` groups. They are descriptive diagnostics, not a legal "
    "compliance test or a recommendation for lending decisions."
)

if fairness_comparison is not None:
    st.subheader("Overall model comparison: with versus without sex")
    st.dataframe(
        style_model_metrics(fairness_comparison),
        use_container_width=True,
        hide_index=True,
    )

if fairness is not None:
    display_fairness = fairness.copy()
    for col in [
        "observed_default_rate",
        "mean_predicted_pd",
        "predicted_positive_rate",
        "precision_at_threshold",
        "recall_at_threshold",
        "false_positive_rate",
        "false_negative_rate",
    ]:
        if col in display_fairness.columns:
            display_fairness[col] = pd.to_numeric(
                display_fairness[col], errors="coerce"
            ).map(format_rate)

    if "brier_score" in display_fairness.columns:
        display_fairness["brier_score"] = pd.to_numeric(
            display_fairness["brier_score"], errors="coerce"
        ).map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    st.subheader("Group-level metrics")
    st.dataframe(
        display_fairness,
        use_container_width=True,
        hide_index=True,
    )
    show_image(
        "fairness_group_rates.png",
        "Observed default and model positive-prediction rates by dataset sex code",
    )
else:
    st.info(
        "Fairness report not found. Run "
        "`.venv/bin/python fairness_audit.py`."
    )

with st.expander("Fairness interpretation cautions"):
    st.markdown(
        "- Similar overall accuracy or ROC-AUC does not guarantee similar "
        "error rates across groups.\n"
        "- Removing a sensitive feature does not guarantee fairness because "
        "other features may act as proxies.\n"
        "- Group metrics depend on sample sizes, observed outcomes, model "
        "thresholds and feature distributions.\n"
        "- The source data's coded categories should not be treated as "
        "evidence about other populations."
    )

st.divider()


# ---------------------------- Largest stress contributions ----------------------------

st.header("6. Largest customer-level stress contributions")

if top_customers is not None:
    st.caption(
        "Ranked by severe-stress ECL proxy. Dataset row references are not "
        "customer identifiers. These figures are not validated losses or "
        "lending-suitability scores."
    )

    display_customers = top_customers.copy()
    display_customers = display_customers.rename(
        columns={
            "row_index": "Dataset row",
            "actual_default": "Observed default",
            "predicted_pd": "Baseline PD",
            "Severe adverse_pd": "Stressed PD",
            "ead_proxy": "Exposure proxy",
            "Severe adverse_ecl_proxy": "Stressed ECL proxy",
        }
    )

    for col in ["Baseline PD", "Stressed PD"]:
        if col in display_customers.columns:
            display_customers[col] = pd.to_numeric(
                display_customers[col], errors="coerce"
            ).map(format_rate)

    for col in ["Exposure proxy", "Stressed ECL proxy"]:
        if col in display_customers.columns:
            display_customers[col] = pd.to_numeric(
                display_customers[col], errors="coerce"
            ).map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")

    if "Observed default" in display_customers.columns:
        display_customers["Observed default"] = (
            display_customers["Observed default"]
            .map({1: "Yes", 0: "No", "1": "Yes", "0": "No"})
            .fillna(display_customers["Observed default"].astype(str))
        )

    st.dataframe(
        display_customers,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Run stress_testing.py to generate the customer-level report.")


# ---------------------------- Footer ----------------------------

st.divider()
st.caption(
    "Dataset: UCI Default of Credit Card Clients, historical Taiwanese "
    "credit-card data. Evaluation is based on a random held-out test set, "
    "not out-of-time validation. Exposure is proxied by credit limit. "
    "All results are for research and educational purposes."
)
