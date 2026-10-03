
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# MONTE CARLO PORTFOLIO CREDIT RISK
# ============================================================

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
INPUT_FILE = REPORTS / "threshold_test_predictions.csv"

N_SIMULATIONS = 20_000
RANDOM_SEED = 42

# Illustrative assumptions, not estimated from this dataset.
LGD = 0.45
ASSET_CORRELATION = 0.15

# Load customer-level out-of-sample predictions.
df = pd.read_csv(INPUT_FILE)

required = {"row_index", "actual_default", "predicted_pd"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

pd_values = df["predicted_pd"].to_numpy(dtype=float)

if not np.isfinite(pd_values).all():
    raise ValueError("Predicted PD contains non-finite values.")

if ((pd_values < 0) | (pd_values > 1)).any():
    raise ValueError("Predicted PD must be between 0 and 1.")

if not 0 <= ASSET_CORRELATION < 1:
    raise ValueError("Asset correlation must be in [0, 1).")

# Credit limit is only an exposure proxy, not actual EAD.
raw_data = pd.read_csv(ROOT / "data" / "raw" / "credit_card_default.csv")
raw_data.index = raw_data.index.astype(int)

df["row_index"] = df["row_index"].astype(int)

if df["row_index"].duplicated().any():
    raise ValueError("Duplicate row_index values found.")

df = df.set_index("row_index")
df = df.join(
    raw_data[["credit_limit"]].rename(
        columns={"credit_limit": "credit_limit_raw"}
    ),
    how="left",
    validate="one_to_one",
)

if df["credit_limit_raw"].isna().any():
    raise ValueError("Some prediction rows could not be matched to raw data.")

ead = df["credit_limit_raw"].clip(lower=0).to_numpy(dtype=float)
pd_values = df["predicted_pd"].to_numpy(dtype=float)

# Convert unconditional PDs into latent normal default thresholds.
# Defaults occur when the latent asset variable falls below the threshold.
thresholds = norm.ppf(np.clip(pd_values, 1e-10, 1 - 1e-10))

rng = np.random.default_rng(RANDOM_SEED)

# Common economic factor plus idiosyncratic customer-level risk.
systematic = rng.standard_normal(N_SIMULATIONS)
idiosyncratic = rng.standard_normal((N_SIMULATIONS, len(df)))

latent_asset = (
    np.sqrt(ASSET_CORRELATION) * systematic[:, None]
    + np.sqrt(1 - ASSET_CORRELATION) * idiosyncratic
)

defaults = latent_asset < thresholds[None, :]

# Loss proxy per scenario: default indicator × LGD × exposure proxy.
losses = (defaults * ead[None, :]).sum(axis=1) * LGD

# A comparison model with independent defaults (zero asset correlation).
rng_independent = np.random.default_rng(RANDOM_SEED + 1)
independent_defaults = (
    rng_independent.random((N_SIMULATIONS, len(df)))
    < pd_values[None, :]
)
independent_losses = (
    independent_defaults * ead[None, :]
).sum(axis=1) * LGD

def risk_summary(loss_array, model_name):
    var_95 = np.quantile(loss_array, 0.95)
    var_99 = np.quantile(loss_array, 0.99)

    # Expected Shortfall: mean loss in the worst 5% / 1% of scenarios.
    es_95 = loss_array[loss_array >= var_95].mean()
    es_99 = loss_array[loss_array >= var_99].mean()

    return {
        "model": model_name,
        "mean_loss_proxy": loss_array.mean(),
        "std_loss_proxy": loss_array.std(ddof=1),
        "VaR_95": var_95,
        "ES_95": es_95,
        "VaR_99": var_99,
        "ES_99": es_99,
    }


summary = pd.DataFrame(
    [
        risk_summary(independent_losses, "Independent defaults"),
        risk_summary(losses, f"Correlated defaults (rho={ASSET_CORRELATION:.2f})"),
    ]
)

print("=" * 72)
print("MONTE CARLO PORTFOLIO CREDIT RISK")
print("=" * 72)
print(f"Customers:                 {len(df):,}")
print(f"Simulations per model:     {N_SIMULATIONS:,}")
print(f"LGD assumption:            {LGD:.0%}")
print(f"Asset correlation:         {ASSET_CORRELATION:.2f}")
print(f"Total exposure proxy:      {ead.sum():,.2f}")
print()
print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# Save risk metrics.
summary.to_csv(REPORTS / "monte_carlo_risk_summary.csv", index=False)

# Save scenario-level losses for further analysis.
scenario_results = pd.DataFrame(
    {
        "scenario": np.arange(1, N_SIMULATIONS + 1),
        "independent_loss_proxy": independent_losses,
        "correlated_loss_proxy": losses,
    }
)
scenario_results.to_csv(
    REPORTS / "monte_carlo_scenario_losses.csv", index=False
)

# Plot both simulated portfolio loss distributions.
plt.figure(figsize=(10, 6))
plt.hist(
    independent_losses / 1_000_000,
    bins=70,
    alpha=0.60,
    density=True,
    label="Independent defaults",
)
plt.hist(
    losses / 1_000_000,
    bins=70,
    alpha=0.60,
    density=True,
    label=f"Correlated defaults (rho={ASSET_CORRELATION:.2f})",
)
plt.xlabel("Portfolio loss proxy (millions of dataset currency units)")
plt.ylabel("Density")
plt.title("Monte Carlo Portfolio Loss Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS / "monte_carlo_loss_distribution.png", dpi=160)
plt.close()

print("\nSaved:")
print(REPORTS / "monte_carlo_risk_summary.csv")
print(REPORTS / "monte_carlo_scenario_losses.csv")
print(REPORTS / "monte_carlo_loss_distribution.png")
print("\nMonte Carlo simulation complete.")
print("All loss estimates use illustrative LGD and exposure assumptions.")