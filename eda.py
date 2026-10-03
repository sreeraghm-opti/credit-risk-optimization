
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
target = "default_payment_next_month"

sns.set_theme(style="whitegrid")

print("=" * 65)
print("CREDIT RISK DATA — EXPLORATORY ANALYSIS")
print("=" * 65)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum():,}")
print(f"Missing values: {df.isna().sum().sum():,}")

print("\nColumn data types:")
print(df.dtypes)

# 2. Target distribution
print("\n2. TARGET DISTRIBUTION")
counts = df[target].value_counts().sort_index()
rates = df[target].value_counts(normalize=True).sort_index()

print("Counts:")
print(counts)
print("\nProportions:")
print(rates.round(4))

print(f"\nOverall default rate: {df[target].mean():.2%}")

# 3. Summary statistics
print("\n3. SUMMARY STATISTICS")
print(df.describe().T.round(2).to_string())

# 4. Default rates by selected customer characteristics
print("\n4. DEFAULT RATES BY CUSTOMER CHARACTERISTICS")

for column in ["sex", "education", "marriage"]:
    print(f"\nDefault rate by {column}:")
    print(
        df.groupby(column, observed=False)[target]
        .agg(["count", "mean"])
        .rename(columns={"mean": "default_rate"})
        .round(4)
        .to_string()
    )

# 5. Default rate by age group
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["<=25", "26-35", "36-45", "46-55", "56+"],
)

print("\nDefault rate by age group:")
print(
    df.groupby("age_group", observed=False)[target]
    .agg(["count", "mean"])
    .rename(columns={"mean": "default_rate"})
    .round(4)
    .to_string()
)

# 6. Check unusual categorical codes
print("\n5. CATEGORICAL VALUE CHECKS")

for column in ["sex", "education", "marriage"]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).sort_index())

# 7. Plot default distribution
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x=target, ax=ax)
ax.set_title("Credit Card Default: Target Distribution")
ax.set_xlabel("Default next month (0 = No, 1 = Yes)")
ax.set_ylabel("Number of customers")
fig.tight_layout()
fig.savefig(REPORT_DIR / "target_distribution.png", dpi=200)
plt.close(fig)

# 8. Plot default rate by age group
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="age_group", y=target, errorbar=None, ax=ax)
ax.set_title("Observed Default Rate by Age Group")
ax.set_xlabel("Age group")
ax.set_ylabel("Observed default rate")
fig.tight_layout()
fig.savefig(REPORT_DIR / "default_by_age.png", dpi=200)
plt.close(fig)

# 9. Correlation heatmap for numeric features
corr_columns = [
    "credit_limit",
    "age",
    "pay_status_sep",
    "pay_status_aug",
    "pay_status_jul",
    "pay_status_jun",
    "pay_status_may",
    "pay_status_apr",
    target,
]

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(
    df[corr_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
)
ax.set_title("Selected Feature Correlations")
fig.tight_layout()
fig.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=200)
plt.close(fig)

print("\n6. OUTPUTS")
print(f"Plots saved to: {REPORT_DIR}")
print(" - target_distribution.png")
print(" - default_by_age.png")
print(" - correlation_heatmap.png")

print("\nEDA complete.")