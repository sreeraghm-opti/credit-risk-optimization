
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download the UCI credit card default dataset

# Download the UCI credit card default dataset
print("Downloading credit-risk dataset...")

dataset = fetch_ucirepo(id=350)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Give the target a consistent name
if y.shape[1] != 1:
    raise ValueError("Expected exactly one target column.")

target = y.iloc[:, 0].rename("default_payment_next_month")

# Assign documented feature names in dataset order
feature_names = [
    "credit_limit",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_status_sep",
    "pay_status_aug",
    "pay_status_jul",
    "pay_status_jun",
    "pay_status_may",
    "pay_status_apr",
    "bill_amount_sep",
    "bill_amount_aug",
    "bill_amount_jul",
    "bill_amount_jun",
    "bill_amount_may",
    "bill_amount_apr",
    "payment_amount_sep",
    "payment_amount_aug",
    "payment_amount_jul",
    "payment_amount_jun",
    "payment_amount_may",
    "payment_amount_apr",
]

if X.shape[1] != len(feature_names):
    raise ValueError(
        f"Expected 23 features, found {X.shape[1]}"
    )

X.columns = feature_names

# Combine features and target
df = pd.concat(
    [
        X.reset_index(drop=True),
        target.reset_index(drop=True),
    ],
    axis=1,
)

# Save the downloaded data
output_path = RAW_DIR / "credit_card_default.csv"
df.to_csv(output_path, index=False)

# Print a basic data-quality report
print("\nDownload successful!")
print(f"Dataset shape: {df.shape}")
print(f"Saved to: {output_path}")
print(f"Missing values: {df.isna().sum().sum()}")

print("\nTarget distribution:")
print(df["default_payment_next_month"].value_counts())

print("\nDefault rate:")
print(
    f"{df['default_payment_next_month'].mean():.2%}"
)

print("\nFirst five rows:")
print(df.head())