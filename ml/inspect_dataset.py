
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "fraud_dataset.csv"
output_file = DATA_DIR / "cleaned_dataset.csv"

# Load dataset
df = pd.read_csv(input_file)

print("=" * 50)
print("FraudGuard AI - Dataset Inspection")
print("=" * 50)

# Original dataset
print("\nOriginal dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing records
df = df.dropna().copy()

# Standardize message text
df["message"] = (
    df["message"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Remove empty messages
df = df[df["message"] != ""].copy()

# Validate labels and categories
category_labels = {
    "Legitimate": 0,
    "Digital Arrest": 1,
    "KYC Fraud": 1,
    "Courier Fraud": 1,
    "UPI Fraud": 1
}

valid_rows = (
    df["category"].isin(category_labels)
    & df["label"].isin([0, 1])
    & (df["category"].map(category_labels) == df["label"])
)

invalid_count = (~valid_rows).sum()
print("\nInvalid records:", invalid_count)

df = df[valid_rows].copy()

# Check contradictory categories
conflicts = (
    df.groupby("message")["category"].nunique()
)

conflicting_messages = conflicts[conflicts > 1].index

print("\nConflicting messages:", len(conflicting_messages))

df = df[
    ~df["message"].isin(conflicting_messages)
].copy()

# Check duplicate messages
duplicates = df["message"].duplicated().sum()

print("\nDuplicate messages found:", duplicates)

# Remove duplicate messages
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

# Final dataset information
print("\nCleaned dataset shape:", df.shape)

print("\nFinal category distribution:")
print(df["category"].value_counts())

print("\nFinal label distribution:")
print(df["label"].value_counts())

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nRemaining duplicate messages:")
print(df["message"].duplicated().sum())

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:")
print(output_file)

print("\nDataset cleaning completed successfully!")
