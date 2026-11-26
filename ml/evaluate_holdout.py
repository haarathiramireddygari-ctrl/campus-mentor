
from pathlib import Path

import joblib
import pandas as pd

from ml.train_binary_model import (
    load_dataset,
    evaluate_policy
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = (
    BASE_DIR / "reports" / "independent_holdout"
)

HOLDOUT_FILE = (
    DATA_DIR / "independent_holdout_100.csv"
)


def evaluate_holdout():

    holdout = load_dataset(HOLDOUT_FILE)

    expected_counts = {
        "Legitimate": 50,
        "Digital Arrest": 13,
        "KYC Fraud": 13,
        "Courier Fraud": 12,
        "UPI Fraud": 12
    }

    if holdout["category"].value_counts().to_dict() != expected_counts:
        raise ValueError(
            "Incorrect holdout category distribution."
        )

    if holdout["processed_message"].str.strip().eq("").any():
        raise ValueError("Empty holdout messages detected.")

    if holdout["processed_message"].duplicated().any():
        raise ValueError("Duplicate holdout messages detected.")

    valid_sources = {
        "WhatsApp", "SMS", "Call", "UPI"
    }

    if "source" not in holdout.columns:
        raise ValueError("Source column is missing.")

    if not set(holdout["source"]).issubset(valid_sources):
        raise ValueError("Invalid source values.")

    holdout_messages = set(
        holdout["processed_message"]
    )

    previous_datasets = [

        "fraud_dataset.csv",

        "improved_dataset.csv",

        "training_augmentation.csv",

        "independent_test.csv",

        "final_test_dataset.csv",

        "hybrid_development_100.csv"

    ]

    print("=" * 65)
    print("FraudGuard AI - Independent Holdout Evaluation")
    print("=" * 65)

    print("Holdout records:", len(holdout))

    # Check for exact normalized overlap
    for filename in previous_datasets:

        previous = load_dataset(
            DATA_DIR / filename
        )

        overlap = holdout_messages & set(
            previous["processed_message"]
        )

        print(
            f"Overlap with {filename}:",
            len(overlap)
        )

        if overlap:
            raise ValueError(
                f"Holdout leakage detected: {filename}"
            )

    print("\nAll exact-overlap checks passed.")

    # Previously selected thresholds
    candidates = [

        (
            "Binary Candidate V1",
            "binary_candidate.joblib",
            0.53,
            0.76
        ),

        (
            "Binary Candidate V2",
            "binary_candidate_v2.joblib",
            0.36,
            0.50
        )

    ]

    actual = holdout["binary_label"].to_numpy()

    messages = holdout["processed_message"]

    metrics = []

    predictions = holdout[
        ["category", "source", "binary_label"]
    ].copy()

    for name, filename, low, high in candidates:

        model = joblib.load(
            MODEL_DIR / filename
        )

        fraud_index = list(
            model.classes_
        ).index(1)

        scores = model.predict_proba(
            messages
        )[:, fraud_index]

        result = evaluate_policy(

            actual,

            scores,

            alert_threshold=high,

            low_threshold=low,

            policy_name=name

        )

        metrics.append(result)

        predictions[name + " Score"] = (
            scores * 100
        ).round(2)

        predictions[name + " Decision"] = [

            "Potential Fraud"
            if score >= high

            else "Low Detected Risk"
            if score <= low

            else "Manual Verification Required"

            for score in scores

        ]

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(metrics).to_csv(

        REPORT_DIR / "holdout_metrics.csv",

        index=False

    )

    predictions.to_csv(

        REPORT_DIR / "holdout_predictions.csv",

        index=False

    )

    print("\n" + "=" * 65)
    print("Holdout evaluation completed!")
    print("Reports saved in:", REPORT_DIR)
    print("=" * 65)


if __name__ == "__main__":

    evaluate_holdout()
