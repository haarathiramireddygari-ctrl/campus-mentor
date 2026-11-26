
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone

from ml.train_binary_model import (
    load_dataset,
    evaluate_policy
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = (
    BASE_DIR / "reports" / "binary_candidate_v2"
)

LOW_THRESHOLD = 0.53
ALERT_THRESHOLD = 0.76


def train_candidate_v2():

    original = load_dataset(
        DATA_DIR / "improved_dataset.csv"
    )

    extra = load_dataset(
        DATA_DIR / "training_augmentation.csv"
    )

    dev = load_dataset(
        DATA_DIR / "hybrid_development_100.csv"
    )

    # Reject empty messages
    if extra["processed_message"].str.strip().eq("").any():

        raise ValueError(
            "Augmentation contains empty messages."
        )

    # Reject duplicate augmentation records
    if extra["processed_message"].duplicated().any():

        raise ValueError(
            "Duplicate augmentation messages detected."
        )

    original_messages = set(
        original["processed_message"]
    )

    extra_messages = set(
        extra["processed_message"]
    )

    if original_messages & extra_messages:

        raise ValueError(
            "Augmentation overlaps original training data."
        )

    # Merge training data
    combined = pd.concat(
        [original, extra],
        ignore_index=True
    )

    # Check known evaluation datasets for leakage
    evaluation_files = [

        "independent_test.csv",

        "final_test_dataset.csv",

        "hybrid_development_100.csv"

    ]

    for filename in evaluation_files:

        evaluation = load_dataset(
            DATA_DIR / filename
        )

        overlap = set(
            combined["processed_message"]
        ) & set(
            evaluation["processed_message"]
        )

        # Existing training may already overlap
        # an older development benchmark.
        new_overlap = extra_messages & set(
            evaluation["processed_message"]
        )

        if new_overlap:

            raise ValueError(
                f"New training data overlaps "
                f"{filename}: {len(new_overlap)}"
            )

        print(
            f"Total overlap with {filename}:",
            len(overlap)
        )

        if (
            filename == "hybrid_development_100.csv"
            and overlap
        ):

            raise ValueError(
                "Development dataset leakage detected."
            )

    print("=" * 65)
    print("FraudGuard AI - Binary Candidate V2")
    print("=" * 65)

    print("Original records:", len(original))

    print("Additional records:", len(extra))

    print("Combined records:", len(combined))

    print("\nTraining distribution:")

    print(
        combined["category"].value_counts()
    )

    # Load the existing locally trained candidate
    baseline = joblib.load(
        MODEL_DIR / "binary_candidate.joblib"
    )

    # Preserve the same model architecture
    candidate = clone(baseline)

    print("\nTraining improved candidate...")

    candidate.fit(
        combined["processed_message"],
        combined["binary_label"]
    )

    y_true = dev["binary_label"].to_numpy()

    messages = dev["processed_message"]

    baseline_index = list(
        baseline.classes_
    ).index(1)

    candidate_index = list(
        candidate.classes_
    ).index(1)

    baseline_scores = baseline.predict_proba(
        messages
    )[:, baseline_index]

    candidate_scores = candidate.predict_proba(
        messages
    )[:, candidate_index]

    # Compare both models using identical thresholds
    metrics_v1 = evaluate_policy(

        y_true,
        baseline_scores,

        alert_threshold=ALERT_THRESHOLD,
        low_threshold=LOW_THRESHOLD,

        policy_name="Binary Candidate V1"
    )

    metrics_v2 = evaluate_policy(

        y_true,
        candidate_scores,

        alert_threshold=ALERT_THRESHOLD,
        low_threshold=LOW_THRESHOLD,

        policy_name="Binary Candidate V2"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save separate model - do not overwrite
    joblib.dump(

        candidate,

        MODEL_DIR / "binary_candidate_v2.joblib"
    )

    pd.DataFrame([
        metrics_v1,
        metrics_v2
    ]).to_csv(

        REPORT_DIR / "model_comparison.csv",

        index=False
    )

    predictions = dev[
        ["message", "category", "binary_label"]
    ].copy()

    predictions["v1_fraud_score"] = (
        baseline_scores * 100
    ).round(2)

    predictions["v2_fraud_score"] = (
        candidate_scores * 100
    ).round(2)

    predictions.to_csv(

        REPORT_DIR / "predictions.csv",

        index=False
    )

    print("\n" + "=" * 65)

    print("Training completed!")

    print(
        "Model:",
        MODEL_DIR / "binary_candidate_v2.joblib"
    )

    print("Reports:", REPORT_DIR)

    print("=" * 65)


if __name__ == "__main__":

    train_candidate_v2()
