
import argparse
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from core.fraud_detector import analyze_fraud


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATASET = (
    BASE_DIR / "data" / "final_test_dataset.csv"
)

VALID_CATEGORIES = {
    "Legitimate",
    "Digital Arrest",
    "KYC Fraud",
    "Courier Fraud",
    "UPI Fraud"
}

VALID_SOURCES = {
    "whatsapp": "WhatsApp",
    "sms": "SMS",
    "call": "Call",
    "upi": "UPI"
}


def evaluate_hybrid(dataset_file):

    dataset_file = Path(dataset_file)

    report_dir = (
        BASE_DIR
        / "reports"
        / "hybrid"
        / dataset_file.stem
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(dataset_file)

    required_columns = {
        "message",
        "category"
    }

    if not required_columns.issubset(df.columns):

        raise ValueError(
            "Dataset must contain message "
            "and category columns."
        )

    if df.empty:

        raise ValueError("Dataset is empty.")

    if df[["message", "category"]].isna().any().any():

        raise ValueError(
            "Dataset contains missing messages "
            "or categories."
        )

    results = []

    print("=" * 65)
    print("FraudGuard AI - Hybrid Engine Evaluation")
    print("=" * 65)

    print("Dataset:", dataset_file.name)
    print("Total records:", len(df))

    # Analyse every evaluation message
    for index, row in df.iterrows():

        message = str(row["message"]).strip()
        actual_category = str(row["category"]).strip()

        if actual_category not in VALID_CATEGORIES:

            raise ValueError(
                f"Invalid category at row {index}: "
                f"{actual_category}"
            )

        if not message:

            raise ValueError(
                f"Empty message at row {index}"
            )

        source = str(
            row.get("source", "SMS")
        ).strip().lower()

        if source not in VALID_SOURCES:

            raise ValueError(
                f"Invalid source at row {index}: "
                f"{source}"
            )

        source = VALID_SOURCES[source]

        result = analyze_fraud(
            message,
            source=source
        )

        if not result["success"]:

            raise RuntimeError(
                f"Analysis failed at row {index}: "
                f"{result.get('error')}"
            )

        actual_fraud = int(
            actual_category != "Legitimate"
        )

        decision = result["decision"]

        # Strict detection: explicit fraud alerts only
        fraud_alert = int(
            decision == "Potential Fraud"
        )

        # Review includes uncertain and insufficient cases
        manual_review = int(
            decision in [
                "Manual Verification Required",
                "Insufficient Information"
            ]
        )

        # Triage means alert or further verification
        triage_flag = int(
            fraud_alert or manual_review
        )

        results.append({

            "message": message,

            "source": source,

            "actual_category": actual_category,

            "predicted_category":
                result["category"],

            "ml_category":
                result["ml_category"],

            "actual_fraud": actual_fraud,

            "fraud_alert": fraud_alert,

            "manual_review": manual_review,

            "triage_flag": triage_flag,

            "ml_fraud_score":
                result["ml_fraud_score"],

            "rule_score":
                result["rule_score"],

            "hybrid_risk_score":
                result["risk_score"],

            "risk_level":
                result["risk_level"],

            "decision": decision,

            "uncertain":
                result["uncertain"],

            "indicator_count":
                result["indicator_count"]

        })

    predictions = pd.DataFrame(results)

    y_true = predictions["actual_fraud"]

    y_pred = predictions["fraud_alert"]

    # Strict fraud-alert performance
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    if (tp + fn) == 0 or (tn + fp) == 0:

        raise ValueError(
            "Evaluation requires both legitimate "
            "and fraudulent examples."
        )

    accuracy = accuracy_score(
        y_true, y_pred
    ) * 100

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    ) * 100

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    ) * 100

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    ) * 100

    false_positive_rate = (
        fp / (fp + tn)
    ) * 100

    category_accuracy = accuracy_score(
        predictions["actual_category"],
        predictions["predicted_category"]
    ) * 100

    # Evaluate digital arrest detection
    arrest = predictions[
        predictions["actual_category"]
        == "Digital Arrest"
    ]

    arrest_recall = (
        arrest["fraud_alert"].mean() * 100
        if not arrest.empty else None
    )

    arrest_category_recall = (
        (
            arrest["predicted_category"]
            == "Digital Arrest"
        ).mean() * 100
        if not arrest.empty else None
    )

    # Evaluate alert + manual-review policy
    triage_recall = recall_score(
        y_true,
        predictions["triage_flag"],
        zero_division=0
    ) * 100

    review_rate = (
        predictions["manual_review"].mean()
        * 100
    )

    legitimate = predictions[
        predictions["actual_fraud"] == 0
    ]

    legitimate_triage_rate = (
        legitimate["triage_flag"].mean()
        * 100
    )

    # Investigate different error types
    false_positives = predictions[
        (predictions["actual_fraud"] == 0)
        & (predictions["fraud_alert"] == 1)
    ]

    non_alerted_fraud = predictions[
        (predictions["actual_fraud"] == 1)
        & (predictions["fraud_alert"] == 0)
    ]

    missed_fraud = predictions[
        (predictions["actual_fraud"] == 1)
        & (predictions["triage_flag"] == 0)
    ]

    review_cases = predictions[
        predictions["manual_review"] == 1
    ]

    # Save evaluation results
    metrics = {

        "Total Records": len(predictions),

        "Binary Accuracy": round(accuracy, 2),

        "Fraud Precision": round(precision, 2),

        "Fraud Recall": round(recall, 2),

        "Fraud F1": round(f1, 2),

        "Multiclass Accuracy":
            round(category_accuracy, 2),

        "Digital Arrest Alert Recall":
            round(arrest_recall, 2)
            if arrest_recall is not None else None,

        "Digital Arrest Category Recall":
            round(arrest_category_recall, 2)
            if arrest_category_recall is not None
            else None,

        "False Positive Rate":
            round(false_positive_rate, 2),

        "Triage Fraud Recall":
            round(triage_recall, 2),

        "Manual Review Rate":
            round(review_rate, 2),

        "Legitimate Triage Rate":
            round(legitimate_triage_rate, 2),

        "True Positives": int(tp),

        "True Negatives": int(tn),

        "False Positives": int(fp),

        "Fraud Without Explicit Alert": int(fn),

        "Fraud Without Alert or Review":
            len(missed_fraud),

        "Manual Review Cases":
            len(review_cases)

    }

    pd.DataFrame([metrics]).to_csv(
        report_dir / "hybrid_metrics.csv",
        index=False
    )

    predictions.to_csv(
        report_dir / "hybrid_predictions.csv",
        index=False
    )

    false_positives.to_csv(
        report_dir / "false_positives.csv",
        index=False
    )

    non_alerted_fraud.to_csv(
        report_dir / "non_alerted_fraud.csv",
        index=False
    )

    missed_fraud.to_csv(
        report_dir / "missed_fraud.csv",
        index=False
    )

    review_cases.to_csv(
        report_dir / "manual_review_cases.csv",
        index=False
    )

    # Print evaluation summary
    print("\nHYBRID ENGINE PERFORMANCE")
    print("-" * 65)

    for name, value in metrics.items():

        print(f"{name:<38}: {value}")

    print("\nBINARY FRAUD-ALERT CONFUSION MATRIX")
    print("-" * 65)

    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    print("\nFALSE POSITIVE ANALYSIS")
    print("-" * 65)

    if false_positives.empty:

        print("No false positives.")

    else:

        for _, row in false_positives.iterrows():

            print("\nMessage:", row["message"])
            print(
                "Predicted:", row["predicted_category"]
            )
            print(
                "Risk score:", row["hybrid_risk_score"]
            )

    print("\nFRAUD WITHOUT EXPLICIT ALERT")
    print("-" * 65)

    if non_alerted_fraud.empty:

        print("All fraud examples generated alerts.")

    else:

        for _, row in non_alerted_fraud.iterrows():

            print("\nMessage:", row["message"])
            print("Actual:", row["actual_category"])
            print("Decision:", row["decision"])
            print(
                "Risk score:", row["hybrid_risk_score"]
            )

    print("\n" + "=" * 65)
    print("Evaluation completed!")
    print("Reports saved in:", report_dir)
    print("=" * 65)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET
    )

    args = parser.parse_args()

    evaluate_hybrid(args.dataset)
