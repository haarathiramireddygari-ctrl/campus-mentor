
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from core.preprocessing import preprocess_text

from core.fraud_detector import analyze_fraud
from core.unified_detector import choose_unified_decision


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR / "models" / "binary_candidate_v2.joblib"
)

def evaluate_dataset(model, filename):

    dataset_file = BASE_DIR / "data" / filename

    df = pd.read_csv(dataset_file)

    classes = list(model.classes_)

    fraud_index = classes.index(1)

    processed = df["message"].map(
        preprocess_text
    )

    scores = model.predict_proba(
        processed
    )[:, fraud_index] * 100

    records = []

    for index, row in df.iterrows():

        message = row["message"]

        source = row["source"]

        legacy = analyze_fraud(
            message,
            source=source
        )

        if not legacy["success"]:

            raise RuntimeError(
                f"Analysis failed at row {index}"
            )

        binary_score = float(scores[index])

        decision, reason = choose_unified_decision(
            binary_score,
            legacy
        )

        records.append({

            "row_id": index,

            "message": message,

            "source": source,

            "actual_category": row["category"],

            "actual_fraud": int(
                row["category"] != "Legitimate"
            ),

            "binary_score": round(binary_score, 4),

            "legacy_score": legacy["risk_score"],

            "legacy_decision": legacy["decision"],

            "provisional_category": legacy["category"],

            "rule_score": legacy["rule_score"],

            "unified_decision": decision,

            "decision_reason": reason,

            "fraud_alert": int(
                decision == "Potential Fraud"
            ),

            "manual_review": int(
                decision == "Manual Verification Required"
            ),

            "triage_flag": int(
                decision != "Low Detected Risk"
            )

        })

    results = pd.DataFrame(records)

    actual = results["actual_fraud"]

    alerts = results["fraud_alert"]

    triage = results["triage_flag"]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        alerts,
        labels=[0, 1]
    ).ravel()

    if tn + fp == 0 or tp + fn == 0:

        raise ValueError(
            "Both actual classes are required."
        )

    missed = results[

        (actual == 1)

        & (triage == 0)

    ]

    false_positives = results[

        (actual == 0)

        & (alerts == 1)

    ]

    legitimate = results[actual == 0]

    digital_arrest = results[
        results["actual_category"] == "Digital Arrest"
    ]

    metrics = {

        "Dataset": filename,

        "Total Records": len(results),

        "Binary Accuracy": round(
            accuracy_score(actual, alerts) * 100, 2
        ),

        "Fraud Precision": round(
            precision_score(
                actual, alerts, zero_division=0
            ) * 100, 2
        ),

        "Fraud Recall": round(
            recall_score(
                actual, alerts, zero_division=0
            ) * 100, 2
        ),

        "Fraud F1": round(
            f1_score(
                actual, alerts, zero_division=0
            ) * 100, 2
        ),

        "False Positive Rate": round(
            fp / (fp + tn) * 100, 2
        ),

        "Triage Fraud Recall": round(
            recall_score(
                actual, triage, zero_division=0
            ) * 100, 2
        ),

        "Legitimate Triage Rate": round(
            legitimate["triage_flag"].mean() * 100,
            2
        ),

        "Manual Review Rate": round(
            results["manual_review"].mean() * 100,
            2
        ),

        "Digital Arrest Alert Recall": round(
            digital_arrest["fraud_alert"].mean()
            * 100, 2
        ) if not digital_arrest.empty else None,

        "Fraud Without Alert or Review": len(missed),

        "True Positives": int(tp),

        "True Negatives": int(tn),

        "False Positives": int(fp),

        "Fraud Without Explicit Alert": int(fn)

    }

    report_dir = (

        BASE_DIR

        / "reports"

        / "unified_v4"

        / dataset_file.stem

    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        report_dir / "unified_predictions.csv",
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        report_dir / "unified_metrics.csv",
        index=False
    )

    missed.to_csv(
        report_dir / "missed_fraud.csv",
        index=False
    )

    false_positives.to_csv(
        report_dir / "false_positives.csv",
        index=False
    )

    print("\n" + "=" * 65)

    print("FraudGuard AI - Unified V4 Evaluation")

    print("=" * 65)

    for name, value in metrics.items():

        print(f"{name:<38}: {value}")

    print("\nDECISION DISTRIBUTION")

    print(
        pd.crosstab(
            results["actual_category"],
            results["unified_decision"]
        ).to_string()
    )

    print("\nDECISION REASONS")

    print(
        results["decision_reason"]
        .value_counts()
        .to_string()
    )

    print("\nMISSED FRAUD CASES")

    if missed.empty:

        print("No fraud cases classified as low risk.")

    else:

        print(
            missed[
                [
                    "message",
                    "binary_score",
                    "legacy_score",
                    "decision_reason"
                ]
            ].to_string(index=False)
        )

    print("\nFALSE POSITIVE CASES")

    if false_positives.empty:

        print("No explicit false fraud alerts.")

    else:

        print(
            false_positives[
                [
                    "message",
                    "binary_score",
                    "rule_score",
                    "decision_reason"
                ]
            ].to_string(index=False)
        )

    print("\nReports saved in:", report_dir)


def main():

    model = joblib.load(MODEL_FILE)

    datasets = [

        "hybrid_development_100.csv",

        "independent_holdout_100.csv"

    ]

    for filename in datasets:

        evaluate_dataset(model, filename)

    print("\nUnified V4 evaluation completed!")


if __name__ == "__main__":

    main()
