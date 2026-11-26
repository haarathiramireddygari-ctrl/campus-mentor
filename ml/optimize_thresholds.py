
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.preprocessing import preprocess_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR / "models" / "binary_candidate.joblib"
)

DATA_FILE = (
    BASE_DIR / "data" / "hybrid_development_100.csv"
)

REPORT_DIR = (
    BASE_DIR / "reports" / "threshold_optimization"
)


def optimize_thresholds():

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(DATA_FILE)

    messages = df["message"].map(preprocess_text)

    actual = (
        df["category"] != "Legitimate"
    ).astype(int).to_numpy()

    classes = list(
        model.named_steps["classifier"].classes_
    )

    fraud_index = classes.index(1)

    # Use original unrounded model scores
    scores = model.predict_proba(
        messages
    )[:, fraud_index]

    fraud = actual == 1
    legitimate = actual == 0

    if not fraud.any() or not legitimate.any():
        raise ValueError("Both classes are required.")

    results = []

    print("=" * 70)
    print("FraudGuard AI - Threshold Optimization")
    print("=" * 70)

    print("Development records:", len(df))

    # Evaluate thresholds from 0 to 100
    # in increments of one percentage point
    for low in range(0, 100):

        for high in range(low + 1, 101):

            alerts = scores >= high / 100

            low_risk = scores <= low / 100

            review = ~alerts & ~low_risk

            triage = alerts | review

            tp = int((alerts & fraud).sum())
            fp = int((alerts & legitimate).sum())
            fn = int((~alerts & fraud).sum())
            tn = int((~alerts & legitimate).sum())

            precision = (
                tp / (tp + fp) * 100
                if tp + fp else 0
            )

            recall = tp / (tp + fn) * 100

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall else 0
            )

            results.append({

                "Low Threshold": low,

                "Alert Threshold": high,

                "Binary Accuracy": round(
                    (tp + tn) / len(df) * 100, 2
                ),

                "Fraud Precision": round(
                    precision, 2
                ),

                "Fraud Recall": round(
                    recall, 2
                ),

                "Fraud F1": round(f1, 2),

                "False Positive Rate": round(
                    fp / legitimate.sum() * 100, 2
                ),

                "Triage Fraud Recall": round(
                    triage[fraud].mean() * 100, 2
                ),

                "Legitimate Triage Rate": round(
                    triage[legitimate].mean() * 100, 2
                ),

                "Manual Review Rate": round(
                    review.mean() * 100, 2
                ),

                "Missed Fraud": int(
                    (fraud & low_risk).sum()
                )

            })

    results_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        REPORT_DIR / "all_threshold_results.csv",
        index=False
    )

    print(
        "\nThreshold combinations tested:",
        len(results_df)
    )

    # Target 1: Satisfy all performance objectives
    qualified = results_df[

        (results_df["Binary Accuracy"] >= 90)

        & (results_df["Fraud Precision"] >= 95)

        & (results_df["Fraud Recall"] >= 95)

        & (results_df["False Positive Rate"] <= 10)

        & (results_df["Triage Fraud Recall"] >= 98)

        & (results_df["Legitimate Triage Rate"] <= 20)

    ]

    qualified.to_csv(
        REPORT_DIR / "qualified_thresholds.csv",
        index=False
    )

    print("\nPOLICIES MEETING ALL TARGETS")
    print("-" * 70)
    print("Qualified combinations:", len(qualified))

    if not qualified.empty:

        print(
            qualified.sort_values(
                [
                    "Missed Fraud",
                    "Fraud F1",
                    "Legitimate Triage Rate"
                ],
                ascending=[True, False, True]
            ).head(5).to_string(index=False)
        )

    else:

        print(
            "No tested threshold combination "
            "satisfies all targets."
        )

    # Target 2: Preserve zero missed scams
    # while reducing legitimate triage
    safe = results_df[
        results_df["Missed Fraud"] == 0
    ]

    safe = safe.sort_values(
        [
            "Legitimate Triage Rate",
            "False Positive Rate",
            "Fraud Recall"
        ],
        ascending=[True, True, False]
    )

    print("\nBEST ZERO-MISSED-FRAUD POLICIES")
    print("-" * 70)

    print(
        safe.head(5).to_string(index=False)
    )

    safe.head(20).to_csv(
        REPORT_DIR / "best_safe_policies.csv",
        index=False
    )

    # Target 3: High recall with limited false alerts
    balanced = results_df[

        (results_df["False Positive Rate"] <= 10)

        & (results_df["Fraud Precision"] >= 95)

        & (results_df["Missed Fraud"] == 0)

    ]

    balanced = balanced.sort_values(
        [
            "Fraud Recall",
            "Legitimate Triage Rate",
            "Manual Review Rate"
        ],
        ascending=[False, True, True]
    )

    print("\nBEST HIGH-PRECISION POLICIES")
    print("-" * 70)

    if balanced.empty:

        print("No qualifying policies found.")

    else:

        print(
            balanced.head(5).to_string(index=False)
        )

    balanced.head(20).to_csv(
        REPORT_DIR / "best_balanced_policies.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("Threshold optimization completed!")
    print("Reports saved in:", REPORT_DIR)
    print("=" * 70)


if __name__ == "__main__":

    optimize_thresholds()
