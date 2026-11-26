
from pathlib import Path

import pandas as pd

from core.fraud_detector import analyze_fraud


BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "hybrid"
    / "final_test_dataset"
)

PREDICTIONS_FILE = (
    REPORT_DIR / "hybrid_predictions.csv"
)


def analyze_errors():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "Run python -m ml.evaluate_hybrid first."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    # Fraudulent messages without explicit alerts
    fraud_review = df[
        (df["actual_fraud"] == 1)
        & (df["fraud_alert"] == 0)
    ]

    # Legitimate messages requiring further attention
    legitimate_review = df[
        (df["actual_fraud"] == 0)
        & (df["triage_flag"] == 1)
    ]

    groups = [
        (
            "FRAUD WITHOUT EXPLICIT ALERT",
            fraud_review
        ),
        (
            "LEGITIMATE MESSAGES FLAGGED FOR ATTENTION",
            legitimate_review
        )
    ]

    diagnostics = []

    print("=" * 65)
    print("FraudGuard AI - Hybrid Error Analysis")
    print("=" * 65)

    print("\nDECISION DISTRIBUTION")

    print(
        pd.crosstab(
            df["actual_category"],
            df["decision"]
        ).to_string()
    )

    for group_name, cases in groups:

        print("\n" + "=" * 65)
        print(group_name)
        print("Total cases:", len(cases))
        print("=" * 65)

        for number, (_, row) in enumerate(
            cases.iterrows(), start=1
        ):

            result = analyze_fraud(
                row["message"],
                source=row["source"]
            )

            if not result["success"]:

                raise RuntimeError(
                    result.get("error")
                )

            reasons = result.get(
                "uncertainty_reasons", []
            )

            indicators = [
                item["indicator"]
                for item in result.get(
                    "indicators", []
                )
            ]

            explanations = result.get(
                "explanations", []
            )

            print(f"\nCASE {number}")
            print("-" * 65)

            print("Message:", row["message"])

            print(
                "Actual Category:",
                row["actual_category"]
            )

            print(
                "ML Category:",
                result["ml_category"]
            )

            print(
                "Final Category:",
                result["category"]
            )

            print(
                "Confidence:",
                result["confidence"]
            )

            print(
                "ML Fraud Score:",
                result["ml_fraud_score"]
            )

            print(
                "Rule Score:",
                result["rule_score"]
            )

            print(
                "Hybrid Score:",
                result["risk_score"]
            )

            print(
                "Decision:",
                result["decision"]
            )

            print(
                "Recognized Features:",
                result["recognized_features"]
            )

            print(
                "Uncertain:",
                result["uncertain"]
            )

            print(
                "Safety Context:",
                result.get("safety_context")
            )

            print(
                "Mixed Context:",
                result.get("mixed_context")
            )

            print(
                "Detected Indicators:",
                ", ".join(indicators)
                if indicators else "None"
            )

            print("\nUncertainty Reasons:")

            if reasons:

                for reason in reasons:

                    print("-", reason)

            else:

                print("None")

            print("\nExplanations:")

            for explanation in explanations:

                print("-", explanation)

            diagnostics.append({

                "error_group": group_name,

                "message": row["message"],

                "actual_category":
                    row["actual_category"],

                "ml_category":
                    result["ml_category"],

                "predicted_category":
                    result["category"],

                "confidence":
                    result["confidence"],

                "ml_fraud_score":
                    result["ml_fraud_score"],

                "rule_score":
                    result["rule_score"],

                "hybrid_score":
                    result["risk_score"],

                "decision":
                    result["decision"],

                "recognized_features":
                    result["recognized_features"],

                "uncertain":
                    result["uncertain"],

                "indicators":
                    " | ".join(indicators),

                "uncertainty_reasons":
                    " | ".join(reasons),

                "explanations":
                    " | ".join(explanations)

            })

    output_file = (
        REPORT_DIR / "error_diagnostics.csv"
    )

    pd.DataFrame(diagnostics).to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 65)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 65)

    print(
        "Fraud without explicit alert:",
        len(fraud_review)
    )

    print(
        "Legitimate messages flagged:",
        len(legitimate_review)
    )

    print(
        "Total diagnostic cases:",
        len(diagnostics)
    )

    print(
        "Report saved to:",
        output_file
    )

    print("Error analysis completed!")


if __name__ == "__main__":

    analyze_errors()
