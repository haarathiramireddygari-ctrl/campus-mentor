
from pathlib import Path
from functools import lru_cache
from math import isfinite

import joblib

from core.preprocessing import preprocess_text

from core.fraud_detector import (
    analyze_fraud as analyze_legacy
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "binary_candidate_v2.joblib"
)

LOW_THRESHOLD = 36.0
ALERT_THRESHOLD = 50.0
LEGACY_DISAGREEMENT_THRESHOLD = 60.0

ENGINE_VERSION = "Unified-V4"


@lru_cache(maxsize=1)
def load_binary_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Binary Candidate V2 model not found."
        )

    return joblib.load(MODEL_FILE)


def predict_binary_score(message):

    model = load_binary_model()

    processed = preprocess_text(message)

    fraud_index = list(model.classes_).index(1)

    score = float(
        model.predict_proba(
            [processed]
        )[0][fraud_index] * 100
    )

    if not isfinite(score) or not 0 <= score <= 100:

        raise ValueError(
            "Invalid binary model score."
        )

    return score


def choose_unified_decision(binary_score, legacy):

    indicators = {
        item["indicator"]
        for item in legacy.get("indicators", [])
    }

    strong_indicator = (

        "credential_request" in indicators

        or "upi_collect_risk" in indicators

        or {
            "legal_threat",
            "payment_demand"
        }.issubset(indicators)

    )

    # Preserve strong contextual warnings
    if strong_indicator:

        return (
            "Potential Fraud",
            "Strong contextual security indicators."
        )

    # Insufficient information
    if legacy.get("recognized_features", 0) < 5:

        return (
            "Manual Verification Required",
            "Insufficient recognized text features."
        )

    # Mixed protective and suspicious language
    if legacy.get("mixed_context"):

        return (
            "Manual Verification Required",
            "Mixed protective and suspicious language."
        )

    # Primary binary fraud alert
    if binary_score >= ALERT_THRESHOLD:

        if legacy.get("safety_context"):

            return (
                "Manual Verification Required",
                "Fraud score conflicts with safety context."
            )

        return (
            "Potential Fraud",
            "Binary fraud-alert threshold exceeded."
        )

    # Intermediate binary score
    if binary_score > LOW_THRESHOLD:

        return (
            "Manual Verification Required",
            "Intermediate binary fraud score."
        )

    legacy_score = legacy.get("risk_score")

    if legacy_score is None:

        return (
            "Manual Verification Required",
            "Existing engine lacks sufficient evidence."
        )

    # Model disagreement safeguard
    if legacy_score > LEGACY_DISAGREEMENT_THRESHOLD:

        return (
            "Manual Verification Required",
            "Binary and existing hybrid scores disagree."
        )

    if indicators:

        return (
            "Manual Verification Required",
            "Security indicators require verification."
        )

    return (
        "Low Detected Risk",
        "Low binary score without conflicting indicators."
    )


def analyze_unified(message, source="SMS"):
    """
    Run the legacy detection engine and binary model,
    then apply the evaluated Unified V4 policy.
    """

    legacy = analyze_legacy(
        message,
        source=source
    )

    # Preserve existing input validation
    if not legacy["success"]:
        return legacy

    binary_score = predict_binary_score(message)

    decision, reason = choose_unified_decision(
        binary_score,
        legacy
    )

    return {

        "success": True,

        "source": source,

        "engine_version": ENGINE_VERSION,

        "binary_fraud_score": round(
            binary_score, 4
        ),

        "legacy_risk_score": legacy["risk_score"],

        "rule_score": legacy["rule_score"],

        "decision": decision,

        "decision_reason": reason,

        "provisional_category": legacy["category"],

        "category_confidence": legacy["confidence"],

        "legacy_assessment": legacy,

        "disclaimer": (
            "Experimental AI assessment. "
            "Scores are not calibrated fraud probabilities. "
            "Low detected risk does not guarantee safety."
        )

    }


if __name__ == "__main__":

    examples = [

        (
            "Transfer money immediately "
            "to avoid arrest.",
            "WhatsApp"
        ),

        (
            "Enter your UPI PIN to receive cashback.",
            "UPI"
        ),

        (
            "Your class starts at 9 AM tomorrow.",
            "SMS"
        ),

        (
            "Never share your OTP with anyone.",
            "SMS"
        ),

        (
            "They said I was being monitored remotely "
            "and could not consult a lawyer until I paid them.",
            "Call"
        )

    ]

    print("=" * 65)
    print("FraudGuard AI - Unified V4 Detection Engine")
    print("=" * 65)

    for message, source in examples:

        result = analyze_unified(
            message,
            source=source
        )

        print("\n" + "-" * 65)
        print("Message:", message)

        if not result["success"]:

            print("Error:", result.get("error"))
            continue

        print("Source:", result["source"])

        print(
            "Binary score:",
            result["binary_fraud_score"]
        )

        print(
            "Legacy score:",
            result["legacy_risk_score"]
        )

        print(
            "Rule score:",
            result["rule_score"]
        )

        print(
            "Provisional category:",
            result["provisional_category"]
        )

        print(
            "Decision:",
            result["decision"]
        )

        print(
            "Reason:",
            result["decision_reason"]
        )

    print("\nUnified detector testing completed!")
