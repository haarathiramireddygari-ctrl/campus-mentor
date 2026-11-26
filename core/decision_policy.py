
from copy import deepcopy


HIGH_ML_FRAUD_THRESHOLD = 85.0

MIN_RECOGNIZED_FEATURES = 5

POLICY_VERSION = "Hybrid-V2"

FRAUD_CATEGORIES = {
    "Digital Arrest",
    "KYC Fraud",
    "Courier Fraud",
    "UPI Fraud"
}


def apply_decision_policy(result):
    """
    Separate overall fraud risk from uncertainty
    about the exact fraud category.

    Preserve safety checks and existing alerts.
    """

    result = deepcopy(result)

    if not result.get("success"):
        return result

    result["decision_policy_version"] = POLICY_VERSION

    result["decision_override"] = False

    # Preserve decisions other than manual review
    if result.get("decision") != (
        "Manual Verification Required"
    ):
        return result

    # Never override protective or mixed contexts
    if result.get("safety_context"):
        return result

    if result.get("mixed_context"):
        return result

    # Preserve insufficient feature protection
    if result.get("recognized_features", 0) < (
        MIN_RECOGNIZED_FEATURES
    ):
        return result

    ml_score = result.get("ml_fraud_score")
    risk_score = result.get("risk_score")
    ml_category = result.get("ml_category")

    if ml_score is None or risk_score is None:
        return result

    # Preserve conflicting legitimate predictions
    if ml_category not in FRAUD_CATEGORIES:
        return result

    reasons = set(
        result.get("uncertainty_reasons", [])
    )

    # Only override when the sole uncertainty
    # concerns the exact fraud category
    if reasons != {
        "Low category prediction confidence."
    }:
        return result

    # Require strong combined ML fraud evidence
    if (
        ml_score >= HIGH_ML_FRAUD_THRESHOLD
        and risk_score >= HIGH_ML_FRAUD_THRESHOLD
    ):

        result["decision"] = "Potential Fraud"

        result["decision_override"] = True

        # Do not pretend the exact category is known
        result["category"] = "Unknown"

        result["uncertain"] = True

        result["explanations"].append(
            "The combined ML fraud score is high, "
            "although the exact fraud category "
            "remains uncertain."
        )

        result["recommendations"].append(
            "Avoid payments or disclosure of "
            "confidential information until "
            "the request is independently verified."
        )

        result["recommendations"] = list(
            dict.fromkeys(
                result["recommendations"]
            )
        )

    return result
