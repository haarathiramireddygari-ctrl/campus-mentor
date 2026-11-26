"""Application-facing adapter for the evaluated Unified V4 policy."""
from core.unified_detector import analyze_unified, ENGINE_VERSION
from core.fraud_detector import CATEGORY_ADVICE


def decision_risk_level(decision, score):
    """Severity follows the decision, not the legacy numerical bands."""
    if decision == "Potential Fraud":
        return "Critical" if score is not None and score > 80 else "High"
    if decision == "Manual Verification Required":
        return "Medium"
    if decision == "Low Detected Risk":
        return "Low"
    return "Insufficient Information"


def analyze_application(message, source="SMS"):
    unified = analyze_unified(message, source=source)
    if not unified["success"]:
        return unified
    legacy = unified["legacy_assessment"]
    decision = unified["decision"]
    score = unified["binary_fraud_score"]
    provisional = unified["provisional_category"]
    # A category estimate must not contradict the final safety decision.
    category = provisional
    if (decision == "Low Detected Risk" and provisional != "Legitimate") or (
        decision != "Low Detected Risk" and provisional == "Legitimate"
    ):
        category = "Unknown"
    reasons = list(legacy.get("uncertainty_reasons", []))
    if decision == "Manual Verification Required":
        reasons.append(unified["decision_reason"])
    if category != provisional:
        reasons.append("Category estimate conflicts with the final decision.")
    recommendations = ["Verify unexpected requests through independent official channels."]
    if decision != "Low Detected Risk":
        recommendations.insert(0, "Do not transfer money or disclose credentials until independently verified.")
    else:
        recommendations.append("Low detected risk does not establish sender authenticity.")
    if legacy.get("safety_context"):
        recommendations.append("Safety-related wording alone does not establish sender authenticity.")
    recommendations.extend(CATEGORY_ADVICE.get(category, CATEGORY_ADVICE["Unknown"]))
    return {
        "success": True,
        "source": unified["source"],
        "engine_version": ENGINE_VERSION,
        "model_version": ENGINE_VERSION,
        "category": category,
        "provisional_category": provisional,
        "confidence": unified["category_confidence"],
        "binary_fraud_score": score,
        "legacy_risk_score": unified["legacy_risk_score"],
        "ml_fraud_score": legacy["ml_fraud_score"],
        "rule_score": unified["rule_score"],
        "risk_score": score,
        "score_basis": "Binary model score (uncalibrated)",
        "risk_level": decision_risk_level(decision, score),
        "decision": decision,
        "decision_reason": unified["decision_reason"],
        "uncertain": bool(reasons) or category == "Unknown",
        "uncertainty_reasons": list(dict.fromkeys(reasons)),
        "indicator_count": legacy["indicator_count"],
        "indicators": legacy["indicators"],
        "explanations": [unified["decision_reason"]] + ["Legacy evidence: " + x for x in legacy["explanations"]],
        "recommendations": list(dict.fromkeys(recommendations)),
        "probabilities": legacy["probabilities"],
        "disclaimer": unified["disclaimer"]
    }
