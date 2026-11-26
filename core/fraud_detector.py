
from ml.predict import predict_fraud
from core.risk_score import assess_risk, get_risk_level
from security.scam_patterns import detect_scam_patterns
from core.decision_policy import apply_decision_policy


VALID_SOURCES = [
    "WhatsApp",
    "SMS",
    "Call",
    "UPI"
]


# Category-specific safety recommendations
CATEGORY_ADVICE = {

    "Digital Arrest": [
        "Do not transfer money under threats of arrest.",
        "End suspicious calls and verify the claim independently.",
        "Contact the police through official channels."
    ],

    "KYC Fraud": [
        "Never disclose your OTP, password or CVV.",
        "Do not follow unsolicited verification links.",
        "Verify KYC requirements through your bank."
    ],

    "Courier Fraud": [
        "Do not pay unexpected penalties to unknown callers.",
        "Verify shipment details through the official courier service.",
        "Do not disclose confidential financial information."
    ],

    "UPI Fraud": [
        "Never enter a UPI PIN merely to receive money.",
        "Do not approve unknown payment requests.",
        "Verify the recipient and payment purpose independently."
    ],

    "Legitimate": [
        "No definitive fraud conclusion can be made.",
        "Verify unexpected requests independently."
    ],

    "Unknown": [
        "Provide additional context for analysis.",
        "Avoid sharing confidential information."
    ]
}


def calculate_hybrid_score(ml_score, rule_score):
    """
    Combine ML and rule scores.

    The result is a heuristic risk indicator,
    not a calibrated fraud probability.
    """

    score = ml_score + (
        (100 - ml_score) * rule_score / 100
    )

    return round(max(0, min(score, 100)), 2)


def _analyze_fraud_v1(message, source="SMS"):
    """
    Main hybrid fraud detection function.
    """

    # Validate source
    if source not in VALID_SOURCES:

        return {
            "success": False,
            "error": "Unsupported input source."
        }

    # Machine learning prediction
    prediction = predict_fraud(message)

    if not prediction["success"]:
        return prediction

    # ML risk assessment
    assessment = assess_risk(prediction)

    # Rule-based analysis
    rules = detect_scam_patterns(message)

    if not rules["success"]:
        return rules

    # Extract model and security results
    ml_score = prediction["fraud_score"]
    rule_score = rules["rule_score"]
    confidence = prediction["confidence"]
    features = prediction["recognized_features"]

    category = prediction["category"]

    uncertainty_reasons = list(
        assessment["uncertainty_reasons"]
    )

    explanations = list(
        rules["explanations"]
    )

    indicator_names = {
        item["indicator"]
        for item in rules["indicators"]
    }

    # Handle unfamiliar text
    if features < 5:

        category = "Unknown"

        if rule_score == 0:

            final_score = None

        else:

            final_score = float(rule_score)

            explanations.append(
                "Assessment is based mainly on "
                "predefined security rules."
            )

    else:

        final_score = calculate_hybrid_score(
            ml_score,
            rule_score
        )

    # High-priority security indicators
    strong_indicators = False

    if "credential_request" in indicator_names:

        final_score = max(final_score or 0, 80)
        strong_indicators = True

        explanations.append(
            "A confidential credential request "
            "requires additional caution."
        )

    if "upi_collect_risk" in indicator_names:

        final_score = max(final_score or 0, 85)
        strong_indicators = True

        explanations.append(
            "Potential misuse of payment "
            "authorization detected."
        )

    if (
        "legal_threat" in indicator_names
        and "payment_demand" in indicator_names
    ):

        final_score = max(final_score or 0, 85)
        strong_indicators = True

        explanations.append(
            "Legal threats combined with payment "
            "demands are significant scam indicators."
        )

    # Handle protective language
    if rules["safety_context"]:

        if rules["mixed_context"]:

            uncertainty_reasons.append(
                "Both protective and suspicious "
                "language were detected."
            )

        elif rule_score == 0:

            explanations.append(
                "The message contains safety-awareness "
                "language rather than an obvious demand."
            )

            if ml_score > 30:

                uncertainty_reasons.append(
                    "Safety language and ML risk "
                    "estimates require contextual review."
                )

    # Handle conflict between ML and security rules
    if category == "Legitimate" and strong_indicators:

        category = "Unknown"

        uncertainty_reasons.append(
            "The ML classification conflicts "
            "with significant security indicators."
        )

    # Handle low category confidence
    if confidence < 50:

        category = "Unknown"

    # Remove repeated uncertainty messages
    uncertainty_reasons = list(
        dict.fromkeys(uncertainty_reasons)
    )

    uncertain = bool(uncertainty_reasons)

    # Determine final risk level
    if final_score is None:

        risk_level = "Insufficient Information"
        decision = "Insufficient Information"

    else:

        final_score = round(
            max(0, min(final_score, 100)), 2
        )

        risk_level = get_risk_level(final_score)

        # Prioritize significant warning signals
        if strong_indicators:

            decision = "Potential Fraud"

        elif uncertain:

            decision = "Manual Verification Required"

        elif final_score > 60:

            decision = "Potential Fraud"

        elif final_score > 30:

            decision = "Manual Verification Required"

        else:

            decision = "Low Detected Risk"

    # Generate recommendations
    recommendations = []

    if strong_indicators or (
        final_score is not None
        and final_score > 60
    ):

        recommendations.append(
            "Do not transfer money, disclose credentials "
            "or act on suspicious payment instructions "
            "until independently verified."
        )

    if uncertain:

        recommendations.append(
            "The system is uncertain. Verify the request "
            "through independent official channels."
        )

    if rules["safety_context"]:

        recommendations.append(
            "Safety-related wording alone does not "
            "establish that a sender is genuine."
        )

    recommendations.extend(
        CATEGORY_ADVICE.get(
            category,
            CATEGORY_ADVICE["Unknown"]
        )
    )

    # Remove duplicate recommendations
    recommendations = list(
        dict.fromkeys(recommendations)
    )

    return {

        "success": True,

        "source": source,

        "category": category,

        "ml_category": prediction["category"],

        "confidence": confidence,

        "ml_fraud_score": ml_score,

        "rule_score": rule_score,

        "risk_score": final_score,

        "risk_level": risk_level,

        "decision": decision,

        "uncertain": uncertain,

        "uncertainty_reasons": uncertainty_reasons,

        "recognized_features": features,

        "indicator_count": rules["indicator_count"],

        "indicators": rules["indicators"],

        "safety_context": rules["safety_context"],

        "mixed_context": rules["mixed_context"],

        "explanations": explanations,

        "recommendations": recommendations,

        "probabilities": prediction["probabilities"],

        "model_version": "V2-Hybrid",

        "disclaimer": (
            "This is an experimental fraud risk assessment. "
            "It cannot guarantee that a message is safe "
            "or fraudulent."
        )
    }


# Independent hybrid engine testing

def analyze_fraud(message, source="SMS"):

    result = _analyze_fraud_v1(
        message,
        source=source
    )

    return apply_decision_policy(result)

if __name__ == "__main__":

    sample_messages = [

        (
            "Transfer money immediately to avoid arrest.",
            "Call"
        ),

        (
            "Your bank account will be blocked. Update KYC now.",
            "SMS"
        ),

        (
            "Pay a penalty to release your parcel.",
            "WhatsApp"
        ),

        (
            "Enter your UPI PIN to receive cashback.",
            "UPI"
        ),

        (
            "Your class starts at 9 AM tomorrow.",
            "WhatsApp"
        ),

        (
            "Never share your UPI PIN with anyone.",
            "SMS"
        ),

        (
            "The bank warns customers about fake verification links.",
            "SMS"
        ),

        (
            "Share your OTP immediately.",
            "Call"
        ),

        (
            "Never share your OTP. Transfer money immediately to avoid arrest.",
            "WhatsApp"
        ),

        (
            "xyz qqq zzzz",
            "SMS"
        ),

        (
            "",
            "SMS"
        )
    ]

    print("=" * 60)
    print("FraudGuard AI - Hybrid Fraud Detection")
    print("=" * 60)

    for message, source in sample_messages:

        result = analyze_fraud(
            message,
            source
        )

        print("\n" + "-" * 60)
        print("Input:", repr(message))
        print("Source:", source)

        if not result["success"]:

            print("Error:", result["error"])
            continue

        print("ML Category:", result["ml_category"])
        print("Final Category:", result["category"])
        print("Confidence:", result["confidence"])
        print("ML Score:", result["ml_fraud_score"])
        print("Rule Score:", result["rule_score"])

        if result["risk_score"] is None:

            print("Hybrid Risk Score: N/A")

        else:

            print(
                f"Hybrid Risk Score: "
                f"{result['risk_score']}/100"
            )

        print("Risk Level:", result["risk_level"])
        print("Decision:", result["decision"])
        print("Uncertain:", result["uncertain"])

        print("\nExplanations:")

        for explanation in result["explanations"]:

            print("-", explanation)

        if result["uncertainty_reasons"]:

            print("\nUncertainty Reasons:")

            for reason in result["uncertainty_reasons"]:

                print("-", reason)

        print("\nSafety Recommendations:")

        for advice in result["recommendations"]:

            print("-", advice)

    print("\n" + "=" * 60)
    print("Hybrid fraud detection tests completed!")
    print("=" * 60)
