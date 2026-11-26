
def get_risk_level(score):
    """
    Convert an estimated ML fraud score into a risk level.
    """

    if not 0 <= score <= 100:
        raise ValueError("Risk score must be between 0 and 100.")

    if score <= 30:
        return "Low"

    elif score <= 60:
        return "Medium"

    elif score <= 80:
        return "High"

    else:
        return "Critical"


def assess_risk(prediction):
    """
    Assess ML predictions and handle uncertain inputs.
    """

    # Handle invalid inputs
    if not prediction["success"]:
        return {
            "success": False,
            "error": prediction["error"]
        }

    # Extract model results
    score = float(prediction["fraud_score"])
    confidence = float(prediction["confidence"])
    features = prediction["recognized_features"]
    category = prediction["category"]

    # Initial uncertainty indicators
    uncertainty_reasons = []

    # Check unfamiliar input
    if features < 5:

        return {
            "success": True,
            "risk_score": None,
            "ml_fraud_score": score,
            "risk_level": "Insufficient Information",
            "category": "Unknown",
            "confidence": confidence,
            "uncertain": True,
            "uncertainty_reasons": [
                "Very few recognized text features."
            ],
            "recommendation": (
                "Provide a more detailed message "
                "or additional context."
            ),
            "model_version": prediction["model_version"]
        }

    # Check category confidence
    if confidence < 50:

        uncertainty_reasons.append(
            "Low category prediction confidence."
        )

    # Check conflicting interpretation
    if category == "Legitimate" and score > 50:

        uncertainty_reasons.append(
            "Combined fraud score conflicts with "
            "the predicted legitimate category."
        )

    # Calculate risk level
    risk_level = get_risk_level(score)

    uncertain = len(uncertainty_reasons) > 0

    # Generate recommendations
    if uncertain:

        recommendation = (
            "The model is uncertain. Verify the message "
            "through independent official channels "
            "before taking any action."
        )

    elif risk_level == "Critical":

        recommendation = (
            "Do not transfer money, share credentials "
            "or follow suspicious links. "
            "Verify the request independently."
        )

    elif risk_level == "High":

        recommendation = (
            "Potential fraud indicators detected. "
            "Avoid acting until the request is verified."
        )

    elif risk_level == "Medium":

        recommendation = (
            "Proceed with caution and verify "
            "the sender independently."
        )

    else:

        recommendation = (
            "Few fraud indicators detected. "
            "This does not guarantee the message is safe."
        )

    # Return structured risk assessment
    return {
        "success": True,
        "risk_score": round(score, 2),
        "ml_fraud_score": round(score, 2),
        "risk_level": risk_level,
        "category": category,
        "confidence": confidence,
        "uncertain": uncertain,
        "uncertainty_reasons": uncertainty_reasons,
        "recommendation": recommendation,
        "model_version": prediction["model_version"]
    }


# Independent module test
if __name__ == "__main__":

    from ml.predict import predict_fraud

    sample_messages = [

        "Transfer money immediately to avoid arrest.",

        "Your bank account will be blocked. Update KYC now.",

        "Pay a penalty to release your parcel.",

        "Enter your UPI PIN to receive cashback.",

        "Your class starts at 9 AM tomorrow.",

        "Never disclose your UPI PIN to anyone.",

        "The bank warns customers about fake verification links.",

        "Urgnt notification. Verify your acc0unt now.",

        "xyz qqq zzzz",

        ""
    ]

    print("=" * 60)
    print("FraudGuard AI - Risk Assessment Module")
    print("=" * 60)

    for message in sample_messages:

        prediction = predict_fraud(message)

        result = assess_risk(prediction)

        print("\n" + "-" * 60)
        print("Input:", repr(message))

        if not result["success"]:

            print("Error:", result["error"])
            continue

        print("Category:", result["category"])

        print("ML Fraud Score:", result["ml_fraud_score"])

        if result["risk_score"] is None:

            print("Risk Score: N/A")

        else:

            print(
                f"Risk Score: {result['risk_score']}/100"
            )

        print("Risk Level:", result["risk_level"])

        print(
            f"Confidence: {result['confidence']}%"
        )

        print("Uncertain:", result["uncertain"])

        if result["uncertainty_reasons"]:

            print("Uncertainty Reasons:")

            for reason in result["uncertainty_reasons"]:

                print("-", reason)

        print(
            "Recommendation:",
            result["recommendation"]
        )

    print("\n" + "=" * 60)
    print("Risk assessment tests completed!")
    print("=" * 60)
