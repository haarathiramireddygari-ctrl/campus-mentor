
import pytest

from core.decision_policy import (
    apply_decision_policy
)


@pytest.fixture
def assessment():

    return {
        "success": True,
        "decision": "Manual Verification Required",
        "category": "Unknown",
        "ml_category": "Digital Arrest",
        "ml_fraud_score": 90.0,
        "risk_score": 90.0,
        "confidence": 45.0,
        "recognized_features": 100,
        "uncertain": True,
        "uncertainty_reasons": [
            "Low category prediction confidence."
        ],
        "safety_context": False,
        "mixed_context": False,
        "explanations": [],
        "recommendations": []
    }


# High overall fraud evidence
def test_high_fraud_score(assessment):

    result = apply_decision_policy(assessment)

    assert result["decision"] == "Potential Fraud"
    assert result["decision_override"] is True
    assert result["category"] == "Unknown"
    assert result["uncertain"] is True


# Test the exact threshold
def test_threshold_boundary(assessment):

    assessment["ml_fraud_score"] = 85.0
    assessment["risk_score"] = 85.0

    result = apply_decision_policy(assessment)

    assert result["decision"] == "Potential Fraud"


# Verify that safety protections remain active
@pytest.mark.parametrize(
    "field,value",
    [
        ("safety_context", True),
        ("mixed_context", True),
        ("recognized_features", 4),
        ("ml_category", "Legitimate"),
        ("ml_fraud_score", 84.99),
        ("risk_score", 84.99),
        ("ml_fraud_score", None),
        ("risk_score", None),
        (
            "uncertainty_reasons",
            ["Additional uncertainty detected."]
        )
    ]
)
def test_preserve_safety_checks(
    assessment,
    field,
    value
):

    assessment[field] = value

    result = apply_decision_policy(assessment)

    assert result["decision"] == (
        "Manual Verification Required"
    )

    assert result["decision_override"] is False


# Preserve existing low-risk decisions
def test_preserve_low_risk(assessment):

    assessment["decision"] = "Low Detected Risk"

    result = apply_decision_policy(assessment)

    assert result["decision"] == "Low Detected Risk"
    assert result["decision_override"] is False


# Preserve failed input validation
def test_invalid_assessment():

    assessment = {
        "success": False,
        "error": "Invalid input."
    }

    result = apply_decision_policy(assessment)

    assert result == assessment


# Ensure original assessment is not modified
def test_original_result_unchanged(assessment):

    apply_decision_policy(assessment)

    assert assessment["decision"] == (
        "Manual Verification Required"
    )

    assert assessment["explanations"] == []
    assert assessment["recommendations"] == []
