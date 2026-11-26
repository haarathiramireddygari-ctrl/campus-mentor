
import pytest

from core.preprocessing import (
    preprocess_text,
    validate_input
)

from core.risk_score import get_risk_level

from core.fraud_detector import (
    analyze_fraud,
    calculate_hybrid_score
)


# Test 1: Text normalization
def test_preprocessing():

    result = preprocess_text(
        "  URGENT!!!   Update KYC.  "
    )

    assert result == "urgent!!! update kyc."


# Test 2: Preserve URLs
def test_preserve_url():

    result = preprocess_text(
        "Visit https://example.com"
    )

    assert "https://example.com" in result


# Test 3: Empty input validation
def test_empty_input():

    valid, status = validate_input("")

    assert valid is False


# Test 4: Short input validation
def test_short_input():

    valid, status = validate_input("Hi")

    assert valid is False


# Test 5: Digital arrest detection
def test_digital_arrest():

    result = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        "Call"
    )

    assert result["success"] is True
    assert result["category"] == "Digital Arrest"
    assert result["risk_level"] == "Critical"
    assert result["decision"] == "Potential Fraud"


# Test 6: KYC fraud detection
def test_kyc_fraud():

    result = analyze_fraud(
        "Your bank account will be blocked. Update KYC now.",
        "SMS"
    )

    assert result["category"] == "KYC Fraud"
    assert result["risk_score"] > 80


# Test 7: Courier fraud detection
def test_courier_fraud():

    result = analyze_fraud(
        "Pay a penalty to release your parcel.",
        "WhatsApp"
    )

    assert result["category"] == "Courier Fraud"
    assert result["decision"] == "Potential Fraud"


# Test 8: UPI fraud detection
def test_upi_fraud():

    result = analyze_fraud(
        "Enter your UPI PIN to receive cashback.",
        "UPI"
    )

    assert result["category"] == "UPI Fraud"
    assert result["risk_score"] > 80


# Test 9: Legitimate message
def test_legitimate_message():

    result = analyze_fraud(
        "Your class starts at 9 AM tomorrow.",
        "SMS"
    )

    assert result["risk_level"] == "Low"
    assert result["decision"] == "Low Detected Risk"


# Test 10: Protective language
def test_safety_warning():

    result = analyze_fraud(
        "Never share your UPI PIN with anyone.",
        "SMS"
    )

    assert result["safety_context"] is True
    assert result["rule_score"] == 0
    assert result["decision"] != "Potential Fraud"


# Test 11: Unknown input
def test_unknown_input():

    result = analyze_fraud(
        "xyz qqq zzzz",
        "SMS"
    )

    assert result["category"] == "Unknown"
    assert result["risk_score"] is None
    assert result["uncertain"] is True

    assert (
        result["decision"]
        == "Insufficient Information"
    )


# Test 12: Unsupported source
def test_invalid_source():

    result = analyze_fraud(
        "Sample message",
        "UnknownSource"
    )

    assert result["success"] is False


# Test 13: Empty message rejection
def test_empty_message_rejection():

    result = analyze_fraud("", "SMS")

    assert result["success"] is False


# Test 14: Risk level boundaries
@pytest.mark.parametrize(
    "score, expected",
    [
        (0, "Low"),
        (30, "Low"),
        (31, "Medium"),
        (60, "Medium"),
        (61, "High"),
        (80, "High"),
        (81, "Critical"),
        (100, "Critical")
    ]
)
def test_risk_boundaries(score, expected):

    assert get_risk_level(score) == expected


# Test 15: Hybrid score calculation
def test_hybrid_score():

    result = calculate_hybrid_score(80, 50)

    assert result == 90


# Test 16: Hybrid score maximum
def test_hybrid_score_maximum():

    result = calculate_hybrid_score(100, 100)

    assert result == 100


# Test 17: Risk score range
def test_risk_score_range():

    result = analyze_fraud(
        "Share your OTP immediately.",
        "Call"
    )

    assert 0 <= result["risk_score"] <= 100


# Test 18: Mixed context
def test_mixed_context():

    result = analyze_fraud(
        "Never share your OTP. "
        "Transfer money immediately to avoid arrest.",
        "WhatsApp"
    )

    assert result["mixed_context"] is True
    assert result["uncertain"] is True
    assert result["decision"] == "Potential Fraud"
