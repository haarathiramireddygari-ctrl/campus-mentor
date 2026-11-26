
import pytest

from security.scam_patterns import detect_scam_patterns


# Test enhanced fraud indicators
@pytest.mark.parametrize(
    "message,expected_indicators",
    [
        (
            "Remain on the video call until you "
            "pay the investigation charge.",
            {"legal_threat", "payment_demand"}
        ),
        (
            "Pay this caller directly to clear "
            "a customs violation.",
            {"legal_threat", "payment_demand"}
        ),
        (
            "Your shipment has been seized. "
            "Send a security payment to this caller.",
            {"legal_threat", "payment_demand"}
        ),
        (
            "The caller requests my card security "
            "number for verification.",
            {"credential_request"}
        ),
        (
            "A prize notification demands an "
            "upfront processing charge.",
            {"payment_demand"}
        ),
        (
            "A refund is available if you authorize "
            "a debit from your account.",
            {"upi_collect_risk"}
        )
    ]
)
def test_enhanced_fraud_indicators(
    message,
    expected_indicators
):

    result = detect_scam_patterns(message)

    detected = {
        item["indicator"]
        for item in result["indicators"]
    }

    assert result["success"] is True

    assert expected_indicators.issubset(detected)

    assert result["rule_score"] > 0


# Test protective security advice
@pytest.mark.parametrize(
    "message",
    [
        "Never share your password with anyone.",

        "The bank advises customers not to open "
        "unfamiliar links.",

        "The bank recommends reporting unexpected "
        "requests for confidential credentials."
    ]
)
def test_enhanced_safety_context(message):

    result = detect_scam_patterns(message)

    assert result["success"] is True
    assert result["safety_context"] is True
    assert result["rule_score"] == 0


# Test mixed context handling
@pytest.mark.parametrize(
    "message",
    [
        "Never share your OTP, but transfer money "
        "immediately to avoid arrest.",

        "Never share your OTP\n"
        "Transfer money immediately to avoid arrest."
    ]
)
def test_enhanced_mixed_context(message):

    result = detect_scam_patterns(message)

    assert result["safety_context"] is True
    assert result["mixed_context"] is True
    assert result["rule_score"] > 0


# Ensure ordinary messages do not trigger new rules
@pytest.mark.parametrize(
    "message",
    [
        "Your package will arrive tomorrow.",

        "The college library is open today.",

        "Your receipt is available in the student portal."
    ]
)
def test_enhanced_legitimate_messages(message):

    result = detect_scam_patterns(message)

    assert result["success"] is True
    assert result["rule_score"] == 0
    assert result["indicator_count"] == 0
