
import pytest

from security.upi_checker import (
    validate_upi_id,
    validate_amount,
    analyze_upi_payment
)


# Test valid UPI address formats
@pytest.mark.parametrize(
    "upi_id",
    [
        "demo@sample",
        "user123@bank",
        "user.name@provider",
        "user_name@provider"
    ]
)
def test_valid_upi_id(upi_id):

    valid, status = validate_upi_id(upi_id)

    assert valid is True


# Test invalid UPI address formats
@pytest.mark.parametrize(
    "upi_id",
    [
        "",
        "invalid-upi",
        "@provider",
        "user@",
        "user@@provider",
        "user name@provider",
        None
    ]
)
def test_invalid_upi_id(upi_id):

    valid, status = validate_upi_id(upi_id)

    assert valid is False


# Test valid payment amounts
@pytest.mark.parametrize(
    "amount",
    [1, 100, 250.50, "500.25"]
)
def test_valid_amount(amount):

    valid, status = validate_amount(amount)

    assert valid is True


# Test invalid payment amounts
@pytest.mark.parametrize(
    "amount",
    [
        0,
        -100,
        "abc",
        "NaN",
        "Infinity",
        None,
        "10.999"
    ]
)
def test_invalid_amount(amount):

    valid, status = validate_amount(amount)

    assert valid is False


# Test cashback scam
def test_cashback_scam():

    result = analyze_upi_payment(
        "demo@sample",
        5000,
        "Receive Money",
        "Enter your UPI PIN to receive cashback."
    )

    assert result["success"] is True
    assert result["risk_level"] == "Critical"
    assert result["decision"] == "Potential Fraud"


# Test fraudulent refund request
def test_refund_collect_scam():

    result = analyze_upi_payment(
        "demo@sample",
        1000,
        "Collect Request",
        "Approve this collect request to receive your refund."
    )

    assert result["success"] is True
    assert result["decision"] == "Potential Fraud"


# Test ordinary payment
def test_normal_payment():

    result = analyze_upi_payment(
        "demo@sample",
        250,
        "Send Money",
        "Payment for college books."
    )

    assert result["success"] is True
    assert result["account_verified"] is False

    assert (
        result["decision"]
        == "Manual Verification Required"
    )


# Test missing message
def test_missing_payment_context():

    result = analyze_upi_payment(
        "demo@sample",
        1000,
        "Receive Money",
        ""
    )

    assert result["risk_score"] is None

    assert (
        result["decision"]
        == "Insufficient Information"
    )


# Test collect request without context
def test_collect_request_without_message():

    result = analyze_upi_payment(
        "demo@sample",
        500,
        "Collect Request",
        ""
    )

    assert result["risk_score"] == 40
    assert result["risk_level"] == "Medium"

    assert (
        result["decision"]
        == "Manual Verification Required"
    )


# Test unsupported payment type
def test_invalid_payment_type():

    result = analyze_upi_payment(
        "demo@sample",
        500,
        "Unknown Payment",
        ""
    )

    assert result["success"] is False


# Test invalid payment message
def test_invalid_payment_message():

    result = analyze_upi_payment(
        "demo@sample",
        500,
        "Send Money",
        None
    )

    assert result["success"] is False


# Test account verification limitation
def test_account_not_verified():

    result = analyze_upi_payment(
        "demo@sample",
        100,
        "Send Money",
        ""
    )

    assert result["upi_format_valid"] is True
    assert result["account_verified"] is False


# Test score boundaries
def test_upi_risk_score_range():

    result = analyze_upi_payment(
        "demo@sample",
        5000,
        "Receive Money",
        "Enter your UPI PIN to receive cashback."
    )

    assert 0 <= result["risk_score"] <= 100
