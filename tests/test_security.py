
from security.scam_patterns import detect_scam_patterns


# Test 1: Legal threat
def test_legal_threat():

    result = detect_scam_patterns(
        "Transfer money immediately to avoid arrest."
    )

    indicators = {
        item["indicator"]
        for item in result["indicators"]
    }

    assert "legal_threat" in indicators
    assert "payment_demand" in indicators
    assert "urgency" in indicators

    assert result["rule_score"] == 55


# Test 2: Credential request
def test_credential_request():

    result = detect_scam_patterns(
        "Share your OTP immediately."
    )

    indicators = {
        item["indicator"]
        for item in result["indicators"]
    }

    assert "credential_request" in indicators
    assert result["rule_score"] == 40


# Test 3: UPI payment manipulation
def test_upi_collect_risk():

    result = detect_scam_patterns(
        "Enter your UPI PIN to receive cashback."
    )

    indicators = {
        item["indicator"]
        for item in result["indicators"]
    }

    assert "upi_collect_risk" in indicators
    assert result["rule_score"] == 60


# Test 4: URL detection
def test_external_link():

    result = detect_scam_patterns(
        "Visit https://example.com"
    )

    indicators = {
        item["indicator"]
        for item in result["indicators"]
    }

    assert "external_link" in indicators


# Test 5: Protective instructions
def test_protective_language():

    result = detect_scam_patterns(
        "Never share your UPI PIN."
    )

    assert result["safety_context"] is True
    assert result["rule_score"] == 0


# Test 6: Mixed message
def test_mixed_security_context():

    result = detect_scam_patterns(
        "Never share your OTP. "
        "Transfer money immediately to avoid arrest."
    )

    assert result["mixed_context"] is True
    assert result["safety_context"] is True
    assert result["rule_score"] == 55


# Test 7: Legitimate message
def test_normal_message():

    result = detect_scam_patterns(
        "Your class starts at 9 AM tomorrow."
    )

    assert result["indicator_count"] == 0
    assert result["rule_score"] == 0


# Test 8: Rule score limit
def test_rule_score_limit():

    result = detect_scam_patterns(
        "Police will arrest you immediately. "
        "Transfer money. "
        "Share your OTP. "
        "Enter your UPI PIN to receive cashback."
    )

    assert 0 <= result["rule_score"] <= 100


# Test 9: Empty input
def test_empty_security_input():

    result = detect_scam_patterns("")

    assert result["success"] is False


# Test 10: Repeated indicators
def test_duplicate_indicators():

    result = detect_scam_patterns(
        "Transfer money. Transfer money. Transfer money."
    )

    assert result["indicator_count"] == 1
    assert result["rule_score"] == 25
