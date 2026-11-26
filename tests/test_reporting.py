
import pytest
from datetime import datetime

from security.reporting import (
    get_reporting_guidance,
    generate_incident_report
)

from core.fraud_detector import analyze_fraud
from security.upi_checker import analyze_upi_payment


# Test official reporting information
def test_reporting_channels():

    result = get_reporting_guidance()

    assert result["helpline"] == "1930"

    assert (
        result["portal"]
        == "https://cybercrime.gov.in/"
    )


# Test category-specific guidance
@pytest.mark.parametrize(
    "category",
    [
        "Digital Arrest",
        "KYC Fraud",
        "Courier Fraud",
        "UPI Fraud",
        "Legitimate",
        "Unknown"
    ]
)
def test_category_guidance(category):

    result = get_reporting_guidance(category)

    assert len(result["immediate_actions"]) > 0
    assert len(result["evidence_checklist"]) > 0
    assert result["category"] == category


# Test financial loss response
def test_financial_loss_guidance():

    result = get_reporting_guidance(
        "UPI Fraud",
        money_lost=True
    )

    actions = " ".join(result["immediate_actions"])

    assert "1930" in actions
    assert "bank" in actions
    assert result["money_lost"] is True


# Test evidence checklist
def test_evidence_checklist():

    result = get_reporting_guidance()

    assert len(result["evidence_checklist"]) == 7


# Test report generation
def test_generate_report():

    analysis = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        "Call"
    )

    report = generate_incident_report(analysis)

    assert report["success"] is True
    assert report["filename"].endswith(".txt")
    assert report["report_id"].startswith("FG-")

    assert (
        "Digital Arrest"
        in report["report_text"]
    )


# Test unique local reference
def test_unique_report_id():

    analysis = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        "Call"
    )

    report1 = generate_incident_report(analysis)
    report2 = generate_incident_report(analysis)

    assert report1["report_id"] != report2["report_id"]


# Test timezone-aware timestamp
def test_report_timestamp():

    analysis = {
        "success": True,
        "category": "Unknown"
    }

    report = generate_incident_report(analysis)

    timestamp = next(
        line.split(": ", 1)[1]
        for line in report["report_text"].splitlines()
        if line.startswith("Generated At: ")
    )

    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None


# Test report disclaimer
def test_report_disclaimer():

    analysis = {
        "success": True,
        "category": "Digital Arrest"
    }

    report = generate_incident_report(analysis)

    assert (
        "not an FIR or official complaint"
        in report["report_text"]
    )

    assert (
        "No complaint has been submitted automatically."
        in report["report_text"]
    )


# Test rejection of invalid analysis
def test_invalid_analysis():

    with pytest.raises(ValueError):

        generate_incident_report({
            "success": False
        })


# Test incorrect input type
def test_invalid_analysis_type():

    with pytest.raises(TypeError):

        generate_incident_report("Invalid input")


# Test privacy of raw message
def test_raw_message_privacy():

    message = (
        "Transfer money immediately to avoid arrest."
    )

    analysis = analyze_fraud(message, "Call")

    report = generate_incident_report(analysis)

    assert message not in report["report_text"]


# Test UPI account privacy
def test_upi_report_privacy():

    analysis = analyze_upi_payment(
        "privateuser@sample",
        5000,
        "Receive Money",
        "Enter your UPI PIN to receive cashback."
    )

    report = generate_incident_report(
        analysis,
        money_lost=True
    )

    assert report["success"] is True

    assert (
        "privateuser@sample"
        not in report["report_text"]
    )

    assert "Money Lost: Yes" in report["report_text"]


# Test report without numerical risk score
def test_report_without_risk_score():

    analysis = {
        "success": True,
        "category": "Unknown",
        "risk_score": None,
        "risk_level": "Insufficient Information"
    }

    report = generate_incident_report(analysis)

    assert (
        "Risk Score: Not available"
        in report["report_text"]
    )


# Test financial loss report
def test_financial_loss_report():

    analysis = {
        "success": True,
        "category": "UPI Fraud"
    }

    report = generate_incident_report(
        analysis,
        money_lost=True
    )

    assert "Money Lost: Yes" in report["report_text"]
    assert "1930" in report["report_text"]
