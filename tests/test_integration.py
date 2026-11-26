
import pytest

from types import SimpleNamespace

from frontend import session_manager
from core.fraud_detector import analyze_fraud
from security.upi_checker import analyze_upi_payment

from security.reporting import (
    generate_incident_report,
    get_reporting_guidance
)


@pytest.fixture
def session(monkeypatch):

    state = {}

    fake_streamlit = SimpleNamespace(
        session_state=state
    )

    monkeypatch.setattr(
        session_manager,
        "st",
        fake_streamlit
    )

    return state


# Test 1: Assessment persistence
def test_assessment_persistence(session):

    result = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="WhatsApp"
    )

    session_manager.save_assessment(
        result,
        "WhatsApp"
    )

    saved = session_manager.get_saved_assessment(
        "WhatsApp"
    )

    assert saved is not None
    assert saved["category"] == "Digital Arrest"
    assert saved["risk_score"] == result["risk_score"]
    assert "assessment_id" in session


# Test 2: Evidence export
def test_evidence_export():

    analysis = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="WhatsApp"
    )

    guidance = get_reporting_guidance(
        "Digital Arrest",
        money_lost=True
    )

    evidence = guidance["evidence_checklist"]

    report = generate_incident_report(
        analysis,
        money_lost=True,
        evidence_available=evidence[:3]
    )

    text = report["report_text"]

    assert "Money Lost: Yes" in text

    for item in evidence[:3]:

        assert "[x] " + item in text

    for item in evidence[3:]:

        assert "[ ] " + item in text


# Test 3: Reset previous report
def test_identical_assessment_reset(session):

    result = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="WhatsApp"
    )

    session_manager.save_assessment(
        result,
        "WhatsApp"
    )

    previous_id = session["assessment_id"]

    session["generated_report"] = {
        "report_id": "OLD-REPORT"
    }

    session["report_context"] = "old-context"
    session["evidence_0"] = True
    session["report_money_lost"] = "Yes"

    session_manager.save_assessment(
        result,
        "WhatsApp"
    )

    assert session["assessment_id"] != previous_id
    assert "generated_report" not in session
    assert "report_context" not in session
    assert "evidence_0" not in session
    assert "report_money_lost" not in session


# Test 4: Cross-module integration
def test_cross_module_integration(session):

    message_result = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="WhatsApp"
    )

    session_manager.save_assessment(
        message_result,
        "WhatsApp"
    )

    upi_result = analyze_upi_payment(
        "demo@sample",
        5000,
        "Receive Money",
        "Enter your UPI PIN to receive cashback."
    )

    session_manager.save_assessment(
        upi_result,
        "UPI"
    )

    latest = session["last_analysis"]

    assert latest["source"] == "UPI"
    assert latest["category"] == "UPI Fraud"
    assert latest["risk_level"] == "Critical"
    assert latest["risk_score"] == upi_result["risk_score"]

    assert (
        session_manager.get_saved_assessment("WhatsApp")
        is None
    )

    assert (
        session_manager.get_saved_assessment("UPI")
        is not None
    )


# Test 5: Saved message privacy
def test_saved_message_privacy(session):

    message = (
        "Transfer money immediately to avoid arrest."
    )

    result = analyze_fraud(
        message,
        source="Call"
    )

    session_manager.save_assessment(
        result,
        "Call"
    )

    saved = session_manager.get_saved_assessment(
        "Call"
    )

    assert "processed_message" not in saved
    assert message not in str(saved)

    for indicator in saved["indicators"]:

        assert indicator["matched_text"] == (
            "[Not retained in saved result]"
        )


# Test 6: UPI account privacy
def test_saved_upi_privacy(session):

    result = analyze_upi_payment(
        "privateuser@sample",
        5000,
        "Receive Money",
        "Enter your UPI PIN to receive cashback."
    )

    session_manager.save_assessment(
        result,
        "UPI"
    )

    saved = session_manager.get_saved_assessment(
        "UPI"
    )

    assert "upi_id" not in saved
    assert "privateuser@sample" not in str(saved)


# Test 7: Clear incident session
def test_clear_assessment(session):

    result = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="SMS"
    )

    session_manager.save_assessment(
        result,
        "SMS"
    )

    session["evidence_0"] = True

    session["generated_report"] = {
        "report_id": "TEST-REPORT"
    }

    session_manager.clear_assessment()

    assert "last_analysis" not in session
    assert "latest_display" not in session
    assert "assessment_id" not in session
    assert "generated_report" not in session
    assert "evidence_0" not in session
