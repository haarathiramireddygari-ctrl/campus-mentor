"""Application contract tests. Existing trained-model parity tests stay unchanged."""
from types import SimpleNamespace

import pytest

from core import application_detector as app
from frontend import session_manager
from security.upi_checker import analyze_upi_payment
from security.reporting import generate_incident_report


@pytest.fixture
def sample(monkeypatch):
    legacy = {
        "ml_fraud_score": 62.09,
        "indicator_count": 0,
        "indicators": [],
        "explanations": [],
        "uncertainty_reasons": [],
        "probabilities": {"Legitimate": 38.0, "Digital Arrest": 62.0},
        "safety_context": False,
    }
    result = {
        "success": True,
        "source": "Call",
        "binary_fraud_score": 28.4659,
        "legacy_risk_score": 62.09,
        "rule_score": 0,
        "decision": "Manual Verification Required",
        "decision_reason": "Binary and existing hybrid scores disagree.",
        "provisional_category": "Unknown",
        "category_confidence": 38.0,
        "legacy_assessment": legacy,
        "disclaimer": "Experimental AI assessment. Low risk does not guarantee safety.",
    }
    monkeypatch.setattr(app, "analyze_unified", lambda message, source: {**result, "source": source})
    return result


def test_disagreement_severity(sample):
    result = app.analyze_application("Example", "Call")
    assert result["decision"] == sample["decision"]
    assert result["risk_level"] == "Medium"
    assert result["risk_score"] == sample["binary_fraud_score"]
    assert result["legacy_risk_score"] == 62.09
    assert result["decision_reason"] in result["uncertainty_reasons"]
    assert result["engine_version"] == "Unified-V4"


def test_alert_with_low_binary_score(sample):
    sample["decision"] = "Potential Fraud"
    sample["decision_reason"] = "Strong contextual security indicators."
    sample["provisional_category"] = "Legitimate"
    result = app.analyze_application("Example", "Call")
    assert result["risk_level"] == "High"
    assert result["category"] == "Unknown"
    assert result["provisional_category"] == "Legitimate"


def test_low_risk_category_conflict(sample):
    sample["decision"] = "Low Detected Risk"
    sample["provisional_category"] = "Digital Arrest"
    result = app.analyze_application("Example", "Call")
    assert result["risk_level"] == "Low"
    assert result["category"] == "Unknown"
    assert result["uncertain"]


def test_invalid_input_passthrough(monkeypatch):
    expected = {"success": False, "error": "Invalid input"}
    monkeypatch.setattr(app, "analyze_unified", lambda *a, **kw: expected)
    assert app.analyze_application("", "SMS") == expected


def test_session_and_report(sample, monkeypatch):
    state = {}
    monkeypatch.setattr(session_manager, "st", SimpleNamespace(session_state=state))
    sample["legacy_assessment"]["indicators"] = [
        {"indicator": "external_link", "weight": 5, "matched_text": "PRIVATE-STRING"}
    ]
    result = app.analyze_application("Example", "Call")
    session_manager.save_assessment(result, "Call")
    saved = session_manager.get_saved_assessment("Call")
    assert saved["binary_fraud_score"] == result["binary_fraud_score"]
    assert saved["decision_reason"] == result["decision_reason"]
    assert "PRIVATE-STRING" not in str(state)
    report = generate_incident_report(state["last_analysis"])
    assert "Engine: Unified-V4" in report["report_text"]
    assert "Score Basis: Binary model score" in report["report_text"]
    assert "PRIVATE-STRING" not in report["report_text"]
    state["generated_report"] = {"old": True}
    session_manager.clear_assessment()
    assert not state


def test_upi_manual_review_propagates(sample):
    result = analyze_upi_payment("demo@sample", 100, "Send Money", "Example")
    assert result["decision"] == "Manual Verification Required"
    assert result["risk_level"] == "Medium"
    assert result["message_analysis"]["engine_version"] == "Unified-V4"
    assert not result["account_verified"]


def test_upi_transaction_floor(sample):
    sample["legacy_assessment"]["indicators"] = [
        {"indicator": "upi_collect_risk", "weight": 30, "matched_text": "test"}
    ]
    sample["decision"] = "Potential Fraud"
    result = analyze_upi_payment("demo@sample", 100, "Collect Request", "Example")
    assert result["risk_score"] == 85
    assert result["decision"] == "Potential Fraud"
    assert result["risk_level"] == "Critical"


def test_upi_without_message():
    result = analyze_upi_payment("demo@sample", 100, "Receive Money", "")
    assert result["risk_score"] is None
    assert result["decision"] == "Insufficient Information"
    assert result["message_analysis"] is None


@pytest.mark.parametrize("dataset_name", ["hybrid_development_100", "independent_holdout_100"])
def test_application_parity(dataset_name):
    """Run with the actual models and recorded reports in the user's project."""
    from pathlib import Path
    import pandas as pd
    root = Path(__file__).resolve().parent.parent
    dataset = pd.read_csv(root / "data" / f"{dataset_name}.csv")
    previous = pd.read_csv(root / "reports" / "unified_v4" / dataset_name / "unified_predictions.csv")
    assert len(dataset) == len(previous) == 100
    assert dataset["message"].equals(previous["message"])
    assert dataset["source"].equals(previous["source"])
    assert dataset["category"].equals(previous["actual_category"])
    for i, row in dataset.iterrows():
        result = app.analyze_application(row["message"], row["source"])
        assert result["success"]
        assert result["decision"] == previous.loc[i, "unified_decision"]
        assert result["decision_reason"] == previous.loc[i, "decision_reason"]
        assert result["binary_fraud_score"] == pytest.approx(previous.loc[i, "binary_score"], abs=0.00011)
