"""Streamlit AppTest interaction tests using the local trained model artifacts.

Run from the project root. AppTest simulates Streamlit widgets and reruns;
it is not a real-browser screenshot/download test.
"""
from pathlib import Path

import pytest

from frontend.display_utils import unique_recommendations

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 60
THREAT = "Transfer money immediately to avoid arrest."
NORMAL = "Your class starts at 9 AM tomorrow."
DISAGREEMENT = (
    "They said I was being monitored remotely and could not consult "
    "a lawyer until I paid them."
)


def widget(elements, label):
    matches = [element for element in elements if element.label == label]
    assert len(matches) == 1, f"Expected one widget with label {label!r}"
    return matches[0]


def rerun(at):
    at.run(timeout=TIMEOUT)
    assert not at.exception, str(at.exception)
    return at


@pytest.fixture
def dashboard():
    from streamlit.testing.v1 import AppTest

    return rerun(AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT))


def navigate(at, page):
    widget(at.radio, "Navigation").set_value(page)
    return rerun(at)


def click(at, label):
    widget(at.button, label).click()
    return rerun(at)


def analyse(at, page, message):
    navigate(at, page)
    widget(at.selectbox, "Select an example or enter your own message").select("Enter your own message")
    rerun(at)
    widget(at.text_area, "Enter suspicious content").set_value(message)
    return click(at, "Analyse for Fraud")


def prepare_report(at):
    analyse(at, "WhatsApp Analysis", THREAT)
    navigate(at, "Cybercrime Reporting")
    widget(at.radio, "Have you already lost money?").set_value("Yes")
    widget(at.checkbox, "Date and approximate time of the incident").check()
    widget(at.checkbox, "Suspicious message screenshots").check()
    rerun(at)
    click(at, "Generate Incident Summary")
    report = at.session_state["generated_report"]
    assert report["success"]
    assert "Money Lost: Yes" in report["report_text"]
    assert "[x] Suspicious message screenshots" in report["report_text"]
    return report["report_id"]


def test_home(dashboard):
    at = dashboard
    assert widget(at.radio, "Navigation").value == "Home"
    assert any(m.value == "142/142" for m in at.metric)


@pytest.mark.parametrize("page,message,decision,severity", [
    ("WhatsApp Analysis", THREAT, "Potential Fraud", "Critical"),
    ("SMS Analysis", NORMAL, "Low Detected Risk", "Low"),
    ("Call Analysis", DISAGREEMENT, "Manual Verification Required", "Medium"),
])
def test_message_pages(dashboard, page, message, decision, severity):
    at = analyse(dashboard, page, message)
    saved = at.session_state["last_analysis"]
    assert saved["source"] == page.split()[0]
    assert saved["engine_version"] == "Unified-V4"
    assert saved["decision"] == decision
    assert saved["risk_level"] == severity
    assert widget(at.metric, "Decision Severity").value == severity


def test_invalid_upi(dashboard):
    at = navigate(dashboard, "UPI Security")
    widget(at.text_input, "UPI ID").set_value("invalid-upi")
    click(at, "Analyse UPI Security")
    assert any("Invalid basic UPI ID format" in e.value for e in at.error)
    assert "last_analysis" not in at.session_state


def test_collect_request_without_message(dashboard):
    at = navigate(dashboard, "UPI Security")
    widget(at.text_input, "UPI ID").set_value("request@sample")
    widget(at.number_input, "Transaction Amount (₹)").set_value(1000.0)
    widget(at.selectbox, "Payment Type").select("Collect Request")
    widget(at.text_area, "Accompanying Payment Message (Optional)").set_value("")
    click(at, "Analyse UPI Security")
    result = at.session_state["last_analysis"]
    assert result["decision"] == "Manual Verification Required"
    assert result["risk_score"] == 40
    assert widget(at.metric, "Decision Severity").value == "Medium"
    assert any("not been verified" in e.value for e in at.warning)
    assert "request@sample" not in str(at.session_state["latest_display"])


def test_report_navigation_preserves_selections(dashboard):
    at = dashboard
    report_id = prepare_report(at)
    assessment_id = at.session_state["assessment_id"]
    navigate(at, "Home")
    navigate(at, "Cybercrime Reporting")
    assert at.session_state["assessment_id"] == assessment_id
    assert widget(at.radio, "Have you already lost money?").value == "Yes"
    assert widget(at.checkbox, "Suspicious message screenshots").value is True
    assert at.session_state["generated_report"]["report_id"] == report_id


@pytest.mark.parametrize("change", ["loss", "evidence"])
def test_report_invalidation(dashboard, change):
    at = dashboard
    old_id = prepare_report(at)
    if change == "loss":
        widget(at.radio, "Have you already lost money?").set_value("No")
    else:
        widget(at.checkbox, "Suspicious message screenshots").uncheck()
    rerun(at)
    assert "generated_report" not in at.session_state
    click(at, "Generate Incident Summary")
    report = at.session_state["generated_report"]
    assert report["report_id"] != old_id
    expected = "Money Lost: No" if change == "loss" else "[ ] Suspicious message screenshots"
    assert expected in report["report_text"]


def test_new_assessment_resets_report(dashboard):
    at = dashboard
    prepare_report(at)
    old_id = at.session_state["assessment_id"]
    analyse(at, "SMS Analysis", NORMAL)
    navigate(at, "Cybercrime Reporting")
    assert at.session_state["assessment_id"] != old_id
    assert at.session_state["last_analysis"]["source"] == "SMS"
    assert at.session_state["last_analysis"]["decision"] == "Low Detected Risk"
    assert "generated_report" not in at.session_state
    assert widget(at.radio, "Have you already lost money?").value == "No"
    assert not any(c.value for c in at.checkbox)


@pytest.mark.parametrize("page,title", [
    ("Model Performance", "Model Performance Analytics"),
    ("About Project", "About FraudGuard AI"),
])
def test_information_pages(dashboard, page, title):
    at = navigate(dashboard, page)
    assert any(t.value == title for t in at.title)
    assert len(at.dataframe) >= 1


def test_recommendation_deduplication():
    recommendations = [
        "Verify unexpected requests through independent official channels.",
        "Verify unexpected requests independently.",
        "Never enter a UPI PIN merely to receive money.",
        "Entering a UPI PIN is not required merely to receive money.",
        "Contact your bank.",
        "Contact your bank.",
        "Preserve transaction reference details.",
    ]
    assert list(unique_recommendations(recommendations)) == [
        recommendations[0], recommendations[2], recommendations[4], recommendations[6]
    ]
