
from datetime import datetime, timezone
from uuid import uuid4


# Official reporting channels
REPORTING_PORTAL = "https://cybercrime.gov.in/"
CYBERCRIME_HELPLINE = "1930"


# Category-specific guidance
CATEGORY_GUIDANCE = {

    "Digital Arrest": [
        "End the suspicious call.",
        "Do not transfer money under threats.",
        "Verify the claim through official police channels.",
        "Preserve caller details and message evidence."
    ],

    "KYC Fraud": [
        "Do not open unsolicited verification links.",
        "Never disclose OTP, password or CVV.",
        "Contact your bank through official channels.",
        "Secure your banking account if credentials were exposed."
    ],

    "Courier Fraud": [
        "Do not pay unexpected penalties.",
        "Verify shipment details independently.",
        "Contact the official courier company.",
        "Preserve suspicious delivery messages."
    ],

    "UPI Fraud": [
        "Do not approve unexpected payment requests.",
        "Never enter a UPI PIN merely to receive money.",
        "Contact your bank or payment provider.",
        "Preserve transaction reference details."
    ],

    "Legitimate": [
        "The assessment does not establish sender authenticity.",
        "Verify unexpected requests independently.",
        "Report the incident if you have evidence of fraud."
    ],

    "Unknown": [
        "Avoid acting on unverified instructions.",
        "Provide additional context when seeking assistance.",
        "Contact official authorities if fraud is suspected."
    ]
}


def get_reporting_guidance(
    category="Unknown",
    money_lost=False
):
    """
    Return category-specific reporting assistance.

    No complaint is submitted automatically.
    """

    immediate_actions = []

    if money_lost:

        immediate_actions.extend([
            "Immediately contact your bank or payment provider.",
            "Call the financial cyber fraud helpline 1930.",
            "Report the incident at cybercrime.gov.in.",
            "Do not delay reporting while collecting evidence."
        ])

    else:

        immediate_actions.extend([
            "Do not make suspicious payments.",
            "Do not disclose confidential credentials.",
            "Verify the sender independently."
        ])

    immediate_actions.extend(
        CATEGORY_GUIDANCE.get(
            category,
            CATEGORY_GUIDANCE["Unknown"]
        )
    )

    evidence_checklist = [
        "Date and approximate time of the incident",
        "Suspicious message screenshots",
        "Sender phone number or account details",
        "Suspicious website address if available",
        "Transaction reference or UTR if applicable",
        "Payment receipt if applicable",
        "Amount lost if applicable"
    ]

    return {

        "category": category,

        "money_lost": money_lost,

        "helpline": CYBERCRIME_HELPLINE,

        "portal": REPORTING_PORTAL,

        "immediate_actions": list(
            dict.fromkeys(immediate_actions)
        ),

        "evidence_checklist": evidence_checklist,

        "privacy_warning": (
            "Never include actual OTPs, UPI PINs, "
            "passwords or card security codes "
            "in this application."
        ),

        "disclaimer": (
            "FraudGuard AI provides reporting assistance only. "
            "No official complaint has been submitted."
        )
    }


def generate_incident_report(
    analysis,
    money_lost=False,
    evidence_available=None
):
    """
    Generate a privacy-conscious text incident summary.

    Raw messages and UPI IDs are not included.
    No report is automatically saved or transmitted.
    """

    if not isinstance(analysis, dict):

        raise TypeError(
            "Analysis must be a dictionary."
        )

    if not analysis.get("success", False):

        raise ValueError(
            "A successful analysis is required."
        )

    # Support both message and UPI analysis results
    category = analysis.get("category")

    if not category:

        nested_analysis = (
            analysis.get("message_analysis") or {}
        )

        category = nested_analysis.get(
            "category", "Unknown"
        )

    source = analysis.get(
        "source",
        "UPI" if "upi_id" in analysis else "Unknown"
    )

    risk_score = analysis.get("risk_score")

    risk_level = analysis.get(
        "risk_level", "Unknown"
    )

    decision = analysis.get(
        "decision", "Unknown"
    )

    guidance = get_reporting_guidance(
        category,
        money_lost
    )

    # Local reference, not an official complaint number
    report_id = (
        "FG-" + uuid4().hex[:12].upper()
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    score_text = (
        f"{risk_score}/100"
        if risk_score is not None
        else "Not available"
    )

    lines = [

        "=" * 55,

        "FRAUDGUARD AI - INCIDENT SUMMARY",

        "=" * 55,

        f"Local Reference: {report_id}",

        f"Generated At: {generated_at}",

        f"Source: {source}",

        f"Predicted Category: {category}",

        f"Risk Score: {score_text}",

        f"Score Basis: {analysis.get('score_basis', 'Legacy heuristic score')}",

        f"Engine: {analysis.get('engine_version', 'Legacy')}",

        "Category is a provisional model estimate.",

        f"Risk Level: {risk_level}",

        f"Assessment: {decision}",

        f"Money Lost: {'Yes' if money_lost else 'No'}",

        "",

        "IMMEDIATE ACTIONS"
    ]

    for action in guidance["immediate_actions"]:

        lines.append("- " + action)

    lines.extend([
        "",
        "EVIDENCE CHECKLIST"
    ])

    available = set(evidence_available or [])

    for item in guidance["evidence_checklist"]:

        marker = "[x] " if item in available else "[ ] "

        lines.append(marker + item)

    lines.extend([

        "",

        "OFFICIAL REPORTING CHANNELS",

        f"Financial Fraud Helpline: {CYBERCRIME_HELPLINE}",

        f"Reporting Portal: {REPORTING_PORTAL}",

        "",

        "PRIVACY NOTICE",

        guidance["privacy_warning"],

        "Raw messages and account identifiers "
        "are excluded from this generated summary.",

        "",

        "DISCLAIMER",

        "This is an experimental AI assessment.",

        "The risk score is not a verified probability of fraud.",

        "This document is not an FIR or official complaint.",

        "No complaint has been submitted automatically.",

        "=" * 55
    ])

    return {
        "success": True,
        "report_id": report_id,
        "filename": f"{report_id}.txt",
        "report_text": "\n".join(lines)
    }


# Independent module testing
if __name__ == "__main__":

    from core.fraud_detector import analyze_fraud

    print("=" * 60)
    print("FraudGuard AI - Cybercrime Reporting Module")
    print("=" * 60)

    # Test 1: Normal reporting guidance
    guidance = get_reporting_guidance(
        "Digital Arrest",
        money_lost=False
    )

    print("\nTEST 1: Reporting Guidance")

    print("Helpline:", guidance["helpline"])
    print("Portal:", guidance["portal"])

    for action in guidance["immediate_actions"]:

        print("-", action)

    # Test 2: Financial loss response
    guidance = get_reporting_guidance(
        "UPI Fraud",
        money_lost=True
    )

    print("\nTEST 2: Financial Loss Guidance")

    for action in guidance["immediate_actions"]:

        print("-", action)

    # Test 3: Generate incident summary
    analysis = analyze_fraud(
        "Transfer money immediately to avoid arrest.",
        source="Call"
    )

    report = generate_incident_report(
        analysis,
        money_lost=False
    )

    print("\nTEST 3: Incident Report Generation")

    print("Filename:", report["filename"])

    print("\n" + report["report_text"])

    print("\n" + "=" * 60)
    print("Reporting module tests completed!")
    print("=" * 60)
