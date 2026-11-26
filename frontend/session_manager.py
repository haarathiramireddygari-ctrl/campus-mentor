
import streamlit as st

from uuid import uuid4
from copy import deepcopy


def clear_assessment():

    keys = [
        "last_analysis",
        "latest_display",
        "assessment_id",
        "generated_report",
        "report_context",
        "report_selection",
        "report_money_lost"
    ]

    for key in keys:

        st.session_state.pop(key, None)

    # Remove evidence from the previous incident
    for key in list(st.session_state.keys()):

        if key.startswith("evidence_"):

            st.session_state.pop(key, None)


def sanitize_message_result(result):

    fields = [
        "success",
        "engine_version",
        "score_basis",
        "binary_fraud_score",
        "legacy_risk_score",
        "provisional_category",
        "decision_reason",
        "source",
        "category",
        "confidence",
        "ml_fraud_score",
        "rule_score",
        "risk_score",
        "risk_level",
        "decision",
        "uncertain",
        "uncertainty_reasons",
        "indicator_count",
        "explanations",
        "recommendations",
        "probabilities",
        "disclaimer"
    ]

    clean = {
        field: deepcopy(result.get(field))
        for field in fields
    }

    clean["indicators"] = []

    for item in result.get("indicators", []):

        clean["indicators"].append({
            "indicator": item["indicator"],
            "weight": item["weight"],
            "matched_text": "[Not retained in saved result]"
        })

    return clean


def save_assessment(result, source):

    if not result.get("success"):

        return

    clear_assessment()

    assessment_id = uuid4().hex

    if source == "UPI":

        fields = [
            "success",
            "engine_version",
            "score_basis",
            "amount",
            "payment_type",
            "risk_score",
            "risk_level",
            "decision",
            "findings",
            "recommendations",
            "disclaimer"
        ]

        clean = {
            field: deepcopy(result.get(field))
            for field in fields
        }

        nested = result.get("message_analysis")

        clean["message_analysis"] = (
            sanitize_message_result(nested)
            if nested else None
        )

        category = (
            nested.get("category", "Unknown")
            if nested else "Unknown"
        )

    else:

        clean = sanitize_message_result(result)

        category = result.get(
            "category", "Unknown"
        )

    st.session_state["assessment_id"] = assessment_id

    st.session_state["last_analysis"] = {
        "success": True,
        "source": source,
        "category": category,
        "risk_score": result.get("risk_score"),
        "risk_level": result.get("risk_level"),
        "decision": result.get("decision"),
        "engine_version": result.get("engine_version", "Legacy"),
        "score_basis": result.get("score_basis", "Legacy heuristic score")
    }

    st.session_state["latest_display"] = {
        "source": source,
        "result": clean
    }


def get_saved_assessment(source):

    saved = st.session_state.get(
        "latest_display"
    )

    if saved and saved["source"] == source:

        return saved["result"]

    return None
