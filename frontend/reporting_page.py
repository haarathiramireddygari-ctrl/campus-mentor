
import streamlit as st

from security.reporting import (
    get_reporting_guidance,
    generate_incident_report
)


def show_reporting_page():

    st.title("Cybercrime Reporting Assistance")

    st.write(
        "Get emergency guidance, preserve evidence "
        "and prepare an incident summary."
    )

    st.warning(
        "If you have lost money through cyber fraud, "
        "contact your bank and call 1930 immediately."
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Financial Fraud Helpline",
        "1930"
    )

    with c2:

        st.link_button(
            "Official Cybercrime Reporting Portal",
            "https://cybercrime.gov.in/"
        )

    st.divider()

    analysis = st.session_state.get(
        "last_analysis"
    )

    assessment_id = st.session_state.get(
        "assessment_id"
    )

    st.subheader("Latest Fraud Assessment")

    if analysis:

        category = (
            analysis.get("category") or "Unknown"
        )

        score = analysis.get("risk_score")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Source",
            analysis.get("source", "Unknown")
        )

        c2.metric(
            "Assessment Score",
            f"{score:.2f}/100"
            if score is not None else "N/A"
        )

        c3.metric(
            "Decision Severity",
            analysis.get("risk_level", "Unknown")
        )

        st.write(
            "Provisional Category Assessment:",
            category
        )

        st.write(
            "Decision:",
            analysis.get("decision", "Unknown")
        )

        st.caption("Engine: " + str(analysis.get("engine_version", "Legacy")))
        st.caption("Score basis: " + str(analysis.get("score_basis", "Legacy heuristic score")))

    else:

        category = "Unknown"

        st.info(
            "No previous assessment is available."
        )

    st.divider()

    st.subheader("Incident Information")

    # Widget keys may be removed when another page is rendered. Keep a
    # separate, non-widget snapshot for the current assessment.
    selection = st.session_state.get("report_selection", {})
    if "report_money_lost" not in st.session_state:
        st.session_state["report_money_lost"] = selection.get("money_lost", "No")

    money_lost = st.radio(
        "Have you already lost money?",
        ["No", "Yes"],
        horizontal=True,
        key="report_money_lost"
    ) == "Yes"

    guidance = get_reporting_guidance(
        category,
        money_lost
    )

    if money_lost:

        st.error(
            "Immediate action required. Contact "
            "your bank and report the incident."
        )

    st.subheader(
        "Recommended Immediate Actions"
    )

    for action in guidance["immediate_actions"]:

        st.write("- " + action)

    st.divider()

    st.subheader("Evidence Checklist")

    available = []

    for index, item in enumerate(
        guidance["evidence_checklist"]
    ):

        widget_key = f"evidence_{index}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = item in selection.get("evidence", ())
        checked = st.checkbox(item, key=widget_key)

        if checked:

            available.append(item)

    st.session_state["report_selection"] = {
        "money_lost": "Yes" if money_lost else "No",
        "evidence": tuple(available),
    }

    total = len(
        guidance["evidence_checklist"]
    )

    completed = len(available)

    st.progress(completed / total)

    st.caption(
        f"{completed} of {total} evidence "
        "items marked as available."
    )

    st.caption(
        "Do not delay urgent reporting "
        "while collecting evidence."
    )

    st.divider()

    st.subheader("Download Incident Summary")

    st.info(
        "Raw messages and UPI IDs are excluded "
        "from the generated summary. "
        "This is not an official police complaint."
    )

    if analysis:

        context = (
            assessment_id,
            money_lost,
            tuple(available)
        )

        if (
            st.session_state.get("report_context")
            != context
        ):

            st.session_state.pop(
                "generated_report", None
            )

            st.session_state[
                "report_context"
            ] = context

        if st.button(
            "Generate Incident Summary",
            type="primary",
            use_container_width=True
        ):

            report = generate_incident_report(
                analysis,
                money_lost=money_lost,
                evidence_available=available
            )

            st.session_state[
                "generated_report"
            ] = report

        report = st.session_state.get(
            "generated_report"
        )

        if report:

            st.success(
                "Incident summary generated successfully."
            )

            st.write(
                "Local Reference:",
                report["report_id"]
            )

            with st.expander(
                "Preview Incident Summary"
            ):

                st.text(
                    report["report_text"]
                )

            st.download_button(
                "Download Incident Summary (.txt)",
                data=report["report_text"],
                file_name=report["filename"],
                mime="text/plain",
                use_container_width=True
            )

    else:

        st.warning(
            "Complete a fraud analysis before "
            "generating an assessment summary."
        )

    st.divider()

    st.subheader("Privacy and Security")

    st.write(guidance["privacy_warning"])

    st.caption(guidance["disclaimer"])
