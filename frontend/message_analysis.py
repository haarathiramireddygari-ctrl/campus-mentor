
from frontend.session_manager import (
    clear_assessment,
    save_assessment,
    get_saved_assessment
)
import streamlit as st
from frontend.display_utils import unique_recommendations
import pandas as pd

from core.application_detector import analyze_application


# Example messages for demonstration
EXAMPLES = {
    "WhatsApp": [
        "Transfer money immediately to avoid arrest.",
        "Pay a penalty to release your parcel.",
        "Your class starts at 9 AM tomorrow."
    ],

    "SMS": [
        "Your bank account will be blocked. Update KYC now.",
        "Share your OTP immediately.",
        "Never share your UPI PIN with anyone."
    ],

    "Call": [
        "Transfer money immediately to avoid arrest.",
        "You must pay a penalty to avoid arrest.",
        "The bank warns customers about fake verification links."
    ]
}


def display_analysis(result):

    if not result["success"]:
        st.error(result["error"])
        return

    st.divider()
    st.subheader("Fraud Analysis Result")

    score = result["risk_score"]
    level = result["risk_level"]
    decision = result["decision"]

    # Display risk decision
    if decision == "Potential Fraud":

        st.error(f"{level} Risk - Potential Fraud")

    elif decision == "Manual Verification Required":

        st.warning("Manual Verification Required")

    elif decision == "Insufficient Information":

        st.info("Insufficient Information")

    else:

        st.success("Low Detected Risk")

    # Main result metrics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Binary Fraud Score",
        f"{score:.2f}/100"
        if score is not None else "N/A"
    )

    c2.metric(
        "Decision Severity",
        level
    )

    c3.metric(
        "Category Confidence",
        f"{result['confidence']:.2f}%"
    )

    # Risk progress bar
    if score is not None:

        st.progress(
            float(score) / 100
        )

    st.caption(
        "Binary score is uncalibrated. Severity follows the final decision, "
        "which also considers legacy and security-rule evidence."
    )

    st.subheader("Provisional Category Assessment")

    st.caption("Engine: " + result["engine_version"])
    st.write("Decision reason:", result["decision_reason"])
    st.write("Original category estimate:", result["provisional_category"])

    st.write(result["category"])

    # ML and rule score comparison
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Legacy Hybrid Score",
        f"{result['legacy_risk_score']:.2f}"
        if result["legacy_risk_score"] is not None else "N/A"
    )

    col2.metric(
        "Security Rule Score",
        result["rule_score"]
    )

    col3.metric(
        "Detected Indicators",
        result["indicator_count"]
    )

    # Uncertainty information
    if result["uncertain"]:

        st.warning(
            "The system is uncertain about this assessment."
        )

        for reason in result["uncertainty_reasons"]:

            st.write("- " + reason)

    # Explainable fraud detection
    st.divider()
    st.subheader("Why Was This Result Generated?")

    for explanation in result["explanations"]:

        st.write("- " + explanation)

    # Detected evidence
    if result["indicators"]:

        with st.expander("View Detected Security Indicators"):

            for item in result["indicators"]:

                st.markdown(
                    "**" + item["indicator"]
                    .replace("_", " ").title() + "**"
                )

                st.text(
                    "Matched text: " + item["matched_text"]
                )

                st.write(
                    "Rule weight:", item["weight"]
                )

                st.divider()

    # Category probability chart
    with st.expander("View ML Category Probabilities"):

        st.bar_chart(
            pd.Series(result["probabilities"], name="Category estimate"), stack=False
        )

        st.caption(
            "These are uncalibrated model estimates."
        )

    # Safety recommendations
    st.divider()
    st.subheader("Recommended Safety Actions")

    for recommendation in unique_recommendations(result["recommendations"]):

        st.write("- " + recommendation)

    # Reporting assistance
    st.divider()
    st.subheader("Need Help?")

    col1, col2 = st.columns(2)

    col1.metric(
        "Financial Fraud Helpline",
        "1930"
    )

    with col2:

        st.link_button(
            "Report Cybercrime",
            "https://cybercrime.gov.in/"
        )

    st.caption(result["disclaimer"])


def show_message_analysis(source):

    st.title(f"{source} Analysis")

    if source == "Call":

        st.write(
            "Paste a written transcript or description "
            "of a suspicious phone call."
        )

        st.info(
            "This feature analyses text transcripts only. "
            "It does not listen to or record live calls."
        )

    else:

        st.write(
            f"Analyse suspicious {source} messages "
            "using the Unified V4 fraud detection engine."
        )

    st.caption(
        "Do not enter actual OTPs, UPI PINs, "
        "passwords or confidential banking details."
    )

    # Example message selector
    options = ["Enter your own message"] + EXAMPLES[source]

    selected = st.selectbox(
        "Select an example or enter your own message",
        options,
        key=f"{source}_example"
    )

    initial_message = (
        "" if selected == options[0] else selected
    )

    # Input form
    with st.form(key=f"{source}_analysis_form"):

        message = st.text_area(
            "Enter suspicious content",
            value=initial_message,
            height=160,
            max_chars=10000,
            key=f"{source}_message_{options.index(selected)}",
            placeholder="Paste the suspicious content here..."
        )

        submitted = st.form_submit_button(
            "Analyse for Fraud",
            type="primary",
            use_container_width=True
        )

    if submitted:
        clear_assessment()
        if not message.strip():
            st.error("Please enter a message.")
            return
        try:
            with st.spinner("Analysing fraud indicators..."):
                result = analyze_application(message, source=source)
            if not result["success"]:
                st.error(result["error"])
                return
            save_assessment(result, source)
        except FileNotFoundError:
            st.error("A required trained model is missing. Check the models folder.")
            return
        except Exception:
            st.error("Analysis failed. Check the application and model configuration.")
            return

    saved = get_saved_assessment(source)
    if saved is not None:
        display_analysis(saved)
