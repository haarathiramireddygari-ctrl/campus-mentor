
from frontend.session_manager import (
    clear_assessment,
    save_assessment,
    get_saved_assessment
)
import streamlit as st
from frontend.display_utils import unique_recommendations

from security.upi_checker import analyze_upi_payment


def display_upi_result(result):

    if not result["success"]:
        st.error(result["error"])
        return

    st.divider()
    st.subheader("UPI Security Assessment")

    score = result["risk_score"]
    level = result["risk_level"]
    decision = result["decision"]

    # Display overall decision
    if decision == "Potential Fraud":

        st.error(f"{level} Risk - Potential Fraud")

    elif decision == "Insufficient Information":

        st.info("Insufficient Information")

    else:

        st.warning("Manual Verification Required")

    # Main assessment metrics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Payment Assessment Score",
        f"{score:.2f}/100"
        if score is not None else "N/A"
    )

    c2.metric(
        "Decision Severity",
        level
    )

    c3.metric(
        "Transaction Amount",
       f"₹{float(result['amount']):,.2f}"
    )

    if score is not None:

        st.progress(float(score) / 100)

    st.caption(
        "Payment score is the maximum of available binary and transaction-rule scores. "
        "Severity follows the decision; account ownership is never verified."
    )

    # UPI validation results
    st.divider()
    st.subheader("UPI Address Validation")

    st.success("Basic UPI address format is valid.")

    st.warning(
        "Account ownership and recipient identity "
        "have not been verified."
    )

    st.write(
        "Payment Type:",
        result["payment_type"]
    )

    # Message analysis
    analysis = result.get("message_analysis")

    if analysis:

        st.divider()
        st.subheader("AI Message Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Binary Fraud Score",
            f"{analysis['binary_fraud_score']:.2f}"
        )

        col2.metric(
            "Security Rule Score",
            analysis["rule_score"]
        )

        col3.metric(
            "Category Confidence",
            f"{analysis['confidence']:.2f}%"
        )

        st.write(
            "Provisional Category Assessment:",
            analysis["category"]
        )

        st.write("Message decision:", analysis["decision"])
        st.write("Decision reason:", analysis["decision_reason"])

        if analysis["uncertain"]:

            st.warning(
                "The model is uncertain about "
                "the message classification."
            )

            for reason in analysis["uncertainty_reasons"]:
                st.write("- " + reason)

        with st.expander("View Detected Security Indicators"):

            if analysis["indicators"]:

                for item in analysis["indicators"]:

                    st.write(
                        item["indicator"]
                        .replace("_", " ").title()
                    )

                    st.text(
                        "Matched text: " + item["matched_text"]
                    )

                    st.write(
                        "Rule weight:",
                        item["weight"]
                    )

                    st.divider()

            else:

                st.info(
                    "No predefined scam indicators detected."
                )

    # Security findings
    st.divider()
    st.subheader("Security Findings")

    for finding in result["findings"]:

        st.write("- " + finding)

    # Safety recommendations
    st.divider()
    st.subheader("Recommended Safety Actions")

    for recommendation in unique_recommendations(result["recommendations"]):

        st.write("- " + recommendation)

    # Emergency reporting
    st.divider()
    st.subheader("Cybercrime Reporting Assistance")

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


def show_upi_analysis():

    st.title("UPI Payment Security")

    st.write(
        "Analyse suspicious UPI payment requests "
        "using machine learning and contextual "
        "security indicators."
    )

    st.warning(
        "Never enter your actual UPI PIN, OTP, "
        "password or confidential banking details."
    )

    st.info(
        "This application does not connect to banks, "
        "verify account ownership or execute payments."
    )

    # User input form
    with st.form("upi_security_form"):

        upi_id = st.text_input(
            "UPI ID",
            value="demo@sample",
            max_chars=255,
            placeholder="example@provider"
        )

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.2f"
        )

        payment_type = st.selectbox(
            "Payment Type",
            [
                "Send Money",
                "Receive Money",
                "Collect Request"
            ],
            index=1
        )

        message = st.text_area(
            "Accompanying Payment Message (Optional)",
            value="Enter your UPI PIN to receive cashback.",
            height=150,
            max_chars=10000
        )

        submitted = st.form_submit_button(
            "Analyse UPI Security",
            type="primary",
            use_container_width=True
        )

    if submitted:
        clear_assessment()
        try:
            with st.spinner("Analysing payment security..."):
                result = analyze_upi_payment(upi_id, amount, payment_type, message)
            if not result["success"]:
                st.error(result["error"])
                return
            save_assessment(result, "UPI")
        except FileNotFoundError:
            st.error("A required trained model is missing. Check the models folder.")
            return
        except Exception:
            st.error("UPI analysis failed. Check the application configuration.")
            return

    saved = get_saved_assessment("UPI")
    if saved is not None:
        display_upi_result(saved)
