
from frontend.about_page import show_about_page
import streamlit as st
from pathlib import Path

from frontend.message_analysis import show_message_analysis
from frontend.upi_analysis import show_upi_analysis
from frontend.reporting_page import show_reporting_page
from frontend.performance_page import show_performance_page


BASE_DIR = Path(__file__).resolve().parent.parent


def load_styles():

    css_file = BASE_DIR / "frontend" / "styles.css"

    if css_file.exists():

        css = css_file.read_text(encoding="utf-8")

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


def show_home():

    st.markdown(
        '<div class="main-title">FraudGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Digital Arrest and UPI Fraud '
        'Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Analyse suspicious messages, call transcripts "
        "and payment requests using machine learning "
        "and contextual security rules."
    )

    st.divider()

    st.subheader("Detection Services")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <h3>WhatsApp Analysis</h3>
                <p>Identify suspicious WhatsApp messages
                and potential scam indicators.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>Call Analysis</h3>
                <p>Analyse call transcripts for digital
                arrest threats and impersonation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <h3>SMS Analysis</h3>
                <p>Detect suspicious banking messages,
                KYC requests and courier scams.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>UPI Security</h3>
                <p>Assess suspicious payment requests
                and payment authorisation risks.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Binary Accuracy", "96%")
    m2.metric("Fraud Recall", "92%")
    m3.metric("Triage Recall", "100%")
    m4.metric("Integration Tests", "142/142")

    st.caption(
        "Unified V4: recorded 100-message synthetic diagnostic benchmark. "
        "This dataset influenced development; real-world accuracy is unverified. "
        "Test count is the recorded post-integration result (24 Nov 2026), not a live test run."
    )

    st.divider()

    st.subheader("Cybercrime Reporting Assistance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Financial Fraud Helpline",
            "1930"
        )

    with col2:

        st.link_button(
            "Official Cybercrime Reporting Portal",
            "https://cybercrime.gov.in/"
        )

    st.warning(
        "If you have lost money through cyber fraud, "
        "contact your bank and call 1930 immediately."
    )

    st.info(
        "FraudGuard AI is an experimental risk "
        "assessment assistant. It does not verify "
        "sender identity or guarantee fraud detection."
    )


def show_placeholder(page):

    st.title(page)

    st.info(
        "This interface will be connected "
        "to the existing backend in the next step."
    )







def run_dashboard():

    load_styles()

    with st.sidebar:

        st.title("FraudGuard AI")
        st.caption("AI Fraud Risk Assessment")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Home",
                "WhatsApp Analysis",
                "SMS Analysis",
                "Call Analysis",
                "UPI Security",
                "Cybercrime Reporting",
                "Model Performance",
                "About Project"
            ]
        )

        st.divider()

        st.caption("Model: Unified V4")
        st.caption("Version: 1.0 Prototype")

        st.write("Emergency helpline: 1930")

    if page == "Home":

        show_home()

    elif page == "WhatsApp Analysis":

        show_message_analysis("WhatsApp")

    elif page == "SMS Analysis":

        show_message_analysis("SMS")

    elif page == "Call Analysis":

        show_message_analysis("Call")

    elif page == "UPI Security":

        show_upi_analysis()

    elif page == "Cybercrime Reporting":

        show_reporting_page()

    elif page == "Model Performance":

        show_performance_page()

    elif page == "About Project":

        show_about_page()

    st.markdown(
        '<div class="footer">'
        'FraudGuard AI | Hackathon Prototype'
        '</div>',
        unsafe_allow_html=True
    )
