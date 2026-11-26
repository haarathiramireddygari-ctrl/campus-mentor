
import streamlit as st

from frontend.dashboard import run_dashboard


st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


if __name__ == "__main__":

    run_dashboard()
