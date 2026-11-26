
import streamlit as st
import pandas as pd
from pathlib import Path


def show_about_page():

    st.title("About FraudGuard AI")

    st.subheader(
        "AI-Powered Digital Arrest and "
        "UPI Fraud Detection System"
    )

    st.caption(
        "Version 1.0 | Hackathon Prototype"
    )

    st.divider()

    # Problem statement
    st.subheader("Problem Statement")

    st.write(
        "Cybercriminals use impersonation, threatening "
        "messages, fake payment requests and social "
        "engineering to manipulate users into transferring "
        "money or disclosing confidential information."
    )

    st.write(
        "Users may struggle to identify suspicious "
        "communication before taking financial action."
    )

    # Proposed solution
    st.subheader("Proposed Solution")

    st.write(
        "FraudGuard AI combines machine learning "
        "with contextual security rules to identify "
        "potential fraud indicators, estimate risk "
        "and provide actionable safety recommendations."
    )

    st.divider()

    # Project objectives
    st.subheader("Project Objectives")

    objectives = [
        "Identify suspicious digital communications.",
        "Detect potential digital arrest scams.",
        "Analyse suspicious UPI payment requests.",
        "Explain detected fraud indicators.",
        "Provide category-specific safety guidance.",
        "Assist users with cybercrime reporting.",
        "Evaluate model performance transparently."
    ]

    for objective in objectives:

        st.write("- " + objective)

    st.divider()

    # System architecture
    st.subheader("System Architecture")

    st.image(str(Path(__file__).with_name("architecture.svg")), use_container_width=True)

    st.caption(
        "Simplified logical architecture. "
        "UPI input validation occurs before message "
        "analysis, and UPI requests without text "
        "can be assessed using limited transaction rules."
    )

    st.divider()

    # Supported categories
    st.subheader("Supported Fraud Categories")

    categories = {

        "Digital Arrest":
            "Threats involving fake police or "
            "law enforcement claims.",

        "KYC Fraud":
            "Fake account verification requests "
            "and credential theft attempts.",

        "Courier Fraud":
            "Suspicious parcel claims, penalties "
            "and delivery-related demands.",

        "UPI Fraud":
            "Misleading payment requests, "
            "refund scams and credential requests.",

        "Legitimate":
            "Messages predicted as non-fraudulent, "
            "without guaranteeing sender authenticity.",

        "Unknown":
            "Messages with insufficient evidence "
            "or uncertain model predictions."
    }

    for category, description in categories.items():

        with st.expander(category):

            st.write(description)

    st.divider()

    # Technology stack
    st.subheader("Technology Stack")

    technologies = pd.DataFrame({

        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "TF-IDF",
            "Logistic Regression",
            "Regular Expressions",
            "Joblib",
            "Streamlit",
            "Matplotlib",
            "Seaborn",
            "Pytest",
            "Git"
        ],

        "Purpose": [
            "Core application development",
            "Dataset processing",
            "Numerical computation",
            "Machine learning pipeline",
            "Text feature extraction",
            "Message classification",
            "Security pattern detection",
            "Model serialization",
            "Interactive web application",
            "Performance visualization",
            "Confusion matrix visualization",
            "Automated software testing",
            "Version control"
        ]
    })

    st.dataframe(
        technologies,
        hide_index=True
    )

    st.divider()

    # Machine learning explanation
    st.subheader("Machine Learning Methodology")

    st.markdown("**1. Text Preprocessing**")

    st.write(
        "Input text is normalized and cleaned "
        "before feature extraction."
    )

    st.markdown("**2. TF-IDF Feature Extraction**")

    st.write(
        "Word-level and character-level TF-IDF "
        "convert text into numerical features."
    )

    st.markdown("**3. Logistic Regression**")

    st.write(
        "Two trained classifiers estimate overall fraud risk "
        "and the provisional message category."
    )

    st.markdown("**4. Security Rule Engine**")

    st.write(
        "Contextual rules identify threats, "
        "credential requests, urgency and "
        "suspicious payment instructions."
    )

    st.markdown("**5. Unified V4 Decision Policy**")
    st.write("Binary V2 (1,665 training records) provides the primary score. "
             "The retained category model (1,525 records) and security rules provide context. "
             "Legacy hybrid risk above 60 can refer a low binary score for verification.")
    st.dataframe(pd.DataFrame({
        "Binary score (before safeguards)": ["0–36", "Above 36, below 50", "50–100"],
        "Default decision": ["Low Detected Risk", "Manual Verification Required", "Potential Fraud"]
    }), hide_index=True)
    st.caption("Strong indicators, insufficient evidence, mixed context and disagreement can override these defaults. "
               "Severity follows the decision; the displayed score is not a calibrated probability.")
    st.write("UPI checks retain transaction-specific rules and never verify account ownership. "
             "The payment assessment score is the maximum of the binary and transaction-rule scores when available.")
    st.divider()

    # Privacy and security
    st.subheader("Privacy and Security")

    privacy_features = [
        "No banking API integration.",
        "No automatic financial transactions.",
        "No live phone call recording.",
        "No automatic police complaint submission.",
        "Raw messages and UPI IDs are excluded "
        "from generated incident summaries.",
        "Report-related metadata is held "
        "temporarily in the application session."
    ]

    for feature in privacy_features:

        st.write("- " + feature)

    st.divider()

    # Limitations
    st.subheader("Current Limitations")

    limitations = [
        "Training data is synthetic.",
        "Real-world fraud detection accuracy "
        "has not been established.",
        "Legitimate messages may be flagged incorrectly.",
        "Some fraudulent messages may be missed.",
        "UPI account ownership is not verified.",
        "Call analysis supports written transcripts only.",
        "Multilingual detection remains unverified.",
        "Risk scores are not calibrated probabilities."
    ]

    for limitation in limitations:

        st.write("- " + limitation)

    st.divider()

    # Future enhancements
    st.subheader("Future Enhancements")

    enhancements = [
        "Collect larger, responsibly sourced datasets.",
        "Improve detection of legitimate messages.",
        "Support multilingual fraud detection.",
        "Integrate speech-to-text for call recordings "
        "with appropriate user consent.",
        "Add suspicious URL reputation checks.",
        "Explore authorized payment verification APIs.",
        "Develop mobile application support.",
        "Implement model monitoring and MLOps pipelines."
    ]

    for enhancement in enhancements:

        st.write("- " + enhancement)

    st.divider()

    # Team responsibilities
    st.subheader("Project Module Responsibilities")

    team = pd.DataFrame({

        "Assigned Role": [
            "Member 1",
            "Member 2",
            "Member 3",
            "Member 4"
        ],

        "Responsibility": [
            "Core detection and system integration",
            "Dataset and machine learning",
            "Frontend and dashboard",
            "Security and reporting"
        ]
    })

    st.dataframe(
        team,
        hide_index=True
    )

    st.caption(
        "This table describes planned module "
        "responsibilities, not verified individual "
        "contributions."
    )

    st.divider()

    # Disclaimer
    st.warning(
        "FraudGuard AI is an experimental hackathon "
        "prototype. It provides risk assessment "
        "assistance and does not replace banks, "
        "law enforcement or cybersecurity professionals."
    )
