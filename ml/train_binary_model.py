
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from core.preprocessing import preprocess_text


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR / "data" / "improved_dataset.csv"
)

DEV_FILE = (
    BASE_DIR / "data" / "hybrid_development_100.csv"
)

MODEL_FILE = (
    BASE_DIR / "models" / "binary_candidate.joblib"
)

REPORT_DIR = (
    BASE_DIR / "reports" / "binary_candidate"
)

VALID_CATEGORIES = {
    "Legitimate",
    "Digital Arrest",
    "KYC Fraud",
    "Courier Fraud",
    "UPI Fraud"
}


def load_dataset(path):

    df = pd.read_csv(path)

    if not {"message", "category"}.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[["message", "category"]].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    df["category"] = df["category"].str.strip()

    if not set(df["category"]).issubset(
        VALID_CATEGORIES
    ):
        raise ValueError("Invalid fraud category.")

    df["processed_message"] = df["message"].map(
        preprocess_text
    )

    df["binary_label"] = (
        df["category"] != "Legitimate"
    ).astype(int)

    if df["binary_label"].nunique() != 2:
        raise ValueError("Both binary classes are required.")

    return df


def evaluate_policy(
    y_true,
    scores,
    alert_threshold,
    low_threshold,
    policy_name
):

    alerts = (
        scores >= alert_threshold
    ).astype(int)

    triage = (
        (scores > low_threshold)
        | (alerts == 1)
    ).astype(int)

    manual_review = (
        (triage == 1) & (alerts == 0)
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        alerts,
        labels=[0, 1]
    ).ravel()

    metrics = {

        "Policy": policy_name,

        "Binary Accuracy": round(
            accuracy_score(y_true, alerts) * 100,
            2
        ),

        "Fraud Precision": round(
            precision_score(
                y_true, alerts, zero_division=0
            ) * 100,
            2
        ),

        "Fraud Recall": round(
            recall_score(
                y_true, alerts, zero_division=0
            ) * 100,
            2
        ),

        "Fraud F1": round(
            f1_score(
                y_true, alerts, zero_division=0
            ) * 100,
            2
        ),

        "False Positive Rate": round(
            fp / (fp + tn) * 100,
            2
        ),

        "Triage Fraud Recall": round(
            recall_score(
                y_true, triage, zero_division=0
            ) * 100,
            2
        ),

        "Legitimate Triage Rate": round(
            triage[y_true == 0].mean() * 100,
            2
        ),

        "Manual Review Rate": round(
            manual_review.mean() * 100,
            2
        ),

        "Fraud Without Alert or Review": int(
            ((y_true == 1) & (triage == 0)).sum()
        ),

        "True Positives": int(tp),

        "True Negatives": int(tn),

        "False Positives": int(fp),

        "False Negatives": int(fn)

    }

    print("\n" + "=" * 65)
    print(policy_name)
    print("=" * 65)

    for name, value in metrics.items():
        print(f"{name:<38}: {value}")

    return metrics


def train_binary_model():

    train = load_dataset(TRAIN_FILE)
    dev = load_dataset(DEV_FILE)

    overlap = set(
        train["processed_message"]
    ) & set(
        dev["processed_message"]
    )

    if overlap:
        raise ValueError(
            f"Training overlap detected: {len(overlap)}"
        )

    print("=" * 65)
    print("FraudGuard AI - Binary Classifier")
    print("=" * 65)

    print("Training records:", len(train))
    print("Development records:", len(dev))
    print("Exact normalized overlap:", len(overlap))

    print("\nTraining class distribution:")

    print(
        train["binary_label"]
        .map({0: "Legitimate", 1: "Fraud"})
        .value_counts()
    )

    features = FeatureUnion([

        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True
            )
        )

    ])

    pipeline = Pipeline([

        ("features", features),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )

    ])

    print("\nTraining binary classifier...")

    pipeline.fit(
        train["processed_message"],
        train["binary_label"]
    )

    classes = list(
        pipeline.named_steps["classifier"].classes_
    )

    fraud_index = classes.index(1)

    scores = pipeline.predict_proba(
        dev["processed_message"]
    )[:, fraud_index]

    y_true = dev["binary_label"].to_numpy()

    # Experiment A:
    # Standard binary classification
    metrics_a = evaluate_policy(
        y_true,
        scores,
        alert_threshold=0.50,
        low_threshold=0.50,
        policy_name="Binary Candidate - Standard"
    )

    # Experiment B:
    # Three-way decision policy
    metrics_b = evaluate_policy(
        y_true,
        scores,
        alert_threshold=0.70,
        low_threshold=0.30,
        policy_name="Binary Candidate - Three Way"
    )

    predictions = dev[
        ["message", "category", "binary_label"]
    ].copy()

    predictions["binary_fraud_score"] = (
        scores * 100
    ).round(2)

    predictions["decision"] = [

        "Potential Fraud"
        if score >= 0.70

        else "Low Detected Risk"
        if score <= 0.30

        else "Manual Verification Required"

        for score in scores
    ]

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    pd.DataFrame([
        metrics_a,
        metrics_b
    ]).to_csv(
        REPORT_DIR / "binary_metrics.csv",
        index=False
    )

    predictions.to_csv(
        REPORT_DIR / "binary_predictions.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("Training and evaluation completed!")
    print("Candidate model:", MODEL_FILE)
    print("Reports:", REPORT_DIR)
    print("=" * 65)


if __name__ == "__main__":

    train_binary_model()
