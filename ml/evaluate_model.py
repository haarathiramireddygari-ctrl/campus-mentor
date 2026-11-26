
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# File paths
MODEL_FILE = MODEL_DIR / "fraud_pipeline.joblib"
TEST_FILE = DATA_DIR / "independent_test.csv"
TRAIN_FILE = DATA_DIR / "cleaned_dataset.csv"

print("=" * 55)
print("FraudGuard AI - Independent Model Evaluation")
print("=" * 55)

# Load trained model and datasets
model = joblib.load(MODEL_FILE)

df = pd.read_csv(TEST_FILE)
train_df = pd.read_csv(TRAIN_FILE)

print("\nModel loaded successfully!")
print("Total test records:", len(df))

# Check exact-message overlap
def normalize(text):
    return " ".join(str(text).lower().split())

train_messages = set(
    train_df["message"].map(normalize)
)

test_messages = set(
    df["message"].map(normalize)
)

overlap = train_messages.intersection(test_messages)

print("Training and testing message overlap:", len(overlap))

if overlap:
    raise ValueError("Data leakage detected!")

# Actual and predicted categories
X_test = df["message"]
y_true = df["category"]

y_pred = model.predict(X_test)

# Multiclass evaluation
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("MULTICLASS EVALUATION RESULTS")
print("=" * 55)

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    zero_division=0
)

print(report)

# Multiclass confusion matrix
categories = [
    "Legitimate",
    "Digital Arrest",
    "KYC Fraud",
    "Courier Fraud",
    "UPI Fraud"
]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=categories
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories
)

plt.title("FraudGuard AI - Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# Binary fraud detection evaluation
actual_binary = df["label"].astype(int)

predicted_binary = [
    0 if category == "Legitimate" else 1
    for category in y_pred
]

binary_cm = confusion_matrix(
    actual_binary,
    predicted_binary,
    labels=[0, 1]
)

tn, fp, fn, tp = binary_cm.ravel()

print("\n" + "=" * 55)
print("BINARY FRAUD DETECTION RESULTS")
print("=" * 55)

print(
    f"Binary Accuracy: "
    f"{accuracy_score(actual_binary, predicted_binary) * 100:.2f}%"
)

print(
    f"Fraud Precision: "
    f"{precision_score(actual_binary, predicted_binary, zero_division=0) * 100:.2f}%"
)

print(
    f"Fraud Recall: "
    f"{recall_score(actual_binary, predicted_binary, zero_division=0) * 100:.2f}%"
)

print(
    f"Fraud F1-Score: "
    f"{f1_score(actual_binary, predicted_binary, zero_division=0) * 100:.2f}%"
)

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Binary confusion matrix graph
plt.figure(figsize=(6, 5))

sns.heatmap(
    binary_cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Binary Fraud Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "binary_confusion_matrix.png",
    dpi=300
)

plt.close()

# Save individual predictions
df["predicted_category"] = y_pred

df["correct_prediction"] = (
    df["category"] == df["predicted_category"]
)

probabilities = model.predict_proba(X_test)

legitimate_index = list(
    model.classes_
).index("Legitimate")

df["fraud_score"] = (
    1 - probabilities[:, legitimate_index]
) * 100

df["fraud_score"] = df["fraud_score"].round(2)

df.to_csv(
    REPORT_DIR / "evaluation_results.csv",
    index=False
)

# Save incorrect predictions separately
incorrect = df[
    df["correct_prediction"] == False
]

incorrect.to_csv(
    REPORT_DIR / "incorrect_predictions.csv",
    index=False
)

print("\nCorrect category predictions:",
      df["correct_prediction"].sum())

print("Incorrect category predictions:",
      len(incorrect))

# Save classification report
with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("FraudGuard AI - Model Evaluation\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Macro Precision: {precision * 100:.2f}%\n")
    file.write(f"Macro Recall: {recall * 100:.2f}%\n")
    file.write(f"Macro F1-Score: {f1 * 100:.2f}%\n\n")
    file.write(report)

# Display incorrect predictions
print("\nIncorrect predictions:")

print(
    incorrect[
        ["message", "category", "predicted_category"]
    ].to_string(index=False)
)

print("\nAll evaluation reports saved to:")
print(REPORT_DIR)

print("\nModel evaluation completed successfully!")
