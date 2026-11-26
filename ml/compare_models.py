
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
    confusion_matrix,
    classification_report
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "comparison"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load fresh test dataset
df = pd.read_csv(
    DATA_DIR / "final_test_dataset.csv"
)

train_df = pd.read_csv(
    DATA_DIR / "improved_dataset.csv"
)

def normalize(text):
    return " ".join(str(text).lower().split())

# Check dataset integrity
assert not df.isnull().any().any(), "Missing values found"
assert not df["message"].duplicated().any(), "Duplicates found"

# Check exact training-test overlap
train_messages = set(
    train_df["message"].map(normalize)
)

test_messages = set(
    df["message"].map(normalize)
)

overlap = train_messages.intersection(test_messages)

if overlap:
    raise ValueError("Training-test overlap detected!")

print("=" * 60)
print("FraudGuard AI - Model Comparison")
print("=" * 60)

print("\nTotal test records:", len(df))
print("Training-test overlap:", len(overlap))

# Input and target variables
X_test = df["message"]
y_true = df["category"]
actual_binary = df["label"].astype(int)

categories = [
    "Legitimate",
    "Digital Arrest",
    "KYC Fraud",
    "Courier Fraud",
    "UPI Fraud"
]

# Model paths
models = {
    "Model V1": MODEL_DIR / "fraud_pipeline.joblib",
    "Model V2": MODEL_DIR / "fraud_pipeline_v2.joblib"
}

results = []

# Evaluate each model
for model_name, model_path in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    predicted_binary = [
        0 if category == "Legitimate" else 1
        for category in y_pred
    ]

    # Multiclass metrics
    multi_accuracy = accuracy_score(
        y_true, y_pred
    )

    multi_precision = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    multi_recall = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    multi_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    # Binary metrics
    binary_accuracy = accuracy_score(
        actual_binary, predicted_binary
    )

    fraud_precision = precision_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    fraud_recall = recall_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    fraud_f1 = f1_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        actual_binary,
        predicted_binary,
        labels=[0, 1]
    ).ravel()

    # Digital arrest recall
    arrest_recall = recall_score(
        y_true == "Digital Arrest",
        y_pred == "Digital Arrest",
        zero_division=0
    )

    false_positive_rate = fp / (fp + tn)

    # Store metrics
    results.append({
        "Model": model_name,
        "Multiclass Accuracy": multi_accuracy * 100,
        "Macro Precision": multi_precision * 100,
        "Macro Recall": multi_recall * 100,
        "Macro F1": multi_f1 * 100,
        "Binary Accuracy": binary_accuracy * 100,
        "Fraud Precision": fraud_precision * 100,
        "Fraud Recall": fraud_recall * 100,
        "Fraud F1": fraud_f1 * 100,
        "Digital Arrest Recall": arrest_recall * 100,
        "False Positive Rate": false_positive_rate * 100,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "True Negatives": tn
    })

    print(f"\nMulticlass Accuracy: {multi_accuracy * 100:.2f}%")
    print(f"Macro Precision: {multi_precision * 100:.2f}%")
    print(f"Macro Recall: {multi_recall * 100:.2f}%")
    print(f"Macro F1-Score: {multi_f1 * 100:.2f}%")

    print(f"\nBinary Accuracy: {binary_accuracy * 100:.2f}%")
    print(f"Fraud Precision: {fraud_precision * 100:.2f}%")
    print(f"Fraud Recall: {fraud_recall * 100:.2f}%")
    print(f"Fraud F1-Score: {fraud_f1 * 100:.2f}%")

    print(f"\nDigital Arrest Recall: {arrest_recall * 100:.2f}%")
    print(f"False Positive Rate: {false_positive_rate * 100:.2f}%")

    print("\nTrue Positives:", tp)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    print("\nClassification Report:")

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    print(report)

    # Save individual predictions
    df[model_name + " Prediction"] = y_pred

    # Generate confusion matrix
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

    plt.title(model_name + " - Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.xticks(rotation=30)
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        REPORT_DIR / f"{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

df.to_csv(
    REPORT_DIR / "all_predictions.csv",
    index=False
)

# Display final comparison
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(2)
    .set_index("Model")
    .transpose()
    .to_string()
)

# Create performance comparison graph
metrics = [
    "Multiclass Accuracy",
    "Binary Accuracy",
    "Fraud Precision",
    "Fraud Recall",
    "Fraud F1",
    "Digital Arrest Recall"
]

ax = results_df.set_index("Model")[metrics].T.plot(
    kind="bar",
    figsize=(12, 7),
    color=["#94a3b8", "#2563eb"]
)

plt.title("FraudGuard AI - Model V1 vs Model V2")
plt.ylabel("Percentage")
plt.xlabel("Evaluation Metrics")
plt.ylim(0, 110)
plt.xticks(rotation=25, ha="right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "performance_comparison.png",
    dpi=300
)

plt.close()

print("\nAll comparison reports saved to:")
print(REPORT_DIR)

print("\nModel comparison completed successfully!")
