
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "cleaned_dataset.csv"
MODEL_FILE = MODEL_DIR / "fraud_pipeline.joblib"

print("=" * 50)
print("FraudGuard AI - Model Training")
print("=" * 50)

# Load training dataset
df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Total training records:", len(df))

# Input and target variables
X = df["message"]
y = df["category"]

print("\nTraining categories:")
print(y.value_counts())

# Create machine learning pipeline
pipeline = Pipeline([

    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))

])

# Train the model
print("\nTraining the machine learning model...")

pipeline.fit(X, y)

print("Model training completed successfully!")

# Display learned categories
print("\nModel categories:")

for category in pipeline.classes_:
    print("-", category)

# Save trained pipeline
joblib.dump(pipeline, MODEL_FILE)

print("\nModel saved successfully:")
print(MODEL_FILE)

# Basic prediction smoke test
sample_messages = [

    "Your bank account will be blocked. Update KYC now.",

    "Transfer money immediately to avoid arrest.",

    "Pay a penalty to release your parcel.",

    "Enter your UPI PIN to receive cashback.",

    "Your class starts at 9 AM tomorrow."

]

print("\n" + "=" * 50)
print("Basic Prediction Test")
print("=" * 50)

for message in sample_messages:

    prediction = pipeline.predict([message])[0]

    probabilities = pipeline.predict_proba([message])[0]

    class_probabilities = dict(
        zip(pipeline.classes_, probabilities)
    )

    confidence = class_probabilities[prediction] * 100

    fraud_probability = (
        1 - class_probabilities["Legitimate"]
    ) * 100

    print("\nMessage:", message)
    print("Predicted category:", prediction)
    print(f"Category confidence: {confidence:.2f}%")
    print(f"Estimated fraud score: {fraud_probability:.2f}/100")

print("\nModel training process completed!")
