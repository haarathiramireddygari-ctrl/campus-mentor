
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# File paths
DATA_FILE = DATA_DIR / "improved_dataset.csv"
MODEL_FILE = MODEL_DIR / "fraud_pipeline_v2.joblib"

print("=" * 55)
print("FraudGuard AI - Model V2 Training")
print("=" * 55)

# Load improved training dataset
df = pd.read_csv(DATA_FILE)

X = df["message"]
y = df["category"]

print("\nTraining dataset loaded successfully!")
print("Total training records:", len(df))

print("\nTraining categories:")
print(y.value_counts())

# Word-level TF-IDF
word_tfidf = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    max_features=15000,
    sublinear_tf=True
)

# Character-level TF-IDF
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    lowercase=True,
    ngram_range=(3, 5),
    max_features=20000,
    sublinear_tf=True
)

# Combine both feature types
combined_features = FeatureUnion([
    ("word_tfidf", word_tfidf),
    ("char_tfidf", char_tfidf)
])

# Create improved ML pipeline
pipeline = Pipeline([
    ("features", combined_features),

    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        C=1.0
    ))
])

# Train Model V2
print("\nTraining Model V2...")
print("Using word-level and character-level TF-IDF.")

pipeline.fit(X, y)

print("\nModel V2 trained successfully!")

# Display number of learned features
feature_count = len(
    pipeline.named_steps["features"]
    .get_feature_names_out()
)

print("Total learned features:", feature_count)

# Display model categories
print("\nModel categories:")

for category in pipeline.classes_:
    print("-", category)

# Save Model V2 separately
joblib.dump(pipeline, MODEL_FILE)

print("\nModel V2 saved successfully:")
print(MODEL_FILE)

# Functional prediction tests
sample_messages = [
    "Your bank account will be blocked. Update KYC now.",

    "Transfer money immediately to avoid arrest.",

    "Pay a penalty to release your parcel.",

    "Enter your UPI PIN to receive cashback.",

    "Your class starts at 9 AM tomorrow.",

    "Never disclose your UPI PIN to anyone.",

    "The bank warns customers about fake verification links.",

    "Urgnt notification. Verify your acc0unt now."
]

print("\n" + "=" * 55)
print("MODEL V2 - BASIC PREDICTION TEST")
print("=" * 55)

for message in sample_messages:

    prediction = pipeline.predict([message])[0]

    probabilities = pipeline.predict_proba([message])[0]

    class_probabilities = dict(
        zip(pipeline.classes_, probabilities)
    )

    confidence = class_probabilities[prediction] * 100

    fraud_score = (
        1 - class_probabilities["Legitimate"]
    ) * 100

    print("\nMessage:", message)
    print("Predicted category:", prediction)

    print(
        f"Category confidence: {confidence:.2f}%"
    )

    print(
        f"Estimated fraud score: {fraud_score:.2f}/100"
    )

print("\n" + "=" * 55)
print("Model V2 training completed successfully!")
print("=" * 55)
