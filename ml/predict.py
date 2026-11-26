
import joblib
from pathlib import Path
from functools import lru_cache

from core.preprocessing import (
    preprocess_text,
    validate_input
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR / "models" / "fraud_pipeline_v2.joblib"
)


# Load model only once per Python process
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Model V2 not found. "
            "Run ml/train_model_v2.py first."
        )

    model = joblib.load(MODEL_FILE)

    return model


# Main prediction function
def predict_fraud(message):

    # Validate user input
    valid, status = validate_input(message)

    if not valid:

        return {
            "success": False,
            "error": status
        }

    # Preprocess input
    cleaned_message = preprocess_text(message)

    # Load trained model
    model = load_model()

    # Predict category
    category = model.predict(
        [cleaned_message]
    )[0]

    # Generate class probabilities
    probabilities = model.predict_proba(
        [cleaned_message]
    )[0]

    # Map categories to probabilities
    class_probabilities = {

        str(name): float(probability)

        for name, probability in zip(
            model.classes_,
            probabilities
        )
    }

    # Calculate category confidence
    confidence = (
        class_probabilities[category] * 100
    )

    # Calculate estimated ML fraud score
    fraud_score = (
        1 - class_probabilities["Legitimate"]
    ) * 100

    # Check whether the model recognizes text features
    features = model.named_steps[
        "features"
    ].transform([cleaned_message])

    recognized_features = int(features.nnz)

    # Prepare structured prediction result
    result = {

        "success": True,

        "processed_message": cleaned_message,

        "category": str(category),

        "confidence": round(confidence, 2),

        "fraud_score": round(fraud_score, 2),

        "recognized_features": recognized_features,

        "probabilities": {
            name: round(probability * 100, 2)

            for name, probability
            in class_probabilities.items()
        },

        "model_version": "V2"
    }

    return result


# Test prediction module independently
if __name__ == "__main__":

    sample_messages = [

        "Transfer money immediately to avoid arrest.",

        "Your bank account will be blocked. Update KYC now.",

        "Pay a penalty to release your parcel.",

        "Enter your UPI PIN to receive cashback.",

        "Your class starts at 9 AM tomorrow.",

        "Never disclose your UPI PIN to anyone.",

        "The bank warns customers about fake verification links.",

        "xyz qqq zzzz",

        ""
    ]

    print("=" * 55)
    print("FraudGuard AI - Prediction Module Test")
    print("=" * 55)

    for message in sample_messages:

        result = predict_fraud(message)

        print("\n" + "-" * 55)
        print("Input:", repr(message))

        if not result["success"]:

            print("Error:", result["error"])
            continue

        print("Category:", result["category"])

        print(
            "Confidence:",
            str(result["confidence"]) + "%"
        )

        print(
            "ML Fraud Score:",
            str(result["fraud_score"]) + "/100"
        )

        print(
            "Recognized Features:",
            result["recognized_features"]
        )

        print("Model:", result["model_version"])

        print("\nCategory Probabilities:")

        for category, probability in (
            result["probabilities"].items()
        ):

            print(
                f"  {category}: {probability}%"
            )

    print("\n" + "=" * 55)
    print("Prediction module tests completed!")
    print("=" * 55)
