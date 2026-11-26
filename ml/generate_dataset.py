
import pandas as pd
import random
from pathlib import Path

random.seed(42)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Input sources
sources = ["WhatsApp", "SMS", "Call", "UPI"]

# Digital arrest scam messages
digital_arrest = [
    "You are under digital arrest.",
    "A criminal case has been registered against you.",
    "Your identity is linked to illegal activities.",
    "You must pay a penalty to avoid arrest.",
    "A police officer is investigating your bank account.",
    "Your Aadhaar is linked to a criminal investigation.",
    "You must remain on this call until verification.",
    "Transfer money to resolve your legal case."
]

# KYC scam messages
kyc_scam = [
    "Your bank account will be blocked due to incomplete KYC.",
    "Update your KYC immediately using the provided link.",
    "Your debit card has expired. Verify your details.",
    "Submit your OTP to complete account verification.",
    "Your banking services will be suspended.",
    "Your account requires urgent identity verification.",
    "Provide your banking details to reactivate your account.",
    "Complete online verification to prevent account closure."
]

# Courier scam messages
courier_scam = [
    "Your parcel contains prohibited items.",
    "Your shipment has been detained by customs.",
    "Pay a penalty to release your parcel.",
    "Your courier is linked to an illegal shipment.",
    "A parcel registered in your name is under investigation.",
    "Your package has been seized by authorities.",
    "Pay additional charges to avoid legal action.",
    "Your shipment requires an urgent security clearance payment."
]

# UPI scam messages
upi_scam = [
    "Enter your UPI PIN to receive cashback.",
    "Approve this collect request to receive your refund.",
    "Scan the QR code to receive your prize money.",
    "Share your UPI PIN for payment verification.",
    "Pay a registration fee to claim your reward.",
    "Approve the payment request to unlock your account.",
    "Send money to verify your UPI account.",
    "Enter your payment credentials to receive money."
]

# Legitimate messages
legitimate = [
    "Your class starts at 9 AM tomorrow.",
    "Your order has been delivered successfully.",
    "Your monthly salary has been credited.",
    "Please attend the scheduled team meeting.",
    "Your electricity bill is available.",
    "Your appointment has been confirmed.",
    "Your payment was completed successfully.",
    "Contact your bank through its official application.",
    "Never share your UPI PIN with anyone.",
    "Your KYC status can be checked at your bank branch.",
    "Your parcel is out for delivery.",
    "Your examination results have been published.",
    "Report suspicious calls to the authorities.",
    "Your account statement is available.",
    "Your refund has been processed.",
    "Please submit your college assignment tomorrow."
]

# Neutral variations for synthetic examples
prefixes = [
    "",
    "Hello, ",
    "Dear customer, ",
    "Please note: ",
    "Notification: ",
    "Important message: ",
    "Attention: ",
    "Update: "
]

suffixes = [
    "",
    " Please check the details.",
    " This is a notification.",
    " Please review this message.",
    " For your information.",
    " Kindly take note.",
    " Please read carefully.",
    " This message concerns your request."
]

records = []

# Generate records for each category
def generate_records(messages, category, label, count):

    for _ in range(count):

        message = (
            random.choice(prefixes)
            + random.choice(messages)
            + random.choice(suffixes)
        )

        records.append({
            "message": message,
            "source": random.choice(sources),
            "category": category,
            "label": label,
            "origin": "synthetic"
        })


# Generate 2,000 records
generate_records(legitimate, "Legitimate", 0, 800)
generate_records(digital_arrest, "Digital Arrest", 1, 300)
generate_records(kyc_scam, "KYC Fraud", 1, 300)
generate_records(courier_scam, "Courier Fraud", 1, 300)
generate_records(upi_scam, "UPI Fraud", 1, 300)

# Shuffle records
random.shuffle(records)

# Create DataFrame
df = pd.DataFrame(records)

# Save CSV file
output_file = DATA_DIR / "fraud_dataset.csv"
df.to_csv(output_file, index=False)

# Display results
print("=" * 45)
print("FraudGuard AI - Dataset Generated")
print("=" * 45)

print("\nTotal records:", len(df))
print("\nDataset shape:", df.shape)

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRepeated messages:")
print(df["message"].duplicated().sum())

print("\nDataset saved to:")
print(output_file)

print("\nDataset generation completed successfully!")
