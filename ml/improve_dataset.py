
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load existing cleaned dataset
df = pd.read_csv(DATA_DIR / "cleaned_dataset.csv")

# Additional diverse training examples
new_examples = {

    "Digital Arrest": [
        "An unknown official threatened to put me in jail.",
        "A caller demanded a financial settlement to cancel a warrant.",
        "Someone impersonating law enforcement requested a deposit.",
        "The caller threatened imprisonment over an imaginary offence.",
        "A suspicious officer asked me to remain on a video call.",
        "Someone demanded money to prevent my detention.",
        "A person claiming government authority requested my savings.",
        "I was threatened with prosecution unless I made a payment.",
        "An alleged investigator demanded confidential financial details.",
        "A caller instructed me to keep a supposed legal case secret.",
        "Someone demanded a security deposit to clear my criminal record.",
        "A fake official threatened legal consequences over my identity."
    ],

    "KYC Fraud": [
        "Someone impersonating bank staff asked for my password.",
        "A suspicious verification page requested confidential details.",
        "The sender threatened to suspend my banking access.",
        "An unknown caller asked for my account security code.",
        "A message demanded sensitive documents through an unofficial website.",
        "A supposed bank employee requested my card number and CVV.",
        "The caller asked for confidential credentials to renew my identity.",
        "A suspicious message requested an authentication code.",
        "Someone demanded my banking password to reactivate services.",
        "An unofficial verification form requested private financial information.",
        "The sender threatened to freeze my account unless I shared credentials.",
        "A fake customer service agent requested personal banking information."
    ],

    "Courier Fraud": [
        "Someone impersonating a delivery company demanded a penalty.",
        "An unknown caller claimed my package was involved in a crime.",
        "A supposed customs officer requested money for a shipment.",
        "The caller threatened prosecution over an unidentified package.",
        "Someone demanded a settlement over goods I did not purchase.",
        "A suspicious delivery representative requested financial credentials.",
        "An alleged shipping agent demanded an unofficial clearance fee.",
        "A caller claimed illegal goods were registered under my identity.",
        "Someone threatened detention over an international package.",
        "A fake delivery representative demanded a security deposit.",
        "An unknown sender requested money over a seized shipment.",
        "A caller demanded compensation for an alleged shipping offence."
    ],

    "UPI Fraud": [
        "Someone asked me to authorise a debit to collect a reward.",
        "A suspicious buyer requested my confidential payment code.",
        "An unknown merchant demanded advance money for reimbursement.",
        "The caller asked me to approve a transaction to receive funds.",
        "A stranger requested payment credentials to issue compensation.",
        "Someone demanded a processing fee to release prize money.",
        "A suspicious payment request was presented as an incoming refund.",
        "An unknown customer asked me to scan a payment QR code.",
        "A fake support representative requested my transaction password.",
        "Someone asked me to confirm an outgoing payment to receive a reward.",
        "The sender demanded money before releasing promised cashback.",
        "An unknown buyer requested a payment PIN for receiving funds."
    ],

    "Legitimate": [
        "Never disclose your banking password to anyone.",
        "Police advise citizens to report suspicious calls.",
        "Visit your nearest bank branch to complete identity verification.",
        "Your university library opens at ten tomorrow.",
        "The cybersecurity workshop will explain common financial scams.",
        "Do not approve unknown payment requests.",
        "Your shipment tracking information is available in the official app.",
        "Never enter your UPI PIN to receive money.",
        "The bank has published a fraud awareness notification.",
        "Your official monthly account statement is ready.",
        "Report suspicious payment requests through official channels.",
        "The college is conducting a digital safety seminar.",
        "Your delivery is scheduled for tomorrow afternoon.",
        "Do not transfer money to unknown callers.",
        "Your examination schedule has been published.",
        "Contact the police if someone threatens you.",
        "Never share an OTP with customer support agents.",
        "Your refund has been credited successfully.",
        "Verify suspicious banking messages independently.",
        "Your appointment has been rescheduled.",
        "The bank warns customers about unofficial verification links.",
        "Your electricity payment receipt is available.",
        "Do not scan unknown QR codes to receive rewards.",
        "The government has released a cybercrime awareness campaign."
    ]
}

records = []

# Create additional labelled records
for category, messages in new_examples.items():

    label = 0 if category == "Legitimate" else 1

    for message in messages:

        records.append({
            "message": message,
            "source": "Unspecified",
            "category": category,
            "label": label,
            "origin": "synthetic_v2"
        })

# Convert additional examples to DataFrame
new_df = pd.DataFrame(records)

# Combine original and additional data
combined_df = pd.concat(
    [df, new_df],
    ignore_index=True
)

# Normalize text
combined_df["message"] = (
    combined_df["message"]
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Remove contradictory labels
conflicts = combined_df.groupby("message")["category"].nunique()
conflicting_messages = conflicts[conflicts > 1].index

combined_df = combined_df[
    ~combined_df["message"].isin(conflicting_messages)
]

# Remove duplicate messages
combined_df = combined_df.drop_duplicates(
    subset=["message"]
)

# Shuffle dataset
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save improved dataset separately
output_file = DATA_DIR / "improved_dataset.csv"

combined_df.to_csv(output_file, index=False)

# Display results
print("=" * 50)
print("FraudGuard AI - Dataset Improvement V2")
print("=" * 50)

print("\nOriginal training records:", len(df))
print("Additional examples:", len(new_df))
print("Final training records:", len(combined_df))

print("\nCategory distribution:")
print(combined_df["category"].value_counts())

print("\nData origin distribution:")
print(combined_df["origin"].value_counts())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nDuplicate messages:",
      combined_df["message"].duplicated().sum())

print("\nDataset saved to:")
print(output_file)

print("\nDataset improvement completed successfully!")
