
import re
from decimal import Decimal, InvalidOperation

from core.application_detector import analyze_application, decision_risk_level


# Basic UPI address structure.
# This does not verify account existence or ownership.
UPI_PATTERN = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
    r"@[a-zA-Z0-9][a-zA-Z0-9.-]*"
)

VALID_PAYMENT_TYPES = [
    "Send Money",
    "Receive Money",
    "Collect Request"
]


def validate_upi_id(upi_id):
    """
    Check the basic format of a UPI ID.
    """

    if not isinstance(upi_id, str):
        return False, "UPI ID must be text."

    upi_id = upi_id.strip()

    if not upi_id:
        return False, "Please enter a UPI ID."

    if not UPI_PATTERN.fullmatch(upi_id):
        return False, "Invalid basic UPI ID format."

    return True, (
        "Basic format appears valid. "
        "Account existence and ownership are unverified."
    )


def validate_amount(amount):
    """
    Validate the transaction amount.
    """

    try:
        value = Decimal(str(amount))

        if not value.is_finite():
            return False, "Amount must be finite."

        if value <= 0:
            return False, "Amount must be greater than zero."

        if value != value.quantize(Decimal("0.01")):
            return False, "Use a maximum of two decimal places."

        return True, "Valid transaction amount."

    except (InvalidOperation, ValueError, TypeError):

        return False, "Invalid transaction amount."


def analyze_upi_payment(
    upi_id,
    amount,
    payment_type,
    message=""
):
    """
    Analyse payment-related fraud indicators.

    This function does not connect to a bank
    or perform an actual financial transaction.
    """

    # Validate UPI ID
    valid_id, id_status = validate_upi_id(upi_id)

    if not valid_id:
        return {
            "success": False,
            "error": id_status
        }

    # Validate transaction amount
    valid_amount, amount_status = validate_amount(amount)

    if not valid_amount:
        return {
            "success": False,
            "error": amount_status
        }

    # Validate payment type
    if payment_type not in VALID_PAYMENT_TYPES:
        return {
            "success": False,
            "error": "Invalid payment type."
        }

    if not isinstance(message, str):
        return {
            "success": False,
            "error": "Payment message must be text."
        }

    amount = Decimal(str(amount))

    findings = []
    recommendations = []
    transaction_score = None
    message_analysis = None
    strong_indicator = False

    # Basic transaction-related findings
    findings.append(
        "UPI ID ownership has not been verified."
    )

    if payment_type == "Collect Request":

        findings.append(
            "Approving a collect request can debit your account."
        )

        transaction_score = 40

        recommendations.append(
            "Approve only if you intend to send money "
            "to the verified requester."
        )

    elif payment_type == "Receive Money":

        recommendations.append(
            "Entering a UPI PIN is not required "
            "merely to receive money."
        )

    elif payment_type == "Send Money":

        recommendations.append(
            "Confirm the recipient name and payment "
            "amount in your trusted UPI application."
        )

    # Analyse accompanying message
    if message.strip():

        message_analysis = analyze_application(
            message,
            source="UPI"
        )

        if not message_analysis["success"]:
            return message_analysis

        indicator_names = {
            item["indicator"]
            for item in message_analysis["indicators"]
        }

        if "upi_collect_risk" in indicator_names:

            strong_indicator = True

            transaction_score = max(
                transaction_score or 0,
                85
            )

            findings.append(
                "Possible misuse of payment authorization."
            )

        if "credential_request" in indicator_names:

            strong_indicator = True

            transaction_score = max(
                transaction_score or 0,
                80
            )

            findings.append(
                "Possible request for confidential credentials."
            )

        findings.extend(
            message_analysis["explanations"]
        )

        recommendations.extend(
            message_analysis["recommendations"]
        )

    else:

        findings.append(
            "No accompanying message was provided. "
            "Fraud detection evidence is limited."
        )

    # Combine message and transaction indicators
    scores = []

    if transaction_score is not None:
        scores.append(transaction_score)

    if message_analysis is not None:

        message_score = message_analysis["risk_score"]

        if message_score is not None:
            scores.append(message_score)

    final_score = max(scores) if scores else None

    # Determine final assessment
    if final_score is None:

        risk_level = "Insufficient Information"
        decision = "Insufficient Information"

    else:

        final_score = round(final_score, 2)

        if strong_indicator:

            decision = "Potential Fraud"

        elif (
            message_analysis is not None
            and message_analysis["decision"] == "Potential Fraud"
        ):

            decision = "Potential Fraud"

        else:

            decision = "Manual Verification Required"

    risk_level = decision_risk_level(decision, final_score)

    # General safety recommendations
    recommendations.extend([
        "Never share your UPI PIN or OTP.",
        "Verify the recipient independently.",
        "Do not approve unexpected payment requests.",
        "Do not rely on a risk score alone."
    ])

    return {

        "success": True,
        "source": "UPI",
        "engine_version": "Unified-V4 + UPI checks",
        "score_basis": "Maximum of available binary and transaction-rule scores",

        "upi_id": upi_id.strip(),

        "upi_format_valid": True,

        "account_verified": False,

        "amount": str(amount),

        "payment_type": payment_type,

        "transaction_score": transaction_score,

        "risk_score": final_score,

        "risk_level": risk_level,

        "decision": decision,

        "findings": list(dict.fromkeys(findings)),

        "recommendations": list(
            dict.fromkeys(recommendations)
        ),

        "message_analysis": message_analysis,

        "disclaimer": (
            "Experimental risk assessment only. "
            "UPI account ownership, transaction authenticity "
            "and recipient reputation are not verified."
        )
    }


# Independent testing
if __name__ == "__main__":

    samples = [

        (
            "demo@sample",
            5000,
            "Receive Money",
            "Enter your UPI PIN to receive cashback."
        ),

        (
            "demo@sample",
            1000,
            "Collect Request",
            "Approve this collect request to receive your refund."
        ),

        (
            "demo@sample",
            250,
            "Send Money",
            "Payment for college books."
        ),

        (
            "invalid-upi",
            1000,
            "Send Money",
            ""
        ),

        (
            "demo@sample",
            -500,
            "Send Money",
            ""
        ),

        (
            "demo@sample",
            2000,
            "Receive Money",
            ""
        ),

        (
            "demo@sample",
            500,
            "Collect Request",
            ""
        )
    ]

    print("=" * 60)
    print("FraudGuard AI - UPI Security Analysis")
    print("=" * 60)

    for upi_id, amount, payment_type, message in samples:

        result = analyze_upi_payment(
            upi_id,
            amount,
            payment_type,
            message
        )

        print("\n" + "-" * 60)

        print("UPI ID:", upi_id)
        print("Amount:", amount)
        print("Payment Type:", payment_type)
        print("Message:", repr(message))

        if not result["success"]:

            print("Error:", result["error"])
            continue

        print(
            "UPI Format Valid:",
            result["upi_format_valid"]
        )

        print(
            "Account Verified:",
            result["account_verified"]
        )

        print(
            "Risk Score:",
            result["risk_score"]
            if result["risk_score"] is not None
            else "N/A"
        )

        print("Risk Level:", result["risk_level"])
        print("Decision:", result["decision"])

        print("\nFindings:")

        for finding in result["findings"]:
            print("-", finding)

        print("\nRecommendations:")

        for advice in result["recommendations"]:
            print("-", advice)

    print("\n" + "=" * 60)
    print("UPI security tests completed!")
    print("=" * 60)
