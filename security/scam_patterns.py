
import re

from core.preprocessing import (
    preprocess_text,
    validate_input
)


# Fraud indicator patterns
SCAM_PATTERNS = {

    "legal_threat": {
        "patterns": [
            r"\b(?:arrest|imprisonment|detention|prosecution|custody)\b",
            r"\bcriminal case\b",
            r"\blegal action\b",
            r"\bput you in jail\b",
            r"\bcriminal charge\b"
        ],
        "weight": 20,
        "explanation": "Legal threat or intimidation language detected."
    },

    "authority_claim": {
        "patterns": [
            r"\bpolice\b",
            r"\binvestigator\b",
            r"\bgovernment official\b",
            r"\bcustoms officer\b",
            r"\bcrime branch\b",
            r"\blaw enforcement\b",
            r"\bcourt officer\b"
        ],
        "weight": 10,
        "explanation": "A claim or reference involving official authority."
    },

    "credential_request": {
        "patterns": [
            r"\b(?:share|send|enter|provide|submit|disclose|type)\b"
            r"[^.!?;]{0,60}"
            r"\b(?:otp|upi pin|password|cvv|card details|security code)\b",

            r"\b(?:requested|demanded|asked for)\b"
            r"[^.!?;]{0,60}"
            r"\b(?:otp|pin|password|credentials|cvv)\b"
        ],
        "weight": 30,
        "explanation": "Possible request for confidential credentials."
    },

    "payment_demand": {
        "patterns": [
            r"\btransfer (?:money|funds)\b",
            r"\bsend money\b",
            r"\bpay (?:a |the )?(?:penalty|fine|fee)\b",
            r"\bsecurity deposit\b",
            r"\bprocessing fee\b",
            r"\bfinancial settlement\b",
            r"\bdemanded (?:money|payment)\b",
            r"\brequested (?:money|payment)\b"
        ],
        "weight": 25,
        "explanation": "Potential financial demand detected."
    },

    "urgency": {
        "patterns": [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bright now\b",
            r"\bact now\b",
            r"\bwithin \d+ (?:minutes|hours)\b",
            r"\bfinal warning\b"
        ],
        "weight": 10,
        "explanation": "Urgency or pressure language detected."
    },

    "account_threat": {
        "patterns": [
            r"\baccount will be blocked\b",
            r"\baccount closure\b",
            r"\bfreeze (?:your|my) account\b",
            r"\bsuspend (?:your|my) account\b",
            r"\bservices will be suspended\b"
        ],
        "weight": 15,
        "explanation": "Threat involving account restrictions."
    },

    "external_link": {
        "patterns": [
            r"https?://[^\s]+",
            r"\bwww\.[^\s]+"
        ],
        "weight": 5,
        "explanation": "External URL detected. Its safety has not been verified."
    },

    "upi_collect_risk": {
        "patterns": [
            r"\b(?:enter|share|type|provide)\b"
            r"[^.!?;]{0,50}\bpin\b"
            r"[^.!?;]{0,50}\b(?:receive|cashback|reward|refund)\b",

            r"\bapprove\b[^.!?;]{0,60}"
            r"\b(?:collect request|payment request)\b"
            r"[^.!?;]{0,50}\b(?:receive|refund|reward)\b",

            r"\bscan\b[^.!?;]{0,40}\bqr\b"
            r"[^.!?;]{0,40}\b(?:receive|reward|cashback)\b"
        ],
        "weight": 30,
        "explanation": "Possible misuse of UPI payment authorisation."
    }
}


# Protective or awareness-related context
SAFETY_PATTERNS = [

    r"\b(?:never|do not|don't)\s+"
    r"(?:share|disclose|enter|send|provide|approve|transfer|scan)\b",

    r"\b(?:warns?|advises?)\b"
    r"[^.!?;]{0,80}"
    r"\b(?:fraud|scam|fake|suspicious|confidential)\b",

    r"\b(?:fraud|cybercrime|cybersecurity)\s+awareness\b",

    r"\bfinancial fraud prevention\b",

    r"\breport suspicious\b"
]


# Enhanced contextual security patterns

SCAM_PATTERNS["legal_threat"]["patterns"].extend([

    # Coercion involving calls and payment demands
    r"\b(?:stay|remain|keep)\b"
    r"[^.!?;]{0,45}"
    r"\b(?:call|video conference|meeting|line)\b"
    r"[^.!?;]{0,70}"
    r"\b(?:until|unless)\b"
    r"[^.!?;]{0,30}"
    r"\b(?:pay|transfer|send)\b",

    # Alleged customs violations
    r"\b(?:remove|clear|settle)\b"
    r"[^.!?;]{0,40}"
    r"\b(?:customs violation|customs case)\b",

    # Alleged seizure of a delivery
    r"\b(?:parcel|package|shipment|consignment)\b"
    r"[^.!?;]{0,50}"
    r"\b(?:confiscated|detained|seized)\b"
])


SCAM_PATTERNS["credential_request"]["patterns"].extend([

    r"\b(?:demand(?:s|ed)?|request(?:s|ed)?|"
    r"ask(?:s|ed)?\s+for|provide|share|send|disclose)\b"
    r"[^.!?;]{0,60}"
    r"\b(?:card security number|"
    r"upi authentication details|"
    r"banking password|verification code|"
    r"authentication code)\b"
])


SCAM_PATTERNS["payment_demand"]["patterns"].extend([

    # Investigation or clearance payment
    r"\bpay\s+(?:(?:a|an|the)\s+)?"
    r"(?:investigation|clearance|security|release)"
    r"\s+(?:fee|charge)\b",

    # Direct payment to an unverified caller
    r"\bpay\s+(?:the|this)\s+caller\s+directly\b",

    # Suspicious security payments
    r"\b(?:send|transfer)\s+"
    r"(?:(?:a|the)\s+)?"
    r"(?:security|release)\s+payment"
    r"\s+to\s+(?:the|this|your)\s+caller\b",

    # Advance-fee reward or refund requests
    r"\b(?:reward|refund|cashback|prize)\b"
    r"[^.!?;]{0,80}"
    r"\b(?:advance|upfront)\s+"
    r"(?:processing\s+)?"
    r"(?:payment|fee|charge)\b"
])


SCAM_PATTERNS["upi_collect_risk"]["patterns"].extend([

    # Refund or reward in exchange for a debit
    r"\b(?:refund|cashback|reward|compensation)\b"
    r"[^.!?;]{0,80}"
    r"\b(?:approve|approved|authorise|authorize|"
    r"authorised|authorized)\b"
    r"[^.!?;]{0,40}"
    r"\b(?:debit|withdrawal)\b"
])


SAFETY_PATTERNS.extend([

    # Advice against opening suspicious links
    r"\b(?:advises?|warns?|recommends?)\b"
    r"[^.!?;]{0,60}"
    r"\b(?:not to|against)\s+"
    r"(?:open|opening|click|clicking)\b",

    # Advice to report confidential credential requests
    r"\b(?:recommends?|advises?)\b"
    r"[^.!?;]{0,60}"
    r"\breport(?:ing)?\b"
    r"[^.!?;]{0,70}"
    r"\b(?:credentials|otp|pin|password|cvv)\b"
])


def detect_scam_patterns(message):
    """
    Detect contextual fraud indicators.

    Rule scores are heuristic indicators,
    not calibrated fraud probabilities.
    """

    valid, status = validate_input(message)

    if not valid:
        return {
            "success": False,
            "error": status
        }

    text = preprocess_text(message)

    detected_indicators = {}
    safety_context = False
    safety_matches = []

    # Analyse individual clauses to preserve context
    clauses = [
        preprocess_text(part)

        for part in re.split(
            r"(?<=[.!?;])\s+|\n+|"
            r"[,;]?\s+\b(?:but|however|yet)\b\s+",
            message,
            flags=re.IGNORECASE
        )

        if part.strip()
    ]

    for clause in clauses:

        # Identify protective instructions
        protective_clause = any(
            re.search(pattern, clause)
            for pattern in SAFETY_PATTERNS
        )

        if protective_clause:

            safety_context = True
            safety_matches.append(clause)

            # Do not suppress separate suspicious clauses
            # elsewhere in the message.
            continue

        # Search for suspicious indicators
        for indicator, config in SCAM_PATTERNS.items():

            for pattern in config["patterns"]:

                match = re.search(pattern, clause)

                if match:

                    if indicator not in detected_indicators:

                        detected_indicators[indicator] = {
                            "indicator": indicator,
                            "matched_text": match.group(),
                            "weight": config["weight"],
                            "explanation": config["explanation"]
                        }

                    break

    indicators = list(
        detected_indicators.values()
    )

    # Sum unique indicator weights
    rule_score = sum(
        item["weight"]
        for item in indicators
    )

    # Keep score within 0-100
    rule_score = min(rule_score, 100)

    explanations = [
        item["explanation"]
        for item in indicators
    ]

    if safety_context:

        explanations.append(
            "Protective or safety-awareness language detected."
        )

    if not explanations:

        explanations.append(
            "No predefined scam patterns detected. "
            "This does not guarantee safety."
        )

    return {
        "success": True,
        "rule_score": rule_score,
        "indicator_count": len(indicators),
        "indicators": indicators,
        "safety_context": safety_context,
        "safety_matches": safety_matches,
        "mixed_context": (
            safety_context and bool(indicators)
        ),
        "explanations": explanations
    }


# Independent module testing
if __name__ == "__main__":

    sample_messages = [

        "Transfer money immediately to avoid arrest.",

        "Your bank account will be blocked. Update KYC now.",

        "Pay a penalty to release your parcel.",

        "Enter your UPI PIN to receive cashback.",

        "Share your OTP immediately.",

        "Verify your account at https://example.com",

        "Never share your UPI PIN with anyone.",

        "The bank warns customers about fake verification links.",

        "Your class starts at 9 AM tomorrow.",

        "Never share your OTP. Transfer money immediately to avoid arrest.",

        "xyz qqq zzzz",

        ""
    ]

    print("=" * 60)
    print("FraudGuard AI - Scam Pattern Detection")
    print("=" * 60)

    for message in sample_messages:

        result = detect_scam_patterns(message)

        print("\n" + "-" * 60)
        print("Input:", repr(message))

        if not result["success"]:

            print("Error:", result["error"])
            continue

        print("Rule Score:", result["rule_score"])

        print(
            "Indicators Detected:",
            result["indicator_count"]
        )

        print(
            "Safety Context:",
            result["safety_context"]
        )

        print(
            "Mixed Context:",
            result["mixed_context"]
        )

        print("\nDetected Indicators:")

        for item in result["indicators"]:

            print(
                f"- {item['indicator']}: "
                f"{item['matched_text']} "
                f"(+{item['weight']})"
            )

        print("\nExplanations:")

        for explanation in result["explanations"]:

            print("-", explanation)

    print("\n" + "=" * 60)
    print("Scam pattern detection tests completed!")
    print("=" * 60)
