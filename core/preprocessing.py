
import re
import unicodedata


def preprocess_text(text):
    """
    Normalize user input before fraud analysis.
    Preserve URLs, numbers and meaningful punctuation.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible formatting characters
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cf"
    )

    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_input(text):
    """
    Validate user input before model prediction.
    """

    if not isinstance(text, str):
        return False, "Input must be text."

    if not text.strip():
        return False, "Please enter a message."

    cleaned = preprocess_text(text)

    if len(cleaned) < 3:
        return False, "Input is too short."

    if len(text) > 10000:
        return False, "Input exceeds 10000 characters."

    return True, "Valid input."


# Test preprocessing independently
if __name__ == "__main__":

    sample_messages = [
        "  URGENT!!! Update your KYC immediately.  ",
        "Enter your UPI PIN to receive Rs. 5000.",
        "Verify your account at https://example.com",
        "Never share your OTP with anyone.",
        "Your class starts at 9 AM tomorrow.",
        "",
        "   ",
        "Hi"
    ]

    print("=" * 55)
    print("FraudGuard AI - Text Preprocessing Test")
    print("=" * 55)

    for message in sample_messages:

        valid, status = validate_input(message)

        print("\nOriginal:", repr(message))
        print("Valid:", valid)
        print("Status:", status)

        if valid:
            print("Processed:", preprocess_text(message))

    print("\nPreprocessing tests completed!")
