"""Display-only cleanup. Never change the underlying detection decisions."""

# Only merge clearly equivalent generic advice; preserve specific instructions.
_GROUPS = {
    "verify unexpected requests through independent official channels.": "verify_requests",
    "verify unexpected requests independently.": "verify_requests",
    "never enter a upi pin merely to receive money.": "receive_pin",
    "entering a upi pin is not required merely to receive money.": "receive_pin",
}


def unique_recommendations(items):
    seen = set()
    for item in items:
        normalized = " ".join(item.split()).casefold()
        key = _GROUPS.get(normalized, normalized)
        if key not in seen:
            seen.add(key)
            yield (
                "Never enter a UPI PIN merely to receive money."
                if key == "receive_pin" else item
            )
