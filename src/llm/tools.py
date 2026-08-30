"""Tool definitions for forced function calling. These are the ONLY places
models emit structured output in this system.

NORMALIZE_TOOL: decline string -> DeclineClass + confidence score
INTENT_TOOL: support message -> exit-intent score + rationale

All models run with `tool_config.function_calling_config.mode = "ANY"` so
the model MUST return a function call, never free text. Malformed JSON is
structurally impossible.
"""
from src.core.types import DeclineClass

NORMALIZE_TOOL: dict = {
    "name": "emit_decline_class",
    "description": "Classify an issuer decline string into the system's taxonomy. "
    "INSUFFICIENT_FUNDS and MANDATE_REVOKED are strictly distinct -- one is transient, "
    "one is dead. Never collapse them.",
    "parameters": {
        "type": "object",
        "properties": {
            "decline_class": {
                "type": "string",
                "enum": [c.value for c in DeclineClass],
                "description": "The normalized decline class from the taxonomy",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the classification (0.0-1.0)",
            },
        },
        "required": ["decline_class", "confidence"],
    },
}

INTENT_TOOL: dict = {
    "name": "emit_intent",
    "description": "Extract exit-intent signal from support message text, including Hinglish. "
    "Return a score in [0.0, 1.0] and a brief rationale.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent_score": {
                "type": "number",
                "description": "Exit-intent score (0.0 = wants to keep paying, 1.0 = wants out)",
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation for the score",
            },
        },
        "required": ["intent_score", "rationale"],
    },
}
