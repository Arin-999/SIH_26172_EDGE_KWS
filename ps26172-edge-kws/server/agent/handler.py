"""
Intent handler / AI agent stub.

Processes ASR transcript text and returns a structured intent response.
Designed as a swap-in point for a real LLM backend (Ollama, OpenAI, etc.).

Current implementation: rule-based pattern matching for hackathon demo.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("kws.agent")


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------

# Each intent: (pattern, action, target, response_template)
_INTENT_RULES: list[tuple[re.Pattern, str, str, str]] = [
    # Lights
    (re.compile(r"\b(turn\s+on|switch\s+on|enable)\b.*\blight", re.I),
     "on", "lights", "Lights turned on."),
    (re.compile(r"\b(turn\s+off|switch\s+off|disable)\b.*\blight", re.I),
     "off", "lights", "Lights turned off."),

    # Temperature / AC
    (re.compile(r"\b(increase|raise|up)\b.*\btemp|hotter|warmer", re.I),
     "increase", "temperature", "Temperature increased."),
    (re.compile(r"\b(decrease|lower|reduce|down)\b.*\btemp|cooler|colder", re.I),
     "decrease", "temperature", "Temperature decreased."),

    # Status
    (re.compile(r"\b(status|report|state|what.s\s+happening)\b", re.I),
     "query", "system", "System is nominal. All subsystems operational."),

    # Emergency
    (re.compile(r"\b(emergency|abort|stop|halt)\b", re.I),
     "abort", "mission", "Abort command received. Initiating emergency stop."),

    # Navigation
    (re.compile(r"\b(go\s+to|navigate\s+to|move\s+to)\b", re.I),
     "navigate", "target", "Navigation command acknowledged."),

    # Camera / visuals
    (re.compile(r"\b(camera|capture|photo|snapshot|image)\b", re.I),
     "capture", "camera", "Camera snapshot triggered."),

    # Power
    (re.compile(r"\b(power\s+on|power\s+up|start)\b", re.I),
     "power_on", "system", "System power-on initiated."),
    (re.compile(r"\b(power\s+off|power\s+down|shutdown)\b", re.I),
     "power_off", "system", "System shutdown initiated."),
]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class IntentHandler:
    """Rule-based intent handler for voice commands.

    Swap the `process` method body for an LLM API call when ready.
    """

    def process(self, transcript: str) -> dict | None:
        """Process a transcript and return an intent response dict.

        Args:
            transcript: ASR output text (already stripped).

        Returns:
            Dict with intent, action, target, response, confidence fields.
            Returns None if no intent matched (unknown command).
        """
        if not transcript:
            return None

        # Try each rule
        for pattern, action, target, response in _INTENT_RULES:
            if pattern.search(transcript):
                intent_name = f"{target}_{action}"
                result = {
                    "intent": intent_name,
                    "action": action,
                    "target": target,
                    "response": response,
                    "confidence": 0.92,
                    "matched_text": transcript,
                }
                logger.info(f"Intent matched: {intent_name} — '{transcript}'")
                return result

        # No match — return unknown intent
        logger.info(f"No intent matched for: '{transcript}'")
        return {
            "intent": "unknown",
            "action": "none",
            "target": "none",
            "response": f"Command not recognized: '{transcript}'",
            "confidence": 0.0,
            "matched_text": transcript,
        }
