# scenario_ticket_extraction/extract.py
# The one prompt template every system in this repo answers: Gemini
# zero-shot, the untuned local model, and the LoRA-tuned local model all
# get the exact same instruction, so the comparison is fair. Parsing and
# validation are shared too, the same honest bar applies to a $0
# hosted-model call and a $0 local model, no separate leniency for
# either side.

import json
import re

from scenario_ticket_extraction.data import VALID_CATEGORIES, VALID_SEVERITIES

EXTRACTION_PROMPT_TEMPLATE = (
    "You are a support ticket triage assistant. Given a ticket, respond with ONLY "
    "a single compact JSON object with exactly these fields: "
    '"category" (one of billing, technical, account), '
    '"severity" (one of low, medium, high), '
    '"root_cause_guess" (a short phrase, under 8 words). '
    "No other text before or after the JSON.\n\nTicket: {ticket_text}"
)


def build_prompt(ticket_text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(ticket_text=ticket_text)


def parse_extraction_response(raw_text: str) -> dict:
    """Real models, especially a small untuned one, don't reliably
    return bare JSON, they wrap it in prose or code fences. This pulls
    the first {...} block out before parsing, and reports a structured
    parse failure rather than raising, since 'did it even produce valid
    JSON' is itself one of the things this repo measures."""
    if not raw_text:
        return {"parse_error": "empty response"}

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {"parse_error": "no JSON object found in response", "raw": raw_text[:200]}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"parse_error": f"invalid JSON: {e}", "raw": match.group(0)[:200]}

    if not isinstance(parsed, dict):
        return {"parse_error": "parsed JSON is not an object", "raw": match.group(0)[:200]}

    return parsed


def validate_extraction(parsed: dict) -> dict:
    """Returns {"valid": bool, "reasons": [...]}. A response only
    counts as fully valid if it parsed as JSON AND every field is
    present with an in-range value, partial credit isn't real credit
    for a system meant to feed a downstream, schema-trusting consumer."""
    if "parse_error" in parsed:
        return {"valid": False, "reasons": [parsed["parse_error"]]}

    reasons = []
    category = parsed.get("category")
    severity = parsed.get("severity")
    root_cause = parsed.get("root_cause_guess")

    if category not in VALID_CATEGORIES:
        reasons.append(f"category '{category}' not in {VALID_CATEGORIES}")
    if severity not in VALID_SEVERITIES:
        reasons.append(f"severity '{severity}' not in {VALID_SEVERITIES}")
    if not isinstance(root_cause, str) or not root_cause.strip():
        reasons.append("root_cause_guess missing or empty")

    return {"valid": len(reasons) == 0, "reasons": reasons}
