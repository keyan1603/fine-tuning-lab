# common/evaluator.py
# Scores every system (Gemini zero-shot, untuned local model, tuned
# local model) on the exact same 3 things: did it produce valid JSON at
# all, did it get the category right, did it get the severity right.
# root_cause_guess is deliberately NOT scored for correctness, it's a
# free-text phrase with no single right answer, only whether it's
# present is checked (inside validate_extraction), scoring it further
# would be fake precision on a subjective field.

import time

from scenario_ticket_extraction.extract import parse_extraction_response, validate_extraction


def score_extraction(raw_text: str, expected_category: str, expected_severity: str) -> dict:
    parsed = parse_extraction_response(raw_text)
    validity = validate_extraction(parsed)

    return {
        "valid_json": validity["valid"],
        "category_correct": validity["valid"] and parsed.get("category") == expected_category,
        "severity_correct": validity["valid"] and parsed.get("severity") == expected_severity,
        "parsed": parsed if validity["valid"] else None,
        "invalid_reasons": validity["reasons"] if not validity["valid"] else None
    }


class ExtractionEvaluator:
    def __init__(self, cases: list):
        """cases: list of {"ticket_id", "text", "expected_category", "expected_severity"}."""
        self.cases = cases

    def run(self, pipeline_fn) -> dict:
        """pipeline_fn(ticket_text: str) -> raw_text: str, the model's
        raw response, before parsing. Keeping parsing inside the
        evaluator (not the pipeline_fn) means every system is parsed
        and scored by the exact same logic."""
        per_case = []
        for case in self.cases:
            start = time.monotonic()
            raw_text = pipeline_fn(case["text"])
            latency_ms = round((time.monotonic() - start) * 1000, 2)

            result = score_extraction(raw_text, case["expected_category"], case["expected_severity"])
            per_case.append({
                "ticket_id": case["ticket_id"],
                "valid_json": result["valid_json"],
                "category_correct": result["category_correct"],
                "severity_correct": result["severity_correct"],
                "parsed": result["parsed"],
                "invalid_reasons": result["invalid_reasons"],
                "latency_ms": latency_ms
            })

        n = len(per_case)
        return {
            "valid_json_rate": sum(1 for c in per_case if c["valid_json"]) / n if n else None,
            "category_accuracy": sum(1 for c in per_case if c["category_correct"]) / n if n else None,
            "severity_accuracy": sum(1 for c in per_case if c["severity_correct"]) / n if n else None,
            "avg_latency_ms": sum(c["latency_ms"] for c in per_case) / n if n else None,
            "per_case": per_case
        }
