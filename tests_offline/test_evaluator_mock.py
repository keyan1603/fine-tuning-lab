# tests_offline/test_evaluator_mock.py
# score_extraction and ExtractionEvaluator tested against a fake
# pipeline_fn, no real model or API call involved.

from common.evaluator import score_extraction, ExtractionEvaluator


def test_score_extraction():
    correct = score_extraction('{"category": "billing", "severity": "high", "root_cause_guess": "double charge"}', "billing", "high")
    assert correct["valid_json"] and correct["category_correct"] and correct["severity_correct"]

    wrong_severity = score_extraction('{"category": "billing", "severity": "low", "root_cause_guess": "double charge"}', "billing", "high")
    assert wrong_severity["valid_json"] and wrong_severity["category_correct"] and not wrong_severity["severity_correct"]

    malformed = score_extraction("I think this is a billing issue", "billing", "high")
    assert not malformed["valid_json"] and not malformed["category_correct"] and not malformed["severity_correct"]
    print("  [score_extraction scores JSON validity, category, and severity independently] OK")


def _fake_pipeline_fn(ticket_text: str) -> str:
    if "refund" in ticket_text:
        return '{"category": "billing", "severity": "medium", "root_cause_guess": "refund request"}'
    return "not valid json"


def test_extraction_evaluator():
    cases = [
        {"ticket_id": "T1", "text": "I need a refund", "expected_category": "billing", "expected_severity": "medium"},
        {"ticket_id": "T2", "text": "something is broken", "expected_category": "technical", "expected_severity": "high"},
    ]
    results = ExtractionEvaluator(cases).run(_fake_pipeline_fn)
    assert results["valid_json_rate"] == 0.5
    assert results["category_accuracy"] == 0.5
    assert results["severity_accuracy"] == 0.5
    assert results["per_case"][0]["category_correct"] is True
    assert results["per_case"][1]["valid_json"] is False
    print("  [ExtractionEvaluator aggregates validity and accuracy across cases correctly] OK")


def main():
    test_score_extraction()
    test_extraction_evaluator()
    print("evaluator offline validation: all checks passed")


if __name__ == "__main__":
    main()
