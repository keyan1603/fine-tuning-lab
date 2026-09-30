# tests_offline/test_extract_mock.py
# Pure unit tests of prompt building, JSON extraction, and validation,
# no model, no API call, no mocking needed at all. This is exactly the
# logic that has to treat Gemini's output and a tiny local model's
# output by the same honest bar, so it's worth pinning down in
# isolation from either.

from scenario_ticket_extraction.extract import build_prompt, parse_extraction_response, validate_extraction


def test_build_prompt():
    prompt = build_prompt("my card was charged twice")
    assert "my card was charged twice" in prompt
    assert "category" in prompt and "severity" in prompt and "root_cause_guess" in prompt
    print("  [build_prompt embeds the ticket text and names every required field] OK")


def test_parse_extraction_response_valid():
    raw = 'Sure, here is the result:\n{"category": "billing", "severity": "medium", "root_cause_guess": "duplicate charge"}\nLet me know if you need anything else.'
    parsed = parse_extraction_response(raw)
    assert parsed == {"category": "billing", "severity": "medium", "root_cause_guess": "duplicate charge"}
    print("  [parse_extraction_response pulls JSON out of surrounding prose] OK")


def test_parse_extraction_response_invalid():
    assert "parse_error" in parse_extraction_response("")
    assert "parse_error" in parse_extraction_response("I'm not sure how to categorize this.")
    assert "parse_error" in parse_extraction_response("{not valid json at all")
    print("  [parse_extraction_response reports a structured error, not a crash, on real malformed output] OK")


def test_validate_extraction():
    valid = validate_extraction({"category": "billing", "severity": "high", "root_cause_guess": "duplicate charge"})
    assert valid["valid"] is True

    bad_category = validate_extraction({"category": "shipping", "severity": "high", "root_cause_guess": "x"})
    assert bad_category["valid"] is False and "category" in bad_category["reasons"][0]

    missing_root_cause = validate_extraction({"category": "billing", "severity": "high", "root_cause_guess": ""})
    assert missing_root_cause["valid"] is False and any("root_cause_guess" in r for r in missing_root_cause["reasons"])

    parse_failure = validate_extraction({"parse_error": "invalid JSON: x"})
    assert parse_failure["valid"] is False and parse_failure["reasons"] == ["invalid JSON: x"]
    print("  [validate_extraction catches an invalid category, a missing root cause, and a parse failure] OK")


def main():
    test_build_prompt()
    test_parse_extraction_response_valid()
    test_parse_extraction_response_invalid()
    test_validate_extraction()
    print("extract offline validation: all checks passed")


if __name__ == "__main__":
    main()
