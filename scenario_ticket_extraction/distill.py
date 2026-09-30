# scenario_ticket_extraction/distill.py
# Real distillation: Gemini's own zero-shot output on TRAIN_TICKETS
# becomes the fine-tuning target for the local model. Only examples
# Gemini itself answered validly are kept, training a small model to
# imitate an invalid teacher answer would just teach it a second way to
# be wrong.

import json
from pathlib import Path

from common.llm_utils import generate_with_backoff
from scenario_ticket_extraction.data import TRAIN_TICKETS
from scenario_ticket_extraction.extract import build_prompt, parse_extraction_response, validate_extraction

DISTILLED_DATASET_PATH = Path("distilled_dataset.json")


def build_distilled_dataset(client, model: str) -> list:
    """Returns a list of {"ticket_id", "prompt", "completion"} dicts,
    completion is Gemini's own validated JSON, re-serialized compactly
    so every training example has a consistent, minimal target format
    rather than whatever whitespace Gemini happened to produce."""
    dataset = []
    skipped = []

    for ticket in TRAIN_TICKETS:
        prompt = build_prompt(ticket["text"])
        response = generate_with_backoff(client, model, prompt, temperature=0.0)
        parsed = parse_extraction_response(response.text)
        validity = validate_extraction(parsed)

        if not validity["valid"]:
            skipped.append({"ticket_id": ticket["ticket_id"], "reasons": validity["reasons"]})
            continue

        completion = json.dumps({
            "category": parsed["category"],
            "severity": parsed["severity"],
            "root_cause_guess": parsed["root_cause_guess"]
        })
        dataset.append({"ticket_id": ticket["ticket_id"], "prompt": prompt, "completion": completion})

    return dataset, skipped


def save_distilled_dataset(dataset: list, path: Path = DISTILLED_DATASET_PATH):
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_distilled_dataset(path: Path = DISTILLED_DATASET_PATH) -> list:
    with open(path) as f:
        return json.load(f)
