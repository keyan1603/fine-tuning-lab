# run_eval.py
# Runs all 8 held-out EVAL_TICKETS through Gemini zero-shot, the
# untuned local model, and the LoRA-tuned local model, and reports
# valid_json_rate, category_accuracy, and severity_accuracy for each,
# the quantitative half of the "with and without fine-tuning" story
# run_demo.py shows qualitatively.

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from common.llm_utils import generate_with_backoff
from common.drift_monitor import save_baseline, check_drift
from common.evaluator import ExtractionEvaluator
from scenario_ticket_extraction.data import EVAL_TICKETS
from scenario_ticket_extraction.extract import build_prompt
from scenario_ticket_extraction.local_model import load_model, generate

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = "adapters/ticket-extraction-lora"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEMS = {
    "gemini": Path("baselines/gemini.json"),
    "base_local_model": Path("baselines/base_local_model.json"),
    "tuned_local_model": Path("baselines/tuned_local_model.json"),
}


def _run_one(name: str, pipeline_fn, baseline_path: Path, save: bool):
    print(f"\n=== {name} ===")
    results = ExtractionEvaluator(EVAL_TICKETS).run(pipeline_fn)
    print(f"  valid_json_rate: {results['valid_json_rate']}")
    print(f"  category_accuracy: {results['category_accuracy']}")
    print(f"  severity_accuracy: {results['severity_accuracy']}")
    print(f"  avg_latency_ms: {results['avg_latency_ms']}")
    for c in results["per_case"]:
        mark = "OK" if c["category_correct"] and c["severity_correct"] else ("PARTIAL" if c["valid_json"] else "INVALID")
        print(f"    [{mark}] {c['ticket_id']}: {c['parsed'] or c['invalid_reasons']}")

    if save:
        save_baseline(results, baseline_path)
        print(f"  saved baseline -> {baseline_path}")
    else:
        drift = check_drift(results, baseline_path)
        print(f"  drift check: {drift['status']}")
        if drift["status"] != "no_baseline":
            print(f"  deltas: {drift['deltas']}")


def main(save: bool):
    def gemini_pipeline(ticket_text: str) -> str:
        response = generate_with_backoff(client, CHAT_MODEL, build_prompt(ticket_text), temperature=0.0)
        return response.text

    _run_one("gemini", gemini_pipeline, SYSTEMS["gemini"], save)

    base_model, base_tokenizer = load_model(BASE_MODEL)

    def base_pipeline(ticket_text: str) -> str:
        return generate(base_model, base_tokenizer, build_prompt(ticket_text))

    _run_one("base_local_model", base_pipeline, SYSTEMS["base_local_model"], save)

    tuned_model, tuned_tokenizer = load_model(BASE_MODEL, adapter_path=ADAPTER_DIR)

    def tuned_pipeline(ticket_text: str) -> str:
        return generate(tuned_model, tuned_tokenizer, build_prompt(ticket_text))

    _run_one("tuned_local_model", tuned_pipeline, SYSTEMS["tuned_local_model"], save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gemini, the untuned local model, and the tuned local model on the ticket-extraction task.")
    parser.add_argument("--save-baseline", action="store_true",
                         help="Save this run's results as the new baseline instead of checking drift against the existing one.")
    args = parser.parse_args()
    main(args.save_baseline)
