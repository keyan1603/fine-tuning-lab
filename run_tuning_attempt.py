# run_tuning_attempt.py
# The real, honest first step: attempt Gemini's own Developer API tuning
# surface before pivoting to local LoRA. This is expected to fail on
# every current model (confirmed via direct research against
# ai.google.dev's own changelog before this repo was built, see
# README.md and this post's Pitfalls section), the point of running it
# for real anyway is to show the actual error, not assume it.

import os
from dotenv import load_dotenv
from google import genai
from common.llm_utils import attempt_gemini_tuning
from common.tracer import Tracer
from scenario_ticket_extraction.data import TRAIN_TICKETS
from scenario_ticket_extraction.extract import build_prompt

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main():
    tracer = Tracer()
    examples = [
        {
            "text_input": build_prompt(t["text"]),
            "output": '{"category": "billing", "severity": "medium", "root_cause_guess": "placeholder"}'
        }
        for t in TRAIN_TICKETS[:2]
    ]

    print(f"Attempting a real tuning job against base_model={CHAT_MODEL} via the Developer API...")
    with tracer.span("attempt_gemini_tuning", base_model=CHAT_MODEL) as meta:
        result = attempt_gemini_tuning(client, CHAT_MODEL, examples)
        meta["submitted"] = result["submitted"]

    if result["submitted"]:
        print("SUBMITTED, unexpected given this repo's research, real job object:")
        print(result["job"])
    else:
        print("NOT SUBMITTED, a real, confirmed failure:")
        print(f"  {result['error'][:400]}")


if __name__ == "__main__":
    main()
