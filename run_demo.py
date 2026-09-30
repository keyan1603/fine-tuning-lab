# run_demo.py
# The qualitative "with and without fine-tuning" walkthrough: the same
# 3 eval tickets, answered by Gemini zero-shot, the untuned local model,
# and the LoRA-tuned local model, side by side.

import os
from dotenv import load_dotenv
from google import genai
from common.llm_utils import generate_with_backoff
from common.tracer import Tracer
from scenario_ticket_extraction.data import EVAL_TICKETS
from scenario_ticket_extraction.extract import build_prompt
from scenario_ticket_extraction.local_model import load_model, generate

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = "adapters/ticket-extraction-lora"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main():
    tracer = Tracer()
    sample_tickets = EVAL_TICKETS[:3]

    print(f"Loading base model ({BASE_MODEL}, untuned)...")
    base_model, base_tokenizer = load_model(BASE_MODEL)

    print(f"Loading tuned model ({BASE_MODEL} + LoRA adapter from {ADAPTER_DIR})...")
    tuned_model, tuned_tokenizer = load_model(BASE_MODEL, adapter_path=ADAPTER_DIR)

    for ticket in sample_tickets:
        prompt = build_prompt(ticket["text"])
        print(f"\n=== {ticket['ticket_id']}: {ticket['text']} ===")
        print(f"expected: category={ticket['expected_category']}, severity={ticket['expected_severity']}")

        with tracer.span("gemini_extract", ticket_id=ticket["ticket_id"]):
            gemini_response = generate_with_backoff(client, CHAT_MODEL, prompt, temperature=0.0)
        print(f"\nGemini ({CHAT_MODEL}):\n  {gemini_response.text.strip()}")

        with tracer.span("base_model_extract", ticket_id=ticket["ticket_id"]):
            base_output = generate(base_model, base_tokenizer, prompt)
        print(f"\nBase local model (untuned {BASE_MODEL}):\n  {base_output.strip()}")

        with tracer.span("tuned_model_extract", ticket_id=ticket["ticket_id"]):
            tuned_output = generate(tuned_model, tuned_tokenizer, prompt)
        print(f"\nTuned local model ({BASE_MODEL} + LoRA):\n  {tuned_output.strip()}")


if __name__ == "__main__":
    main()
