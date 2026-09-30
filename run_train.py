# run_train.py
# The real training pipeline: distill Gemini's own zero-shot behavior on
# TRAIN_TICKETS into a labeled dataset, then LoRA-tune the local base
# model on it, then merge the adapter into a standalone model directory.
# Kept separate from run_demo.py/run_eval.py since this step is the
# long-running one (real wall-clock minutes on a CPU-only machine, not
# something to re-run casually).

import os
from dotenv import load_dotenv
from google import genai
from common.tracer import Tracer
from scenario_ticket_extraction.distill import build_distilled_dataset, save_distilled_dataset
from scenario_ticket_extraction.train_lora import train_lora_adapter, merge_and_save

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = "adapters/ticket-extraction-lora"
MERGED_DIR = "merged_model/ticket-extraction-tuned"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main():
    tracer = Tracer()

    print(f"Distilling {CHAT_MODEL}'s own zero-shot behavior into training examples...")
    with tracer.span("build_distilled_dataset", model=CHAT_MODEL) as meta:
        dataset, skipped = build_distilled_dataset(client, CHAT_MODEL)
        meta["kept"] = len(dataset)
        meta["skipped"] = len(skipped)
    save_distilled_dataset(dataset)
    print(f"  kept {len(dataset)} examples, skipped {len(skipped)} (teacher itself produced invalid JSON)")
    if skipped:
        for s in skipped:
            print(f"    skipped {s['ticket_id']}: {s['reasons']}")

    print(f"\nLoRA fine-tuning {BASE_MODEL} on {len(dataset)} distilled examples...")
    with tracer.span("train_lora_adapter", base_model=BASE_MODEL, num_examples=len(dataset)):
        train_lora_adapter(BASE_MODEL, dataset, ADAPTER_DIR)
    print(f"  adapter saved -> {ADAPTER_DIR}")

    print(f"\nMerging adapter into a standalone model at {MERGED_DIR}...")
    with tracer.span("merge_and_save", base_model=BASE_MODEL):
        merge_and_save(BASE_MODEL, ADAPTER_DIR, MERGED_DIR)
    print(f"  merged model saved -> {MERGED_DIR}")


if __name__ == "__main__":
    main()
