# run_ollama_integration.py
# The real enterprise-architecture proof: take the merged, LoRA-tuned
# model, attempt to make it a real local Ollama model, then call
# whatever IS running in Ollama through requisite-ai's existing
# OllamaProvider with zero changes to that provider.
#
# Real, honest outcome on this machine (Ollama 0.34.2): importing the
# tuned Qwen2 model into Ollama fails, with 2 distinct real errors
# depending on the --quantize value tried (see this post's Pitfalls
# section for both). That is reported here exactly as it happened, not
# smoothed over. The Requisite/OllamaProvider integration itself is
# still verified for real, against a model Ollama can actually run
# (llama3.2:1b, already pulled), proving the architectural claim (a
# real provider abstraction absorbs any locally-hosted model with zero
# framework code changes) independently of whether this specific tuned
# model happens to import on this specific Ollama version.

from scenario_ticket_extraction.data import EVAL_TICKETS
from scenario_ticket_extraction.extract import build_prompt
from scenario_ticket_extraction.ollama_bridge import write_modelfile, create_ollama_model, requisite_smoke_test

MERGED_DIR = "merged_model/ticket-extraction-tuned"
MODELFILE_PATH = "merged_model/Modelfile"
OLLAMA_MODEL_NAME = "ticket-extraction-tuned"
FALLBACK_OLLAMA_MODEL = "llama3.2:1b"


def main():
    print(f"Writing Modelfile pointing at {MERGED_DIR}...")
    write_modelfile(MERGED_DIR, MODELFILE_PATH)

    print(f"Attempting real `ollama create {OLLAMA_MODEL_NAME}` (no quantization, default safetensors import)...")
    result = create_ollama_model(OLLAMA_MODEL_NAME, MODELFILE_PATH, quantize=None)

    if result["success"]:
        print("Ollama model created successfully:")
        print(result["stdout"])
        integration_model = OLLAMA_MODEL_NAME
    else:
        print(f"NOT IMPORTABLE on this Ollama version (returncode={result['returncode']}), a real, confirmed limitation:")
        print(result["stderr"].strip() or result["stdout"].strip())
        print(f"\nFalling back to a model Ollama CAN run ({FALLBACK_OLLAMA_MODEL}, already pulled) "
              f"to verify the Requisite integration itself, independent of this specific model's importability.")
        integration_model = FALLBACK_OLLAMA_MODEL

    ticket = EVAL_TICKETS[0]
    print(f"\nCalling {integration_model} through requisite-ai's OllamaProvider...")
    smoke_result = requisite_smoke_test(integration_model, build_prompt(ticket["text"]))

    if smoke_result["success"]:
        print("SUCCESS, real response through requisite-ai's provider abstraction, zero framework changes:")
        print(f"  {smoke_result['content']}")
    else:
        print(f"FAILED: {smoke_result['error']}")


if __name__ == "__main__":
    main()
