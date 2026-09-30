# fine-tuning-lab

Companion repo for the "Fine-Tuning" post (#16) on
[karthiksolution.wordpress.com](https://karthiksolution.wordpress.com/),
in the AI/agents series.

## What's in here

A real "with and without fine-tuning" comparison, built around a support-
ticket-extraction task (turn free text into a fixed JSON schema:
category, severity, root cause), continuing the ops/support theme from
[MCP](https://karthiksolution.wordpress.com/2026/09/14/mcp-one-protocol-two-ways-to-speak-it/)
and [LLM Internals](https://karthiksolution.wordpress.com/2026/09/22/llm-internals-what-a-hosted-api-actually-lets-you-see/).

**Gemini's own Developer API fine-tuning is confirmed dead**
(`run_tuning_attempt.py`): a real `client.tunings.tune(...)` call
against `gemini-3.5-flash-lite` returns a real `501 UNIMPLEMENTED`,
consistent with Google's own changelog ("no immediate plans to bring
fine-tuning support back" as of May 2025). So this repo does real,
local, free LoRA fine-tuning instead:

- **Real distillation** (`scenario_ticket_extraction/distill.py`):
  Gemini's own zero-shot output on 14 training tickets becomes the LoRA
  fine-tuning target for a small local model, a real, current enterprise
  pattern (train a cheap model to match an expensive one's behavior on a
  narrow task).
- **Real LoRA fine-tuning** (`scenario_ticket_extraction/train_lora.py`):
  CPU-runnable supervised fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct`
  via HuggingFace `transformers`/`peft`, real training loss, real
  before/after outputs.
- **A real 3-way comparison** (`run_demo.py`, `run_eval.py`): Gemini
  zero-shot, the untuned base model, and the LoRA-tuned model, scored on
  the exact same 8 held-out tickets for JSON validity, category
  accuracy, and severity accuracy.
- **A real Requisite-ai integration attempt**
  (`scenario_ticket_extraction/ollama_bridge.py`, `run_ollama_integration.py`):
  the merged, tuned model gets imported into local Ollama and called
  through [requisite-ai](https://github.com/keyan1603/requisite-ai)'s
  existing `OllamaProvider`, zero framework changes, the real proof for
  this post's enterprise-architecture claim that fine-tuning is a
  model-name swap behind a real provider abstraction.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your real key. CPU-only PyTorch
is enough, no GPU required, though CPU training is real minutes, not
seconds, see the post's Walkthrough for actual timing.

## Real results from this repo's own runs

- **Gemini's own tuning API**: confirmed `501 UNIMPLEMENTED`, not implemented on `gemini-3.5-flash-lite`.
- **LoRA training**: 4,399,104 trainable params out of 498,431,872 (0.8826%), 986.3s (~16.4 min) on CPU, loss 1.133 -> 0.004 over 56 steps.
- **3-way eval, 8 held-out tickets**: Gemini 100% category accuracy / 87.5% severity accuracy; untuned base model 37.5% / 25%; LoRA-tuned model 87.5% / 62.5%.
- **Ollama import**: this specific Qwen2 model is NOT importable on Ollama 0.34.2 (`unsupported MLX architecture: model "Qwen2ForCausalLM"`, 2 distinct real errors along the way, see the post's Pitfalls). The Requisite `OllamaProvider` integration itself was still verified for real against `llama3.2:1b`.

## Running everything, in order

```powershell
python run_tuning_attempt.py       # real, confirmed 501 from Gemini's own tuning API
python run_train.py                # real distillation + LoRA training + merge (long-running)
python run_demo.py                 # qualitative before/after walkthrough
python run_eval.py --save-baseline # quantitative 3-way comparison
python run_ollama_integration.py   # real Ollama import + requisite-ai OllamaProvider call
```

## Running the offline validation (no API key or model download required)

```powershell
python -m tests_offline.test_data_mock
python -m tests_offline.test_extract_mock
python -m tests_offline.test_evaluator_mock
```

## License

MIT, see [LICENSE](LICENSE).
