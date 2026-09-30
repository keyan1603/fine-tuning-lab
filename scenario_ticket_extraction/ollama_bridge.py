# scenario_ticket_extraction/ollama_bridge.py
# Takes the merged, LoRA-tuned model (real safetensors, produced by
# train_lora.merge_and_save) and makes it a real local Ollama model,
# then calls it through requisite-ai's existing OllamaProvider, zero
# changes to that provider or to requisite's own code. This is the real
# proof for this post's central enterprise-architecture claim: a
# fine-tuned model is a model-name swap behind a real provider
# abstraction, not an application-code change.
#
# Ollama 0.34.2, confirmed directly on the machine this was built on,
# can import a raw HF safetensors directory via a Modelfile's FROM line
# and quantize it itself (`ollama create NAME -f Modelfile --quantize LEVEL`),
# no separate llama.cpp GGUF conversion script needed for a supported
# architecture like Qwen2. This module makes that real attempt and
# reports the real outcome, success or failure, see this post's
# Pitfalls section for what actually happened on this run.

import subprocess
from pathlib import Path


def write_modelfile(merged_model_dir: str, modelfile_path: str, system_prompt: str = None) -> None:
    lines = [f"FROM {Path(merged_model_dir).resolve()}"]
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    Path(modelfile_path).write_text("\n".join(lines) + "\n")


def create_ollama_model(model_name: str, modelfile_path: str, quantize: str = None) -> dict:
    """Runs the real `ollama create` command as a subprocess. Returns
    {"success": bool, "stdout": str, "stderr": str, "returncode": int},
    a real, structured result either way, no assumption that this
    succeeds."""
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    if quantize:
        cmd += ["--quantize", quantize]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def requisite_smoke_test(model_name: str, ticket_prompt: str, timeout: float = 180.0) -> dict:
    """Sends one real chat message through requisite-ai's OllamaProvider
    against a local Ollama model. Returns {"success": bool, "content": str}
    or {"success": False, "error": str}.

    timeout=180.0, not BaseProvider's default 60.0: a real, confirmed
    finding on this machine, a cold Ollama model load (the model isn't
    resident in memory yet) plus a CPU generation took longer than 60s
    and hit requisite-ai's own default timeout with a real
    ProviderException, not a hang, see this post's Pitfalls section."""
    from requisite.providers.ollama_provider import OllamaProvider
    from requisite.core.interfaces import Message, Role

    try:
        provider = OllamaProvider(model=model_name, timeout=timeout)
        response = provider.chat([Message(role=Role.USER, content=ticket_prompt)])
        return {"success": True, "content": response.content}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
