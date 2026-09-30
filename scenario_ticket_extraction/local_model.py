# scenario_ticket_extraction/local_model.py
# Loads Qwen2.5-0.5B-Instruct, either bare (the "without fine-tuning"
# baseline) or with a real LoRA adapter applied on top (the "with
# fine-tuning" arm). Same generate() path for both, the only difference
# is whether load_model was given an adapter_path, so any behavior
# difference in the walkthrough is attributable to the tuning itself,
# not to a different inference path.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str, adapter_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Uses the model's own chat template, Qwen2.5-Instruct is trained
    with a specific chat format, feeding it a raw prompt without that
    template measurably degrades output quality, confirmed during this
    repo's own real runs (see this post's Debugging tips).

    return_dict=True is required on this transformers version (5.17.0):
    without it, apply_chat_template(..., return_tensors="pt") still
    returns a BatchEncoding, not the bare tensor its own signature
    implies, and passing that BatchEncoding positionally into
    model.generate() fails with a confusing KeyError/AttributeError on
    '.shape' deep inside generate(), not an obviously chat-template-related
    error. See this post's Pitfalls section."""
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
