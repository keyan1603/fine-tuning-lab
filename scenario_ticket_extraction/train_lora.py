# scenario_ticket_extraction/train_lora.py
# Real LoRA supervised fine-tuning, CPU-runnable, on the distilled
# dataset from distill.py. Loss is computed only on the completion
# tokens, not the prompt, standard SFT practice, masking the prompt with
# label id -100 (the value torch's cross-entropy loss is told to ignore).
#
# target_modules covers every linear projection in a Qwen2 decoder
# layer (attention q/k/v/o plus the MLP's gate/up/down projections),
# matching the "all-linear" default this post's research found current
# PEFT/Axolotl/Unsloth stacks favor over the older q/v-only convention,
# real, current 2026 practice, not a 2023-era default carried over
# unexamined.

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ExtractionDataset(Dataset):
    def __init__(self, examples: list, tokenizer):
        self.rows = []
        for ex in examples:
            messages = [{"role": "user", "content": ex["prompt"]}]
            # On this transformers version (5.17.0), apply_chat_template
            # returns a BatchEncoding even with no return_tensors
            # argument, not the plain list its signature implies. The
            # real list of token ids is at ["input_ids"], confirmed
            # directly, not assumed, see this post's Pitfalls section.
            prefix_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
            completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            input_ids = prefix_ids + completion_ids
            labels = [-100] * len(prefix_ids) + completion_ids
            self.rows.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _collate(batch, pad_token_id):
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids, attention_mask, labels = [], [], []
    for row in batch:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


def train_lora_adapter(base_model_name: str, examples: list, output_dir: str, num_epochs: int = 8) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ExtractionDataset(examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id or tokenizer.eos_token_id)
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def merge_and_save(base_model_name: str, adapter_path: str, merged_output_dir: str) -> None:
    """Merges the LoRA adapter into the base model's own weights and
    saves a standalone model directory, real safetensors, no adapter
    dependency, the format scenario_ticket_extraction/ollama_bridge.py
    needs for a direct Ollama import."""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = tuned_model.merge_and_unload()

    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
