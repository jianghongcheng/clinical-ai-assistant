"""
Ch5: QLoRA Fine-tuning on Research Papers Dataset
Model: Llama 3.2 1B (lighter than 8B, fits 3090 comfortably)
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load dataset ─────────────────────────────────────────
pairs = []
with open("finetune_dataset.jsonl") as f:
    for line in f:
        pairs.append(json.loads(line))

def format_prompt(pair):
    return f"""### Instruction:
{pair['instruction']}

### Response:
{pair['answer']}"""

dataset = Dataset.from_list([
    {"text": format_prompt(p)} for p in pairs
])
print(f"Dataset: {len(dataset)} samples")

# ── Model config ─────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# ── LoRA config ───────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./llama-research-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

print("\nStarting QLoRA fine-tuning...")
trainer.train()

# ── Save ──────────────────────────────────────────────────
model.save_pretrained("./llama-research-qlora/final")
tokenizer.save_pretrained("./llama-research-qlora/final")
print("\nModel saved to ./llama-research-qlora/final")
print("Fine-tuning complete!")
