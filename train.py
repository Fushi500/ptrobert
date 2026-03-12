#!/usr/bin/env python3
"""
train.py — Fine-tuning Qwen2.5-14B-Instruct cu Unsloth LoRA
RTX 5090 (32GB VRAM)

Rulare: python train.py
"""

import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Configuratie ──────────────────────────────────────────────────────────────

MODEL_NAME   = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
OUTPUT_DIR   = "./model_output"
DATASET_DIR  = "./data/dataset"
MAX_SEQ_LEN  = 2048

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0        # FIX: 0 in loc de 0.05 — optimizare maxima Unsloth

BATCH_SIZE   = 2        # FIX: 2 in loc de 4 — evita OOM pe 32GB cu 14B
GRAD_ACCUM   = 8        # FIX: 8 in loc de 4 — batch efectiv ramas 16
EPOCHS       = 3
LR           = 2e-4
WARMUP_STEPS = 100      # FIX: warmup_steps in loc de warmup_ratio (deprecated)
MAX_STEPS    = -1       # -1 = toate epocile

# Limiteaza dataset-ul — None = toate cele 40,000
# Schimba in 10000 pentru un run de ~3 ore, 20000 pentru ~6 ore
TRAIN_LIMIT  = 10000

# ── Incarcare model ───────────────────────────────────────────────────────────

print("==> Incarcare model Qwen2.5-14B cu Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    dtype          = None,
)

print("==> Aplicare LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

# ── Incarcare dataset ─────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def format_example(ex):
    return tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

print("==> Incarcare dataset...")
train_raw = load_jsonl(f"{DATASET_DIR}/train.jsonl")
val_raw   = load_jsonl(f"{DATASET_DIR}/val.jsonl")

# Aplica limita daca e setata
if TRAIN_LIMIT is not None:
    train_raw = train_raw[:TRAIN_LIMIT]

train_texts = [format_example(ex) for ex in train_raw]
val_texts   = [format_example(ex) for ex in val_raw]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

print(f"   Train: {len(train_dataset):,} exemple")
print(f"   Val:   {len(val_dataset):,} exemple")

# ── Antrenament ───────────────────────────────────────────────────────────────

print("==> Start antrenament...")
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        max_steps                   = MAX_STEPS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_steps                = WARMUP_STEPS,
        lr_scheduler_type           = "cosine",
        fp16                        = False,
        bf16                        = True,
        logging_steps               = 25,
        eval_strategy               = "no",          # FIX: fara eval — evita OOM
        save_strategy               = "steps",
        save_steps                  = 500,
        save_total_limit            = 3,
        load_best_model_at_end      = False,         # FIX: necesar cand eval="no"
        report_to                   = "none",
        seed                        = 42,
        dataloader_num_workers      = 0,
    ),
)

# resume_from_checkpoint=True — continua automat daca pica curentul
trainer_stats = trainer.train(resume_from_checkpoint=True)

print(f"\n  Antrenament gata!")
print(f"  Timp total: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"  Loss final: {trainer_stats.metrics['train_loss']:.4f}")

# Salveaza adaptorul LoRA
print("\n==> Salvare adaptor LoRA...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"  Salvat in: {OUTPUT_DIR}/lora_adapter/")
print(f"\nPasul urmator: python export_gguf.py")
