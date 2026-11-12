#!/usr/bin/env bash
set -e

# ==============================================================================
# EXPECTED DATASET FORMAT (OpenAI Gold Standard Conversational JSONL)
# Each line in your .jsonl file must follow this exact schema:
# {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
# ==============================================================================

echo "=================================================="
echo "              AARFORGE BUILD WIZARD               "
echo "=================================================="

# Ensure virtual environment is active
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[-] Error: .venv not found in current directory. Please run install.sh first."
    exit 1
fi

# 1. Interactive Prompts
read -p "Enter your AI model's custom name (e.g., 'my_custom_bot'): " BOT_NAME
BOT_NAME=${BOT_NAME:-my_custom_bot}

echo ""
echo "[i] This is the expected dataset format AarForge should use (OpenAI-style JSONL):"
echo '    {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}'
echo ""

read -p "Enter path to your dataset file (.jsonl): " DATASET_INPUT
DATASET_INPUT="${DATASET_INPUT/#\~/$HOME}"
if [ ! -f "$DATASET_INPUT" ]; then
    echo "[-] Error: Dataset file '$DATASET_INPUT' does not exist!"
    exit 1
fi

# Copy dataset to working directory with a standardized name
cp "$DATASET_INPUT" "${BOT_NAME}_dataset.jsonl"
echo "[+] Dataset copied to local workspace as ${BOT_NAME}_dataset.jsonl"

read -p "Enter Base Model ID [Default: unsloth/Qwen2.5-3B-Instruct-bnb-4bit]: " BASE_MODEL
BASE_MODEL=${BASE_MODEL:-unsloth/Qwen2.5-3B-Instruct-bnb-4bit}

read -p "Enter training steps [Default: 60]: " TRAIN_STEPS
TRAIN_STEPS=${TRAIN_STEPS:-60}

read -p "Enter system prompt / description for your AI model (used for Ollama Modelfile): " MODEL_SYSTEM_PROMPT
MODEL_SYSTEM_PROMPT=${MODEL_SYSTEM_PROMPT:-"You are a helpful assistant trained with AarForge."}

read -p "Enter temperature for your AI model [Default: 0.7]: " MODEL_TEMP
MODEL_TEMP=${MODEL_TEMP:-0.7}

# 2. Generate train_bot.py dynamically using FastLanguageModel (Unsloth unified pipeline)
echo "[+] Generating train_bot.py..."
cat << EOF > train_bot.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel

model_id = "$BASE_MODEL"
dataset_path = "${BOT_NAME}_dataset.jsonl"
output_adapter_dir = "${BOT_NAME}_adapter"
max_seq_length = 512

print("[+] Loading model and tokenizer via Unsloth FastLanguageModel...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

print("[+] Setting up PEFT LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"[+] Loading OpenAI-style dataset from {dataset_path}...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

if "messages" not in dataset.column_names:
    raise ValueError(
        "Invalid dataset format! AarForge expects the OpenAI gold standard JSONL format "
        "with a top-level 'messages' key: {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}"
    )

def tokenize_function(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=$TRAIN_STEPS,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

print("[+] Starting training run (VRAM Optimized)...")
trainer.train()

print(f"[+] Training complete! Saving adapter weights to ./{output_adapter_dir}...")
model.save_pretrained(output_adapter_dir)
tokenizer.save_pretrained(output_adapter_dir)
EOF

# 3. Generate export_gguf.py dynamically (merges LoRA + quantizes to GGUF in one pass)
echo "[+] Generating export_gguf.py..."
cat << EOF > export_gguf.py
from unsloth import FastLanguageModel

adapter_dir = "${BOT_NAME}_adapter"
gguf_dir = "${BOT_NAME}_gguf"

print(f"[+] Loading trained LoRA adapter from {adapter_dir}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=adapter_dir,
    max_seq_length=512,
    load_in_4bit=True,
)

print("[+] Merging adapter and exporting directly to GGUF (q4_k_m)...")
model.save_pretrained_gguf(
    gguf_dir,
    tokenizer,
    quantization_method="q4_k_m",
)
print("[+] Merge & GGUF export complete!")
EOF

# 4. Execute Training & Export Pipeline
echo "=================================================="
echo "          STAGE 1: TRAINING MODEL                 "
echo "=================================================="
python train_bot.py

echo "=================================================="
echo "     STAGE 2: MERGING & EXPORTING TO GGUF         "
echo "=================================================="
python export_gguf.py

# 5. Auto-configure Ollama
echo "=================================================="
echo "          STAGE 3: CONFIGURING OLLAMA             "
echo "=================================================="

# Locate the generated GGUF file
GGUF_FILE=$(find "${BOT_NAME}_gguf" -name "*.gguf" | head -n 1)

if [ -z "$GGUF_FILE" ]; then
    echo "[-] Warning: GGUF file could not be automatically located. Skipping Ollama configuration."
else
    echo "[+] Found GGUF file: $GGUF_FILE"

    # Build Modelfile with proper ChatML template & stop tokens matching training format
    cat << MODELFILE_EOF > Modelfile
FROM ./$GGUF_FILE

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range \$i, \$_ := .Messages }}{{- \$last := eq (len (slice \$.Messages \$i)) 1 -}}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not \$last }}<|im_end|>
{{ end }}{{- end }}{{- if and (ne .Role "assistant") \$last }}<|im_start|>assistant
{{ end }}{{- end }}"""

SYSTEM """$MODEL_SYSTEM_PROMPT"""
PARAMETER temperature $MODEL_TEMP
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
MODELFILE_EOF

    echo "[+] Creating Ollama model: ${BOT_NAME}..."
    ollama create "$BOT_NAME" -f Modelfile

    echo "=================================================="
    echo "          FORGING COMPLETE SUCCESSFULLY!          "
    echo "=================================================="
    echo "To run your model instantly, type:"
    echo "  ollama run $BOT_NAME"
    echo "=================================================="
fi
