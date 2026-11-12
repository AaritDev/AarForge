#!/usr/bin/env bash
# AarForge build pipeline
# Requires Linux, Bash 4+, an installed AarForge .venv, NVIDIA CUDA, and Ollama.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PY="${SCRIPT_DIR}/.venv/bin/python"
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

log()  { printf '[+] %s\n' "$*"; }
info() { printf '[i] %s\n' "$*"; }
die()  { printf '[-] %s\n' "$*" >&2; exit 1; }

on_error() {
    local rc=$?
    printf '\n[-] Build failed (exit %s) at line %s.\n' "$rc" "${BASH_LINENO[0]:-unknown}" >&2
    printf '    The run directory and logs have been preserved for inspection.\n' >&2
    exit "$rc"
}
trap on_error ERR

expand_home() {
    case "$1" in
        '~') printf '%s' "$HOME" ;;
        '~/'*) printf '%s/%s' "$HOME" "${1#~/}" ;;
        *) printf '%s' "$1" ;;
    esac
}

# Interactive file browser for JSONL datasets. Navigate folders and select a
# visible .jsonl file, or enter a path manually if it is outside this tree.
select_jsonl_dataset() {
    local current="$PWD" choice entry i
    local -a dirs files
    while :; do
        current="$(cd -- "$current" 2>/dev/null && pwd -P)" || die "Cannot access directory: $current"
        printf '\nDataset location: %s\n' "$current"
        printf '  0) Enter a dataset path manually\n'
        printf '  u) Go to parent directory\n'
        mapfile -d '' -t dirs < <(find "$current" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
        mapfile -d '' -t files < <(find "$current" -mindepth 1 -maxdepth 1 -type f -iname '*.jsonl' -readable -print0 2>/dev/null | sort -z)
        for i in "${!dirs[@]}"; do printf '  d%d) %s/\n' "$((i+1))" "$(basename -- "${dirs[$i]}")"; done
        for i in "${!files[@]}"; do printf '  f%d) %s\n' "$((i+1))" "$(basename -- "${files[$i]}")"; done
        read -r -p 'Choose a folder/file (e.g. d1, f1, u, 0): ' choice || die 'No dataset selected.'
        case "$choice" in
            u|U) current="$(dirname -- "$current")" ;;
            0)
                read -r -p 'Enter path to .jsonl dataset: ' entry || true
                [[ -n "$entry" ]] || { info 'No path entered.'; continue; }
                entry="$(expand_home "$entry")"
                [[ -f "$entry" && -r "$entry" && "${entry,,}" == *.jsonl ]] || { info 'Path must be a readable .jsonl file.'; continue; }
                SELECTED_PATH="$(realpath -e -- "$entry")"
                return 0
                ;;
            d*)
                entry="${choice#d}"
                if [[ "$entry" =~ ^[0-9]+$ ]] && (( entry >= 1 && entry <= ${#dirs[@]} )); then current="${dirs[$((entry-1))]}"; else info 'Invalid folder selection.'; fi
                ;;
            f*)
                entry="${choice#f}"
                if [[ "$entry" =~ ^[0-9]+$ ]] && (( entry >= 1 && entry <= ${#files[@]} )); then SELECTED_PATH="${files[$((entry-1))]}"; return 0; else info 'Invalid file selection.'; fi
                ;;
            *) info 'Invalid selection.' ;;
        esac
    done
}

prompt_value() {
    local var_name="$1" prompt="$2" default="$3" value=""
    read -r -p "$prompt [$default]: " value || true
    printf -v "$var_name" '%s' "${value:-$default}"
}

printf '%s\n' '==================================================' 
printf '%s\n' '               AARFORGE BUILD WIZARD              '
printf '%s\n' '=================================================='

[[ "$(uname -s)" == "Linux" ]] || die "This build script currently supports Linux only."
[[ -x "$VENV_PY" ]] || die "Virtual environment not found at $SCRIPT_DIR/.venv. Run ./install.sh first."
command -v ollama >/dev/null 2>&1 || die "Ollama is not installed. Run ./install.sh first."
command -v flock >/dev/null 2>&1 || die "The 'flock' command is required (usually provided by util-linux)."
command -v realpath >/dev/null 2>&1 || die "The 'realpath' command is required."
command -v nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver tools were not found. AarForge's current QLoRA workflow requires an NVIDIA CUDA GPU."

# Prevent simultaneous builds from corrupting shared workspace state.
exec 9>"${SCRIPT_DIR}/.aarforge-build.lock"
flock -n 9 || die "Another AarForge build is already running in this workspace."

# Check the Python environment before asking for inputs or downloading model files.
"$VENV_PY" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("torch", "unsloth", "trl", "datasets", "transformers", "huggingface_hub", "bitsandbytes")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing packages in .venv: " + ", ".join(missing) + ". Run ./install.sh again.")
# Unsloth must be imported before Torch/Transformers/TRL so its patches apply.
import unsloth
import torch
import transformers
import trl
from packaging.version import Version
if Version(torch.__version__.split("+")[0]) >= Version("2.15.0"):
    raise SystemExit(f"Incompatible PyTorch {torch.__version__}; this AarForge environment requires torch <2.15.0. Run ./install.sh.")
if Version(transformers.__version__) > Version("5.17.0"):
    raise SystemExit(f"Incompatible Transformers {transformers.__version__}; this AarForge environment requires transformers <=5.17.0. Run ./install.sh.")
if Version(trl.__version__) > Version("1.15.0"):
    raise SystemExit(f"Incompatible TRL {trl.__version__}; this AarForge environment requires trl <=1.15.0. Run ./install.sh.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and reinstall compatible dependencies with ./install.sh.")
print(f"[+] Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
PY

# Ollama's client requires a running daemon. Check this before spending time training.
if ! ollama list >/dev/null 2>&1; then
    die "Ollama is installed but its server is not reachable. Start it with 'ollama serve' (or start the Ollama service), then rerun this script."
fi

BOT_NAME=""
prompt_value BOT_NAME "Enter your model name (lowercase letters, digits, . _ -)" "my_custom_bot"
[[ "$BOT_NAME" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || die "Invalid model name. Use 1-64 lowercase letters/digits and . _ -; the first character must be alphanumeric."
REPLACE_MODEL=0
if ollama show "$BOT_NAME" >/dev/null 2>&1; then
    replace_model=""
    read -r -p "Ollama model '$BOT_NAME' already exists. Replace it after a successful build? [y/N]: " replace_model || true
    [[ "${replace_model,,}" == "y" || "${replace_model,,}" == "yes" ]] || die "Existing Ollama model will not be overwritten. Choose another name."
    REPLACE_MODEL=1
fi

printf '\nExpected dataset format (one JSON object per line):\n'
printf '%s\n' '  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
printf '%s\n' 'Allowed roles: system, user, assistant. Each conversation must start with exactly one system message, alternate user/assistant turns, and end with assistant.'
printf '\nBrowse to the JSONL training dataset.\n'
select_jsonl_dataset
DATASET_INPUT="$SELECTED_PATH"
[[ -f "$DATASET_INPUT" && -r "$DATASET_INPUT" ]] || die "Dataset file does not exist or is not readable: $DATASET_INPUT"
[[ "${DATASET_INPUT,,}" == *.jsonl ]] || die "Dataset filename must end in .jsonl."
DATASET_SOURCE="$(realpath -e -- "$DATASET_INPUT")"

BASE_MODEL=""
prompt_value BASE_MODEL "Enter Hugging Face model ID or local model directory" "$DEFAULT_MODEL"
[[ -n "$BASE_MODEL" ]] || die "Base model cannot be empty."
BASE_MODEL="$(expand_home "$BASE_MODEL")"
if [[ -d "$BASE_MODEL" ]]; then
    BASE_MODEL="$(realpath -e -- "$BASE_MODEL")"
elif [[ "$BASE_MODEL" == */* ]]; then
    : # Treat as a Hugging Face repository ID; the loader validates it.
else
    die "Base model must be a Hugging Face repo ID (org/model) or an existing local directory."
fi

TRAIN_STEPS=""
prompt_value TRAIN_STEPS "Enter training steps (positive integer)" "60"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]{0,5}$ ]] || die "Training steps must be an integer from 1 to 999999."

MAX_SEQ_LENGTH=""
prompt_value MAX_SEQ_LENGTH "Enter maximum sequence length (128-131072)" "2048"
[[ "$MAX_SEQ_LENGTH" =~ ^[0-9]+$ ]] || die "Maximum sequence length must be an integer."
[[ "$MAX_SEQ_LENGTH" =~ ^[0-9]{1,6}$ ]] || die "Maximum sequence length must be an integer with at most six digits."
MAX_SEQ_LENGTH=$((10#$MAX_SEQ_LENGTH))
(( MAX_SEQ_LENGTH >= 128 && MAX_SEQ_LENGTH <= 131072 )) || die "Maximum sequence length must be between 128 and 131072."

MODEL_TEMP=""
prompt_value MODEL_TEMP "Enter Ollama temperature (0.0-2.0)" "0.7"
"$VENV_PY" - "$MODEL_TEMP" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("Temperature must be a number between 0.0 and 2.0.")
if not math.isfinite(value) or not 0.0 <= value <= 2.0:
    raise SystemExit("Temperature must be a finite number between 0.0 and 2.0.")
PY

MODEL_SYSTEM_PROMPT=""
read -r -p "Enter the Ollama system prompt [You are a helpful assistant trained with AarForge.]: " MODEL_SYSTEM_PROMPT || true
MODEL_SYSTEM_PROMPT="${MODEL_SYSTEM_PROMPT:-You are a helpful assistant trained with AarForge.}"
[[ "$MODEL_SYSTEM_PROMPT" != *'"""'* ]] || die 'System prompt cannot contain the triple-quote delimiter (""") used by Ollama Modelfile syntax.'

DATASET_COPY="${SCRIPT_DIR}/${BOT_NAME}_dataset.jsonl"
if [[ "$DATASET_SOURCE" != "$(realpath -m -- "$DATASET_COPY")" ]]; then
    if [[ -e "$DATASET_COPY" ]]; then
        overwrite_dataset=""
        read -r -p "$DATASET_COPY already exists. Replace it? [y/N]: " overwrite_dataset || true
        [[ "${overwrite_dataset,,}" == "y" || "${overwrite_dataset,,}" == "yes" ]] || die "Dataset copy not replaced. Choose a different bot name or remove the existing copy."
    fi
    cp -- "$DATASET_SOURCE" "$DATASET_COPY"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${SCRIPT_DIR}/runs/${BOT_NAME}-${RUN_ID}"
mkdir -p -- "$(dirname -- "$RUN_DIR")"
mkdir -- "$RUN_DIR"
ADAPTER_DIR="${RUN_DIR}/adapter"
GGUF_DIR="${RUN_DIR}/gguf"

export AARFORGE_BOT_NAME="$BOT_NAME"
export AARFORGE_BASE_MODEL="$BASE_MODEL"
export AARFORGE_DATASET="$DATASET_COPY"
export AARFORGE_RUN_DIR="$RUN_DIR"
export AARFORGE_ADAPTER_DIR="$ADAPTER_DIR"
export AARFORGE_GGUF_DIR="$GGUF_DIR"
export AARFORGE_MAX_STEPS="$TRAIN_STEPS"
export AARFORGE_MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH"
export AARFORGE_MODEL_TEMP="$MODEL_TEMP"
export AARFORGE_SYSTEM_PROMPT="$MODEL_SYSTEM_PROMPT"

log "Run directory: $RUN_DIR"
log "Generating training and export programs..."

cat > "${RUN_DIR}/train_bot.py" <<'PY'
#!/usr/bin/env python3
"""AarForge supervised fine-tuning runner. Dataset schema is intentionally strict."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

ALLOWED_ROLES = {"system", "user", "assistant"}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"[-] {message}")


def load_and_validate_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"Dataset file not found: {path}")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg}, column {exc.colno})")
                continue
            if not isinstance(record, dict) or set(record) != {"messages"}:
                errors.append(f"line {line_number}: each JSON object must contain only the top-level 'messages' key")
                continue
            messages = record["messages"]
            if not isinstance(messages, list) or not messages:
                errors.append(f"line {line_number}: 'messages' must be a non-empty array")
                continue
            valid = True
            roles: list[str] = []
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or set(message) != {"role", "content"}:
                    errors.append(f"line {line_number}, message {index + 1}: each message must contain exactly 'role' and 'content'")
                    valid = False
                    break
                role, content = message["role"], message["content"]
                if not isinstance(role, str) or role not in ALLOWED_ROLES:
                    errors.append(f"line {line_number}, message {index + 1}: unsupported role {role!r}; allowed roles are system, user, assistant")
                    valid = False
                    break
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"line {line_number}, message {index + 1}: content must be a non-empty string")
                    valid = False
                    break
                roles.append(role)
            if not valid:
                continue
            if not roles or roles[0] != "system":
                errors.append(f"line {line_number}: every conversation must start with a system message")
                continue
            if roles.count("system") != 1:
                errors.append(f"line {line_number}: exactly one system message is allowed, and it must be first")
                continue
            body_roles = roles[1:]
            if not body_roles or body_roles[0] != "user":
                errors.append(f"line {line_number}: conversation must start with a user message after its system message")
                continue
            if len(body_roles) % 2 != 0 or any(role != ("user" if i % 2 == 0 else "assistant") for i, role in enumerate(body_roles)):
                errors.append(f"line {line_number}: messages after the optional system message must alternate user, assistant and end with assistant")
                continue
            records.append(record)
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        extra = f"\n  ... and {len(errors) - 20} more error(s)" if len(errors) > 20 else ""
        fail(f"Dataset validation failed with {len(errors)} error(s):\n{preview}{extra}")
    if not records:
        fail("Dataset contains no non-empty JSONL records.")
    if len(records) < 2:
        print("[!] Dataset contains only one conversation. Training can run, but more diverse examples are strongly recommended.")
    return records


def validate_only() -> None:
    dataset_path = Path(os.environ["AARFORGE_DATASET"]).expanduser().resolve()
    records = load_and_validate_dataset(dataset_path)
    print(f"[+] Dataset validation passed: {len(records)} conversation(s), exact messages JSONL schema.")


def train() -> None:
    # Import Unsloth first; it applies compatibility patches to the HF stack.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model = os.environ["AARFORGE_BASE_MODEL"]
    dataset_path = Path(os.environ["AARFORGE_DATASET"]).expanduser().resolve()
    run_dir = Path(os.environ["AARFORGE_RUN_DIR"]).resolve()
    adapter_dir = Path(os.environ["AARFORGE_ADAPTER_DIR"]).resolve()
    max_steps = int(os.environ["AARFORGE_MAX_STEPS"])
    max_seq_length = int(os.environ["AARFORGE_MAX_SEQ_LENGTH"])
    records = load_and_validate_dataset(dataset_path)

    if not torch.cuda.is_available():
        fail("CUDA is not available to PyTorch. Check the NVIDIA driver and CUDA-enabled PyTorch installation.")
    if not hasattr(torch.cuda, "get_device_capability"):
        fail("This PyTorch installation does not expose CUDA device capability information.")

    print(f"[+] Loading tokenizer for {base_model} to check sequence lengths...")
    tokenizer_check = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=False)
    if not getattr(tokenizer_check, "chat_template", None):
        fail("The selected model has no tokenizer chat_template. Choose an instruction/chat model that provides one.")
    too_long: list[tuple[int, int]] = []
    for idx, record in enumerate(records, start=1):
        try:
            token_ids = tokenizer_check.apply_chat_template(
                record["messages"], tokenize=True, add_generation_prompt=False
            )
            token_count = len(token_ids)
        except Exception as exc:
            fail(f"Could not render conversation {idx} with the model's chat template: {type(exc).__name__}: {exc}")
        # 32-token headroom covers small template changes made by TRL's training-template patch.
        if token_count > max_seq_length - 32:
            too_long.append((idx, token_count))
    del tokenizer_check
    if too_long:
        samples = ", ".join(f"line {line}: {count} tokens" for line, count in too_long[:10])
        fail(
            f"{len(too_long)} conversation(s) exceed the configured {max_seq_length}-token limit (with a 32-token safety margin): {samples}. "
            "Increase max sequence length or shorten the affected examples. AarForge will not silently truncate your data."
        )

    print("[+] Loading model with Unsloth 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            fail("Tokenizer has neither a pad token nor an EOS token; cannot safely batch training samples.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidate_modules = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "query_key_value", "Wqkv", "c_attn", "c_proj",
    )
    target_modules = [name for name in candidate_modules if name in module_names]
    if not target_modules:
        fail(
            "Could not identify supported attention/MLP projection modules in this model. "
            "This build currently supports common decoder-only Transformer architectures; choose a supported model family."
        )
    print(f"[+] LoRA target modules: {', '.join(target_modules)}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=target_modules,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_dataset = Dataset.from_list(records)
    save_steps = max(1, min(25, max_steps))
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    training_args = SFTConfig(
        output_dir=str(run_dir / "trainer_output"),
        max_length=max_seq_length,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=3407,
        report_to="none",
        packing=False,
        assistant_only_loss=True,
    )

    print("[+] Initializing TRL SFTTrainer with assistant-only loss...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Fail closed if the selected template/trainer did not actually mask prompt tokens.
    probe = next(iter(trainer.get_train_dataloader()))
    labels = probe.get("labels")
    if labels is None:
        fail("TRL did not produce training labels; refusing to start training.")
    labels = labels.detach()
    attention = probe.get("attention_mask")
    if attention is None:
        fail("TRL did not return an attention mask; cannot verify assistant-only supervision.")
    attention = attention.detach().bool()
    supervised = int(((labels != -100) & attention).sum().item())
    masked_prompt = int(((labels == -100) & attention).sum().item())
    if supervised <= 0 or masked_prompt <= 0:
        fail(
            "Assistant-only loss verification failed: expected assistant labels plus masked system/user labels. "
            "This model's chat template may not support assistant-only training in the installed TRL version."
        )
    print(f"[+] Assistant-only loss verified (assistant target tokens: {supervised}; masked non-padding prompt tokens: {masked_prompt}).")

    print(f"[+] Training for {max_steps} optimizer step(s)...")
    result = trainer.train()
    if trainer.state.global_step < 1:
        fail("Trainer completed without performing an optimizer step.")

    adapter_dir.mkdir(parents=True, exist_ok=False)
    print(f"[+] Saving LoRA adapter to {adapter_dir}...")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = list(adapter_dir.glob("adapter_model.*"))
    if not adapter_config.is_file() or not adapter_weights:
        fail("Adapter verification failed: adapter_config.json or adapter model weights are missing.")

    dataset_hash = hashlib.sha256()
    with dataset_path.open("rb") as dataset_stream:
        for chunk in iter(lambda: dataset_stream.read(1024 * 1024), b""):
            dataset_hash.update(chunk)
    digest = dataset_hash.hexdigest()
    metadata = {
        "bot_name": os.environ["AARFORGE_BOT_NAME"],
        "base_model": base_model,
        "dataset": str(dataset_path),
        "dataset_sha256": digest,
        "conversation_count": len(records),
        "max_steps_requested": max_steps,
        "global_step": int(trainer.state.global_step),
        "max_seq_length": max_seq_length,
        "assistant_only_loss": True,
        "train_metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in result.metrics.items()},
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("[+] Training and adapter verification completed.")


if __name__ == "__main__":
    if "--validate-only" in sys.argv:
        validate_only()
    else:
        train()
PY

cat > "${RUN_DIR}/export_gguf.py" <<'PY'
#!/usr/bin/env python3
"""Export a fresh AarForge LoRA adapter to a verified GGUF file."""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    from unsloth import FastLanguageModel

    adapter_dir = Path(os.environ["AARFORGE_ADAPTER_DIR"]).resolve()
    gguf_dir = Path(os.environ["AARFORGE_GGUF_DIR"]).resolve()
    max_seq_length = int(os.environ["AARFORGE_MAX_SEQ_LENGTH"])
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"[-] Adapter configuration not found: {adapter_dir / 'adapter_config.json'}")
    gguf_dir.mkdir(parents=True, exist_ok=False)

    print(f"[+] Reloading adapter from {adapter_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("[+] Exporting merged model to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )

    gguf_files = sorted(gguf_dir.rglob("*.gguf"))
    if not gguf_files:
        raise SystemExit("[-] Export completed but no GGUF file was created.")
    canonical_path = gguf_dir / "model.gguf"
    if len(gguf_files) == 1 and gguf_files[0] != canonical_path:
        if canonical_path.exists():
            raise SystemExit(f"[-] Refusing to overwrite existing canonical GGUF file: {canonical_path}")
        gguf_files[0].rename(canonical_path)
        gguf_files = [canonical_path]
    elif len(gguf_files) > 1:
        # Multiple GGUFs are valid only when they are the complete shard set for one model.
        import re
        shard_pattern = re.compile(r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$")
        shard_info = [shard_pattern.match(path.name) for path in gguf_files]
        if any(match is None for match in shard_info):
            raise SystemExit(f"[-] Multiple non-sharded GGUF files were produced; refusing ambiguous export: {gguf_files}")
        stems = {match.group("stem") for match in shard_info if match}
        totals = {int(match.group("total")) for match in shard_info if match}
        indices = sorted(int(match.group("index")) for match in shard_info if match)
        parents = {path.parent for path in gguf_files}
        if len(stems) != 1 or len(totals) != 1 or len(parents) != 1:
            raise SystemExit("[-] GGUF shard files do not belong to one model in one directory.")
        total = next(iter(totals))
        if indices != list(range(1, total + 1)) or len(indices) != total:
            raise SystemExit(f"[-] GGUF shard set is incomplete: found shard indices {indices}, expected 1..{total}.")
    for gguf_path in gguf_files:
        if not gguf_path.is_file() or gguf_path.stat().st_size < 1024 * 1024:
            raise SystemExit(f"[-] GGUF output is missing or unexpectedly small: {gguf_path}")
        with gguf_path.open("rb") as file:
            magic = file.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"[-] Output does not have a GGUF header: {gguf_path}")

    manifest = {
        "gguf_files": [str(path) for path in gguf_files],
        "size_bytes_total": sum(path.stat().st_size for path in gguf_files),
        "quantization": "q4_k_m",
    }
    (gguf_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Verified GGUF: {gguf_path} ({gguf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
PY

chmod 600 "${RUN_DIR}/train_bot.py" "${RUN_DIR}/export_gguf.py"

log "Validating the JSONL dataset before loading any model weights..."
"$VENV_PY" "${RUN_DIR}/train_bot.py" --validate-only
log "Compiling generated Python programs..."
"$VENV_PY" -m py_compile "${RUN_DIR}/train_bot.py" "${RUN_DIR}/export_gguf.py"

printf '\n==================================================\n'
printf '%s\n' '             STAGE 1: MODEL TRAINING              '
printf '%s\n' '=================================================='
"$VENV_PY" "${RUN_DIR}/train_bot.py"

printf '\n==================================================\n'
printf '%s\n' '           STAGE 2: GGUF EXPORT                  '
printf '%s\n' '=================================================='
"$VENV_PY" "${RUN_DIR}/export_gguf.py"

GGUF_FILE="$(find "$GGUF_DIR" -type f -name '*.gguf' -print -quit)"
[[ -n "$GGUF_FILE" && -f "$GGUF_FILE" ]] || die "Verified GGUF file was not found in $GGUF_DIR."

# Ollama reads tokenizer/template metadata from the GGUF. Do not hard-code ChatML:
# different model families use different chat formats and stopping tokens.
cat > "${RUN_DIR}/write_modelfile.py" <<'PY'
import os
from pathlib import Path

path = Path(os.environ["AARFORGE_RUN_DIR"]) / "Modelfile"
gguf = Path(os.environ["AARFORGE_GGUF_DIR"]).resolve()
run_dir = Path(os.environ["AARFORGE_RUN_DIR"]).resolve()
model_files = sorted(gguf.rglob("*.gguf"))
if not model_files:
    raise SystemExit(f"No GGUF files found in {gguf}")
if len(model_files) == 1 and model_files[0].name == "model.gguf":
    from_arg = "./gguf/model.gguf"
else:
    parent_dirs = {file.parent for file in model_files}
    if len(parent_dirs) != 1:
        raise SystemExit("GGUF shard files are not located in one directory.")
    import re
    match = re.match(r"^(?P<stem>.+)-[0-9]{5}-of-[0-9]{5}\.gguf$", model_files[0].name)
    if len(model_files) > 1 and not match:
        raise SystemExit("Multiple GGUF files are not a recognized split-shard set.")
    relative_parent = next(iter(parent_dirs)).relative_to(run_dir)
    wildcard = f"{match.group('stem')}-*.gguf" if match else model_files[0].name
    from_arg = f"./{relative_parent.as_posix()}/{wildcard}"
prompt = os.environ["AARFORGE_SYSTEM_PROMPT"]
if '"""' in prompt or "\x00" in prompt:
    raise SystemExit('System prompt cannot contain the triple-quote delimiter or NUL bytes.')
temperature = os.environ["AARFORGE_MODEL_TEMP"]
content = f'FROM {from_arg}\n\nSYSTEM """{prompt}"""\n\nPARAMETER temperature {temperature}\n'
path.write_text(content, encoding="utf-8")
print(f"[+] Wrote Ollama Modelfile: {path}")
PY
"$VENV_PY" "${RUN_DIR}/write_modelfile.py"

printf '\n==================================================\n'
printf '%s\n' '           STAGE 3: OLLAMA MODEL                  '
printf '%s\n' '=================================================='
if [[ "$REPLACE_MODEL" -eq 0 ]] && ollama show "$BOT_NAME" >/dev/null 2>&1; then
    die "An Ollama model named '$BOT_NAME' appeared during this build. Refusing to overwrite it."
fi
ollama create "$BOT_NAME" -f "${RUN_DIR}/Modelfile"
ollama show "$BOT_NAME" >/dev/null

printf '\n==================================================\n'
printf '%s\n' '          FORGING COMPLETE SUCCESSFULLY           '
printf '%s\n' '=================================================='
printf 'Model:       ollama run %s\n' "$BOT_NAME"
printf 'Run artifacts: %s\n' "$RUN_DIR"
printf 'Dataset copy:  %s\n' "$DATASET_COPY"
printf '%s\n' 'The model was created and verified with ollama show.'
printf '%s\n' '=================================================='
