#!/usr/bin/env bash
# AarForge: dataset validation -> QLoRA fine-tuning -> GGUF export -> optional Ollama deployment.
# Linux, Bash 4+, a prepared .venv, NVIDIA CUDA, and Unsloth are required for training.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PY="${SCRIPT_DIR}/.venv/bin/python"
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

RUN_DIR=""
BOT_NAME=""
BUILD_STATUS="INITIALIZING"
TRAIN_LOG=""
EXPORT_LOG=""
DEPLOY_LOG=""

log()  { printf '[+] %s\n' "$*"; }
info() { printf '[i] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*" >&2; }

print_artifact_location() {
    [[ -n "$RUN_DIR" ]] && printf '    Run artifacts: %s\n' "$RUN_DIR" >&2
    [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]] && printf '    Training log: %s\n' "$TRAIN_LOG" >&2
    [[ -n "$EXPORT_LOG" && -f "$EXPORT_LOG" ]] && printf '    Export log:   %s\n' "$EXPORT_LOG" >&2
    [[ -n "$DEPLOY_LOG" && -f "$DEPLOY_LOG" ]] && printf '    Deploy log:   %s\n' "$DEPLOY_LOG" >&2
    return 0
}

die() {
    local message="$*"
    printf '\n[-] %s\n' "$message" >&2
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        printf '    The run directory and available logs have been preserved.\n' >&2
        print_artifact_location
    fi
    exit 1
}

on_error() {
    local rc=$?
    trap - ERR
    printf '\n[-] Unexpected shell error (exit %s) near line %s.\n' "$rc" "${BASH_LINENO[0]:-unknown}" >&2
    printf '    Build status: %s\n' "$BUILD_STATUS" >&2
    print_artifact_location
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

# Run a command with live output and a persistent log. Return the command's
# status (or tee's status if logging itself fails).
run_streamed() {
    local logfile="$1" rc=0
    shift
    "$@" 2>&1 | tee "$logfile" || {
        local -a pipe_status=("${PIPESTATUS[@]}")
        if (( ${pipe_status[0]:-0} != 0 )); then
            rc="${pipe_status[0]}"
        else
            rc="${pipe_status[1]:-1}"
        fi
    }
    return "$rc"
}

# Interactive JSONL selector. Folders can be browsed recursively; paths outside
# the current tree can be entered manually.
select_jsonl_dataset() {
    local current="$PWD" choice entry i redraw=0
    local -a dirs files
    while :; do
        if (( redraw )) && [[ -t 1 ]]; then printf '\033[2J\033[H'; fi
        current="$(cd -- "$current" 2>/dev/null && pwd -P)" || die "Cannot access directory: $current"
        printf '\nDataset location: %s\n' "$current"
        printf '  0) Enter a dataset path manually\n'
        printf '  u) Go to parent directory\n'
        mapfile -d '' -t dirs < <(find "$current" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
        mapfile -d '' -t files < <(find "$current" -mindepth 1 -maxdepth 1 -type f -iname '*.jsonl' -readable -print0 2>/dev/null | sort -z)
        for i in "${!dirs[@]}"; do printf '  d%d) %s/\n' "$((i+1))" "$(basename -- "${dirs[$i]}")"; done
        for i in "${!files[@]}"; do printf '  f%d) %s\n' "$((i+1))" "$(basename -- "${files[$i]}")"; done
        read -r -p 'Choose a folder/file (e.g. d1, f1, u, 0): ' choice || die 'No dataset selected.'
        redraw=1
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
                if [[ "$entry" =~ ^[0-9]+$ ]] && (( entry >= 1 && entry <= ${#dirs[@]} )); then
                    current="${dirs[$((entry-1))]}"
                else
                    info 'Invalid folder selection.'
                fi
                ;;
            f*)
                entry="${choice#f}"
                if [[ "$entry" =~ ^[0-9]+$ ]] && (( entry >= 1 && entry <= ${#files[@]} )); then
                    SELECTED_PATH="${files[$((entry-1))]}"
                    return 0
                else
                    info 'Invalid file selection.'
                fi
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

write_report() {
    [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || return 0
    local report_status="$1" note="$2"
    {
        printf 'AarForge build report\n'
        printf '====================\n'
        printf 'Status: %s\n' "$report_status"
        printf 'Model name: %s\n' "${BOT_NAME:-not set}"
        printf 'Run directory: %s\n' "$RUN_DIR"
        printf 'Base model: %s\n' "${BASE_MODEL:-not recorded}"
        printf 'Dataset: %s\n' "${DATASET_SOURCE:-not recorded}"
        printf 'Adapter directory: %s\n' "${ADAPTER_DIR:-not created}"
        printf 'GGUF output directory: %s\n' "${GGUF_DIR:-not created}"
        printf 'Main GGUF: %s\n' "${GGUF_MAIN_FILE:-not exported}"
        printf 'Vision projector(s): %s\n' "${MMPROJ_FILES_TEXT:-none detected}"
        printf 'Ollama status: %s\n' "${DEPLOYMENT_STATUS:-not attempted}"
        printf '\nNotes\n-----\n%s\n' "$note"
        printf '\nLogs\n----\n'
        printf 'Training: %s\n' "${TRAIN_LOG:-not created}"
        printf 'Export: %s\n' "${EXPORT_LOG:-not created}"
        printf 'Deployment: %s\n' "${DEPLOY_LOG:-not created}"
    } > "${RUN_DIR}/BUILD_REPORT.txt"
}

print_partial_success() {
    local reason="$1"
    BUILD_STATUS="PARTIAL SUCCESS"
    write_report "$BUILD_STATUS" "$reason"
    printf '\n==================================================\n'
    printf '%s\n' '             AARFORGE: PARTIAL SUCCESS            '
    printf '%s\n' '=================================================='
    warn "$reason"
    printf 'Run artifacts: %s\n' "$RUN_DIR"
    [[ -d "${ADAPTER_DIR:-/nonexistent}" ]] && printf 'LoRA adapter:  %s\n' "$ADAPTER_DIR"
    [[ -n "${GGUF_MAIN_FILE:-}" && -f "$GGUF_MAIN_FILE" ]] && printf 'Main GGUF:     %s\n' "$GGUF_MAIN_FILE"
    if [[ -n "${MMPROJ_FILES_TEXT:-}" ]]; then
        printf 'Vision files:  %s\n' "$MMPROJ_FILES_TEXT"
        local projector
        local -a projectors
        IFS=';' read -r -a projectors <<< "$MMPROJ_FILES_TEXT"
        printf '%s\n' 'Vision inference with llama.cpp (one command per projector):'
        for projector in "${projectors[@]}"; do
            printf '    llama-mtmd-cli -m %q --mmproj %q -p %q\n' "${GGUF_MAIN_FILE:-MAIN_MODEL_GGUF}" "$projector" 'Describe this image.'
        done
        command -v llama-mtmd-cli >/dev/null 2>&1 || printf '%s\n' '    Note: llama-mtmd-cli is not currently on PATH; install/build a compatible llama.cpp version first.'
    fi
    printf 'Build report:  %s/BUILD_REPORT.txt\n' "$RUN_DIR"
    print_artifact_location
    printf '%s\n' 'No training/export artifacts were deleted. Review the report before deployment.'
    exit 0
}

printf '%s\n' '=================================================='
printf '%s\n' '               AARFORGE BUILD WIZARD              '
printf '%s\n' '=================================================='

[[ "$(uname -s)" == "Linux" ]] || die 'This build script currently supports Linux only.'
[[ -x "$VENV_PY" ]] || die "Virtual environment not found at $SCRIPT_DIR/.venv. Run ./install.sh first."
command -v flock >/dev/null 2>&1 || die "The 'flock' command is required (usually provided by util-linux)."
command -v realpath >/dev/null 2>&1 || die "The 'realpath' command is required."
command -v nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver tools were not found. AarForge's current QLoRA workflow requires an NVIDIA CUDA GPU."

# Prevent simultaneous builds from corrupting shared workspace state.
exec 9>"${SCRIPT_DIR}/.aarforge-build.lock"
flock -n 9 || die 'Another AarForge build is already running in this workspace.'

# Confirm the virtual environment is internally consistent. Unsloth is imported first.
if ! "$VENV_PY" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("torch", "unsloth", "trl", "datasets", "transformers", "huggingface_hub", "bitsandbytes", "packaging")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing packages in .venv: " + ", ".join(missing) + ". Run ./install.sh again.")
import unsloth
import torch
import transformers
import trl
from packaging.version import Version
if Version(torch.__version__.split("+")[0]) >= Version("2.15.0"):
    raise SystemExit(f"Incompatible PyTorch {torch.__version__}; AarForge requires torch <2.15.0. Run ./install.sh.")
if Version(transformers.__version__) > Version("5.17.0"):
    raise SystemExit(f"Incompatible Transformers {transformers.__version__}; AarForge requires transformers <=5.17.0. Run ./install.sh.")
if Version(trl.__version__) > Version("1.15.0"):
    raise SystemExit(f"Incompatible TRL {trl.__version__}; AarForge requires trl <=1.15.0. Run ./install.sh.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and CUDA-enabled PyTorch installation.")
print(f"[+] Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
PY
then
    die 'Python/CUDA preflight failed. Review the message above and run ./install.sh if dependencies are inconsistent.'
fi

# Ollama is optional until Stage 3. Training and export remain useful without it.
OLLAMA_READY=0
if command -v ollama >/dev/null 2>&1; then
    if ollama list >/dev/null 2>&1; then
        OLLAMA_READY=1
        info 'Ollama is installed and reachable; deployment will be available after export.'
    else
        warn "Ollama is installed but its server is not reachable. Training/export will continue; Stage 3 will be skipped."
    fi
else
    warn 'Ollama is not installed. Training/export will continue; Stage 3 will be skipped.'
fi

BOT_NAME=""
prompt_value BOT_NAME 'Enter your model name (lowercase letters, digits, . _ -)' 'my_custom_bot'
[[ "$BOT_NAME" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || die 'Invalid model name. Use 1-64 lowercase letters/digits and . _ -; the first character must be alphanumeric.'
REPLACE_MODEL=0
if (( OLLAMA_READY )); then
    if ollama show "$BOT_NAME" >/dev/null 2>&1; then
        replace_model=''
        read -r -p "Ollama model '$BOT_NAME' already exists. Replace it only after a staged model is verified? [y/N]: " replace_model || true
        if [[ "${replace_model,,}" == 'y' || "${replace_model,,}" == 'yes' ]]; then
            REPLACE_MODEL=1
        else
            die "Existing Ollama model '$BOT_NAME' will be preserved. Rerun with a different model name."
        fi
    fi
fi

printf '\nExpected dataset format (one JSON object per line):\n'
printf '%s\n' '  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
printf '%s\n' 'Allowed roles: system, user, assistant. Each conversation must start with exactly one system message, alternate user/assistant turns, and end with assistant.'
printf '\nBrowse to the JSONL training dataset.\n'
select_jsonl_dataset
DATASET_INPUT="$SELECTED_PATH"
[[ -f "$DATASET_INPUT" && -r "$DATASET_INPUT" ]] || die "Dataset file does not exist or is not readable: $DATASET_INPUT"
[[ "${DATASET_INPUT,,}" == *.jsonl ]] || die 'Dataset filename must end in .jsonl.'
DATASET_SOURCE="$(realpath -e -- "$DATASET_INPUT")"

BASE_MODEL=""
prompt_value BASE_MODEL 'Enter Hugging Face model ID or local model directory' "$DEFAULT_MODEL"
[[ -n "$BASE_MODEL" ]] || die 'Base model cannot be empty.'
BASE_MODEL="$(expand_home "$BASE_MODEL")"
if [[ -d "$BASE_MODEL" ]]; then
    BASE_MODEL="$(realpath -e -- "$BASE_MODEL")"
elif [[ "$BASE_MODEL" == */* ]]; then
    : # Hugging Face repository ID; model loader validates access and existence.
else
    die 'Base model must be a Hugging Face repo ID (org/model) or an existing local directory.'
fi

TRAIN_EPOCHS=""
prompt_value TRAIN_EPOCHS 'Enter training epochs (positive integer)' '3'
[[ "$TRAIN_EPOCHS" =~ ^[1-9][0-9]{0,5}$ ]] || die 'Training epochs must be an integer from 1 to 999999.'

MAX_SEQ_LENGTH=""
prompt_value MAX_SEQ_LENGTH 'Enter maximum sequence length (128-131072)' '2048'
[[ "$MAX_SEQ_LENGTH" =~ ^[0-9]{1,6}$ ]] || die 'Maximum sequence length must be an integer with at most six digits.'
MAX_SEQ_LENGTH=$((10#$MAX_SEQ_LENGTH))
(( MAX_SEQ_LENGTH >= 128 && MAX_SEQ_LENGTH <= 131072 )) || die 'Maximum sequence length must be between 128 and 131072.'

MODEL_TEMP=""
prompt_value MODEL_TEMP 'Enter Ollama temperature (0.0-2.0)' '0.7'
if ! "$VENV_PY" - "$MODEL_TEMP" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("Temperature must be a number between 0.0 and 2.0.")
if not math.isfinite(value) or not 0.0 <= value <= 2.0:
    raise SystemExit("Temperature must be a finite number between 0.0 and 2.0.")
PY
then
    die 'Invalid Ollama temperature.'
fi

MODEL_SYSTEM_PROMPT=""
read -r -p 'Enter the Ollama system prompt [You are a helpful assistant trained with AarForge.]: ' MODEL_SYSTEM_PROMPT || true
MODEL_SYSTEM_PROMPT="${MODEL_SYSTEM_PROMPT:-You are a helpful assistant trained with AarForge.}"
[[ "$MODEL_SYSTEM_PROMPT" != *'"""'* ]] || die 'System prompt cannot contain the triple-quote delimiter (""") used by Ollama Modelfile syntax.'

DATASET_COPY="${SCRIPT_DIR}/${BOT_NAME}_dataset.jsonl"
if [[ "$DATASET_SOURCE" != "$(realpath -m -- "$DATASET_COPY")" ]]; then
    if [[ -e "$DATASET_COPY" ]]; then
        overwrite_dataset=''
        read -r -p "$DATASET_COPY already exists. Replace it? [y/N]: " overwrite_dataset || true
        [[ "${overwrite_dataset,,}" == 'y' || "${overwrite_dataset,,}" == 'yes' ]] || die 'Dataset copy was not replaced. Choose another bot name or remove the existing copy.'
    fi
    cp -- "$DATASET_SOURCE" "$DATASET_COPY"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${SCRIPT_DIR}/runs/${BOT_NAME}-${RUN_ID}"
mkdir -p -- "$(dirname -- "$RUN_DIR")"
mkdir -- "$RUN_DIR"
ADAPTER_DIR="${RUN_DIR}/adapter"
GGUF_DIR="${RUN_DIR}/gguf"
TRAIN_LOG="${RUN_DIR}/training.log"
EXPORT_LOG="${RUN_DIR}/export.log"
DEPLOY_LOG="${RUN_DIR}/ollama-deploy.log"
BUILD_STATUS='INPUTS VALIDATED'

export AARFORGE_BOT_NAME="$BOT_NAME"
export AARFORGE_BASE_MODEL="$BASE_MODEL"
export AARFORGE_DATASET="$DATASET_COPY"
export AARFORGE_RUN_DIR="$RUN_DIR"
export AARFORGE_ADAPTER_DIR="$ADAPTER_DIR"
export AARFORGE_GGUF_DIR="$GGUF_DIR"
export AARFORGE_NUM_EPOCHS="$TRAIN_EPOCHS"
export AARFORGE_MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH"
export AARFORGE_MODEL_TEMP="$MODEL_TEMP"
export AARFORGE_SYSTEM_PROMPT="$MODEL_SYSTEM_PROMPT"

log "Run directory: $RUN_DIR"
log 'Generating training and export programs...'

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
                errors.append(f"line {line_number}: messages after system must alternate user, assistant and end with assistant")
                continue
            records.append(record)
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        extra = f"\n  ... and {len(errors) - 20} more error(s)" if len(errors) > 20 else ""
        fail(f"Dataset validation failed with {len(errors)} error(s):\n{preview}{extra}")
    if not records:
        fail("Dataset contains no non-empty JSONL records.")
    if len(records) == 1:
        print("[!] Dataset contains only one conversation. Training can run, but more diverse examples are strongly recommended.")
    return records


def validate_only() -> None:
    dataset_path = Path(os.environ["AARFORGE_DATASET"]).expanduser().resolve()
    records = load_and_validate_dataset(dataset_path)
    print(f"[+] Dataset validation passed: {len(records)} conversation(s), exact messages JSONL schema.")


def train() -> None:
    # Unsloth must be imported before Torch/Transformers/TRL.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model = os.environ["AARFORGE_BASE_MODEL"]
    dataset_path = Path(os.environ["AARFORGE_DATASET"]).expanduser().resolve()
    run_dir = Path(os.environ["AARFORGE_RUN_DIR"]).resolve()
    adapter_dir = Path(os.environ["AARFORGE_ADAPTER_DIR"]).resolve()
    num_epochs = int(os.environ["AARFORGE_NUM_EPOCHS"])
    max_seq_length = int(os.environ["AARFORGE_MAX_SEQ_LENGTH"])
    records = load_and_validate_dataset(dataset_path)

    if not torch.cuda.is_available():
        fail("CUDA is not available to PyTorch. Check the NVIDIA driver and CUDA-enabled PyTorch installation.")

    # A PEFT repository is not a plain base checkpoint. Detect local adapter repos
    # before loading, then also inspect the loaded object for an already-attached PEFT config.
    local_candidate = Path(base_model).expanduser()
    if local_candidate.is_dir() and (local_candidate / "adapter_config.json").is_file():
        fail(
            f"'{base_model}' is a PEFT/LoRA adapter repository (adapter_config.json was found), not a standalone base model. "
            "Select the original base checkpoint, or merge the existing adapter into its base model before using AarForge."
        )

    print(f"[+] Loading tokenizer for {base_model} to check sequence lengths...")
    try:
        tokenizer_check = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=False)
    except Exception as exc:
        fail(f"Could not load tokenizer for '{base_model}': {type(exc).__name__}: {exc}")
    if not getattr(tokenizer_check, "chat_template", None):
        fail("The selected model has no tokenizer chat_template. Choose an instruction/chat model that provides one.")
    too_long: list[tuple[int, int]] = []
    for idx, record in enumerate(records, start=1):
        try:
            token_ids = tokenizer_check.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=False)
            token_count = len(token_ids)
        except Exception as exc:
            fail(f"Could not render conversation {idx} with the model's chat template: {type(exc).__name__}: {exc}")
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
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as exc:
        fail(
            f"Unsloth could not load this model as a 4-bit fine-tuning base: {type(exc).__name__}: {exc}. "
            "The repository may use an unsupported architecture, require remote custom code, or be an adapter rather than a base checkpoint."
        )

    existing_peft_config = getattr(model, "peft_config", None)
    if existing_peft_config:
        fail(
            "The loaded model already has PEFT/LoRA adapter configuration. AarForge will not attach a second adapter with different settings. "
            "Use the original adapter-free base checkpoint, or merge the existing adapter first."
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
            "This build supports common decoder-style Transformer projections; choose a compatible architecture."
        )
    print(f"[+] LoRA target modules: {', '.join(target_modules)}")
    try:
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
    except TypeError as exc:
        if "already has LoRA adapters" in str(exc):
            fail(
                "Unsloth reports that this model already has LoRA adapters. It cannot accept AarForge's new adapter settings. "
                "Use an adapter-free base model or merge the repository's existing adapter first."
            )
        raise
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_dataset = Dataset.from_list(records)
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    training_args = SFTConfig(
        output_dir=str(run_dir / "trainer_output"),
        max_length=max_seq_length,
        num_train_epochs=num_epochs,
        max_steps=-1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
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
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, processing_class=tokenizer)

    # Assistant-only loss is represented by labels: prompt tokens are -100 and
    # assistant target tokens retain token IDs. Some TRL versions omit attention_mask.
    probe = next(iter(trainer.get_train_dataloader()))
    labels = probe.get("labels")
    if labels is None:
        fail("TRL did not produce training labels; refusing to start training.")
    labels = labels.detach()
    supervised = int((labels != -100).sum().item())
    masked = int((labels == -100).sum().item())
    if supervised <= 0 or masked <= 0:
        fail("Assistant-only loss verification failed: expected assistant target labels and masked prompt labels. Check the model chat template and TRL configuration.")
    print(f"[+] Assistant-only loss labels verified (assistant target tokens: {supervised}; masked tokens: {masked}).")

    print(f"[+] Training for {num_epochs} epoch(s)...")
    result = trainer.train()
    if trainer.state.global_step < 1:
        fail("Trainer completed without performing an optimizer step.")

    adapter_dir.mkdir(parents=True, exist_ok=False)
    print(f"[+] Saving LoRA adapter to {adapter_dir}...")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if not (adapter_dir / "adapter_config.json").is_file() or not list(adapter_dir.glob("adapter_model.*")):
        fail("Adapter verification failed: adapter_config.json or adapter model weights are missing.")

    dataset_hash = hashlib.sha256()
    with dataset_path.open("rb") as dataset_stream:
        for chunk in iter(lambda: dataset_stream.read(1024 * 1024), b""):
            dataset_hash.update(chunk)
    metadata = {
        "bot_name": os.environ["AARFORGE_BOT_NAME"],
        "base_model": base_model,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash.hexdigest(),
        "conversation_count": len(records),
        "num_epochs_requested": num_epochs,
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
"""Export adapter to GGUF; distinguish main model weights from mmproj sidecars and shards."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$", re.IGNORECASE)


def fail(message: str) -> None:
    raise SystemExit(f"[-] {message}")


def validate_gguf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        fail(f"GGUF output is missing or unexpectedly small: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            fail(f"Output does not have a valid GGUF header: {path}")


def group_main_files(paths: list[Path]) -> list[dict]:
    groups: dict[tuple, list[Path]] = {}
    for path in paths:
        match = SHARD_RE.match(path.name)
        if match:
            key = (str(path.parent), match.group("stem"), int(match.group("total")))
        else:
            key = (str(path.parent), path.name, 1)
        groups.setdefault(key, []).append(path)

    result: list[dict] = []
    for (parent, stem_or_name, total), files in groups.items():
        shard_matches = [SHARD_RE.match(path.name) for path in files]
        if any(shard_matches):
            if not all(shard_matches):
                fail(f"Mixed shard and non-shard GGUF files in candidate group: {files}")
            indices = sorted(int(match.group("index")) for match in shard_matches if match)
            if indices != list(range(1, total + 1)) or len(files) != total:
                fail(f"Incomplete GGUF shard group '{stem_or_name}': found {indices}, expected 1..{total}.")
            stem = stem_or_name
        else:
            stem = Path(stem_or_name).stem
        names = " ".join(path.name.lower() for path in files)
        normalized = names.replace("-", "_")
        score = 0
        if "q4_k_m" in normalized:
            score += 100
        elif "q4_k_s" in normalized:
            score += 90
        elif "q4_0" in normalized:
            score += 80
        elif "q5_k_m" in normalized:
            score += 75
        elif "q8_0" in normalized:
            score += 70
        elif "q6_k" in normalized:
            score += 65
        if "bf16" in normalized:
            score -= 25
        if "f16" in normalized:
            score -= 20
        result.append({"parent": parent, "stem": stem, "total": total, "files": sorted(files), "score": score})
    return result


def main() -> None:
    from unsloth import FastLanguageModel

    adapter_dir = Path(os.environ["AARFORGE_ADAPTER_DIR"]).resolve()
    merged_dir = Path(os.environ["AARFORGE_GGUF_DIR"]).resolve()
    max_seq_length = int(os.environ["AARFORGE_MAX_SEQ_LENGTH"])
    if not (adapter_dir / "adapter_config.json").is_file():
        fail(f"Adapter configuration not found: {adapter_dir / 'adapter_config.json'}")
    merged_dir.mkdir(parents=True, exist_ok=False)
    sibling_export_dir = Path(f"{merged_dir}_gguf")

    print(f"[+] Reloading adapter from {adapter_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("[+] Exporting merged model to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(str(merged_dir), tokenizer, quantization_method="q4_k_m")

    # Unsloth commonly writes GGUF files to <save_directory>_gguf. Older builds
    # may write directly to save_directory. Prefer the sibling export directory.
    chosen_dir = None
    discovered: list[Path] = []
    for candidate_dir in (sibling_export_dir, merged_dir):
        if candidate_dir.is_dir():
            candidate_files = sorted(path for path in candidate_dir.rglob("*.gguf") if path.is_file())
            if candidate_files and any("mmproj" not in path.name.lower() for path in candidate_files):
                chosen_dir = candidate_dir.resolve()
                discovered = candidate_files
                break
    if chosen_dir is None:
        fail(f"Export command finished but no main GGUF model file was found in {sibling_export_dir} or {merged_dir}.")

    mmproj_files = sorted(path for path in discovered if "mmproj" in path.name.lower())
    main_candidates = sorted(path for path in discovered if "mmproj" not in path.name.lower())
    if not main_candidates:
        fail("Only mmproj GGUF projector files were produced; no main model GGUF was found.")

    groups = group_main_files(main_candidates)
    highest_score = max(group["score"] for group in groups)
    preferred = [group for group in groups if group["score"] == highest_score]
    if len(preferred) != 1:
        details = [([str(path) for path in group["files"]], group["score"]) for group in preferred]
        fail(f"Multiple equally suitable main GGUF exports were produced; cannot choose safely: {details}")
    selected = preferred[0]
    main_files: list[Path] = selected["files"]

    for path in main_files + mmproj_files:
        validate_gguf(path)

    other_main_files = [path for path in main_candidates if path not in main_files]
    manifest = {
        "output_directory": str(chosen_dir),
        "main_model_files": [str(path) for path in main_files],
        "main_model_file_count": len(main_files),
        "main_model_sharded": len(main_files) > 1,
        "main_model_shard_stem": selected["stem"] if len(main_files) > 1 else None,
        "main_model_score": selected["score"],
        "requested_quantization": "q4_k_m",
        "mmproj_files": [str(path) for path in mmproj_files],
        "is_multimodal_export": bool(mmproj_files),
        "other_main_gguf_files": [str(path) for path in other_main_files],
        "size_bytes_main": sum(path.stat().st_size for path in main_files),
        "size_bytes_mmproj": sum(path.stat().st_size for path in mmproj_files),
        "size_bytes_other": sum(path.stat().st_size for path in other_main_files),
    }
    (chosen_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("[+] GGUF export verification passed.")
    print(f"[+] Selected main model file(s): {len(main_files)}")
    for path in main_files:
        print(f"    MAIN: {path} ({path.stat().st_size:,} bytes)")
    if mmproj_files:
        print(f"[+] Vision projector file(s): {len(mmproj_files)} (preserved separately)")
        for path in mmproj_files:
            print(f"    MMPROJ: {path} ({path.stat().st_size:,} bytes)")
    if other_main_files:
        print(f"[i] Additional non-selected GGUF file(s) were retained: {len(other_main_files)}")
        for path in other_main_files:
            print(f"    OTHER: {path}")
    print(f"[+] Export manifest: {chosen_dir / 'export_manifest.json'}")
    # Machine-readable markers consumed by Bash after the log has been written.
    print(f"AARFORGE_GGUF_OUTPUT_DIR={chosen_dir}")
    print(f"AARFORGE_GGUF_HAS_MMPROJ={'yes' if mmproj_files else 'no'}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        print(f"[-] GGUF export exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(2)
PY

chmod 600 "${RUN_DIR}/train_bot.py" "${RUN_DIR}/export_gguf.py"

log 'Validating the JSONL dataset before loading any model weights...'
if ! run_streamed "${RUN_DIR}/dataset-validation.log" "$VENV_PY" "${RUN_DIR}/train_bot.py" --validate-only; then
    die 'Dataset validation failed. Review dataset-validation.log.'
fi
log 'Compiling generated Python programs...'
if ! "$VENV_PY" -m py_compile "${RUN_DIR}/train_bot.py" "${RUN_DIR}/export_gguf.py"; then
    die 'Generated Python program compilation failed.'
fi

printf '\n==================================================\n'
printf '%s\n' '             STAGE 1: MODEL TRAINING              '
printf '%s\n' '=================================================='
BUILD_STATUS='TRAINING'
if ! run_streamed "$TRAIN_LOG" "$VENV_PY" "${RUN_DIR}/train_bot.py"; then
    die 'Training failed. Review training.log; any checkpoint/adapter already written has been preserved.'
fi
[[ -s "${ADAPTER_DIR}/adapter_config.json" ]] || die 'Training process ended without a verified adapter_config.json.'
BUILD_STATUS='TRAINING COMPLETE'

printf '\n==================================================\n'
printf '%s\n' '      STAGE 2: MERGE ADAPTER & GGUF EXPORT       '
printf '%s\n' '=================================================='
BUILD_STATUS='GGUF EXPORT'
if ! run_streamed "$EXPORT_LOG" "$VENV_PY" "${RUN_DIR}/export_gguf.py"; then
    print_partial_success "Training completed and the LoRA adapter was preserved, but GGUF export failed. Review export.log. You can retry export without retraining by using the saved adapter at ${ADAPTER_DIR}, once the export issue is corrected."
fi

# Read the final export directory/vision marker from the completed exporter log.
GGUF_OUTPUT_DIR="$(sed -n 's/^AARFORGE_GGUF_OUTPUT_DIR=//p' "$EXPORT_LOG" | tail -n 1)"
GGUF_HAS_MMPROJ="$(sed -n 's/^AARFORGE_GGUF_HAS_MMPROJ=//p' "$EXPORT_LOG" | tail -n 1)"
[[ -n "$GGUF_OUTPUT_DIR" && -d "$GGUF_OUTPUT_DIR" ]] || print_partial_success 'GGUF export ran but did not report a usable output directory. The adapter and export log are preserved.'
export AARFORGE_GGUF_DIR="$GGUF_OUTPUT_DIR"
GGUF_DIR="$GGUF_OUTPUT_DIR"

# Read the main-file selection from the manifest; never pick an mmproj by glob order.
GGUF_MAIN_FILE="$("$VENV_PY" - "$GGUF_OUTPUT_DIR/export_manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
files = manifest.get("main_model_files", [])
if not files:
    raise SystemExit("Manifest contains no selected main model GGUF.")
print(files[0])
PY
)" || print_partial_success 'The GGUF export manifest is missing or invalid. Training/export files have been preserved.'
MMPROJ_FILES_TEXT="$("$VENV_PY" - "$GGUF_OUTPUT_DIR/export_manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(";".join(manifest.get("mmproj_files", [])))
PY
)" || print_partial_success 'Could not read the GGUF manifest. The exported files remain on disk.'

# Create an Ollama Modelfile from the exact main model selection. For split GGUF,
# stage only the selected shards in a separate folder so the wildcard cannot
# accidentally include a same-prefix mmproj or unused BF16 model.
cat > "${RUN_DIR}/write_modelfile.py" <<'PY'
import json
import os
import re
from pathlib import Path

run_dir = Path(os.environ["AARFORGE_RUN_DIR"]).resolve()
manifest_path = Path(os.environ["AARFORGE_GGUF_DIR"]) / "export_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
main_files = [Path(item).resolve() for item in manifest.get("main_model_files", [])]
if not main_files or any(not path.is_file() for path in main_files):
    raise SystemExit("The selected main GGUF file(s) are missing.")

if len(main_files) == 1:
    from_path = main_files[0]
else:
    # Use the exact shard filenames in a private import directory; this keeps
    # wildcard matching away from mmproj files and other GGUF variants.
    matches = [re.match(r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$", p.name, re.I) for p in main_files]
    if any(match is None for match in matches):
        raise SystemExit("Multiple main GGUF files were selected but are not a recognized shard set.")
    stems = {match.group("stem") for match in matches if match}
    totals = {int(match.group("total")) for match in matches if match}
    indices = sorted(int(match.group("index")) for match in matches if match)
    if len(stems) != 1 or len(totals) != 1 or indices != list(range(1, next(iter(totals)) + 1)):
        raise SystemExit("Main GGUF shards are not one complete, consistently named shard set.")
    import_dir = run_dir / "ollama_import_shards"
    import_dir.mkdir(exist_ok=False)
    for source in main_files:
        (import_dir / source.name).symlink_to(source)
    from_path = import_dir / f"{next(iter(stems))}-*.gguf"

prompt = os.environ["AARFORGE_SYSTEM_PROMPT"]
if '"""' in prompt or "\x00" in prompt:
    raise SystemExit('System prompt cannot contain the triple-quote delimiter or NUL bytes.')
temperature = os.environ["AARFORGE_MODEL_TEMP"]
# Paths emitted by AarForge are absolute. Quote a path with whitespace so it
# remains a single Modelfile FROM argument.
from_value = str(from_path)
if any(ch.isspace() for ch in from_value):
    from_value = '"' + from_value.replace('\\', '\\\\').replace('"', '\\"') + '"'
content = f'FROM {from_value}\n\nSYSTEM """{prompt}"""\n\nPARAMETER temperature {temperature}\n'
modelfile = run_dir / "Modelfile"
modelfile.write_text(content, encoding="utf-8")

# Verify the FROM target or selected shard set still exists.
if len(main_files) == 1 and not from_path.is_file():
    raise SystemExit(f"Modelfile FROM target does not exist: {from_path}")
if len(main_files) > 1 and not list(from_path.parent.glob(from_path.name)):
    raise SystemExit(f"Modelfile shard pattern matched no files: {from_path}")
print(f"[+] Wrote Ollama Modelfile: {modelfile}")
print(f"[+] Modelfile FROM: {from_value}")
PY

if ! "$VENV_PY" "${RUN_DIR}/write_modelfile.py"; then
    print_partial_success 'Training and GGUF export completed, but AarForge could not produce a valid Ollama Modelfile. All artifacts are preserved.'
fi

printf '\n==================================================\n'
printf '%s\n' '           STAGE 3: OLLAMA MODEL                  '
printf '%s\n' '=================================================='
BUILD_STATUS='OLLAMA DEPLOYMENT'

if (( ! OLLAMA_READY )); then
    DEPLOYMENT_STATUS='skipped: Ollama is missing or its daemon is unreachable'
    print_partial_success "Training and GGUF export succeeded. ${DEPLOYMENT_STATUS}. Install/start Ollama and deploy manually with: ollama create ${BOT_NAME} -f '${RUN_DIR}/Modelfile'."
fi

if [[ "$GGUF_HAS_MMPROJ" == 'yes' ]]; then
    printf '\n[!] MULTIMODAL MODEL NOTICE\n'
    printf '%s\n' 'This export contains a separate mmproj GGUF vision projector.'
    printf '%s\n' 'The Modelfile generated here references only the main GGUF. A normal Ollama GGUF import does not provide a separate Modelfile directive for attaching this projector.'
    printf '%s\n' 'The model may therefore import as text-only, or Ollama may reject the architecture. The mmproj file is preserved for llama.cpp multimodal inference.'
    printf '%s\n' 'For llama.cpp, use the main GGUF together with --mmproj and llama-mtmd-cli.'
    import_text_only=''
    read -r -p 'Attempt an Ollama import of the main GGUF as a TEXT-ONLY model anyway? [y/N]: ' import_text_only || true
    if [[ "${import_text_only,,}" != 'y' && "${import_text_only,,}" != 'yes' ]]; then
        DEPLOYMENT_STATUS='skipped by user: multimodal mmproj cannot be attached through this Modelfile workflow'
        print_partial_success "Training and GGUF export succeeded. Ollama deployment was skipped by choice because a separate vision projector was detected. Main GGUF and mmproj files are both preserved."
    fi
fi

# Do not replace an existing model until a staged Ollama model has been created
# and verified. This reduces the chance of losing a working model on import error.
STAGING_MODEL="aarforge-staging-${RUN_ID,,}"
if ollama show "$STAGING_MODEL" >/dev/null 2>&1; then
    DEPLOYMENT_STATUS="skipped: staging name '$STAGING_MODEL' already exists"
    print_partial_success "A staging model with the generated temporary name already exists. No existing model was changed. Rerun the build or remove '$STAGING_MODEL' manually."
fi

log "Creating temporary Ollama model '$STAGING_MODEL' for verification..."
if ! run_streamed "$DEPLOY_LOG" ollama create "$STAGING_MODEL" -f "${RUN_DIR}/Modelfile"; then
    DEPLOYMENT_STATUS='failed: Ollama rejected the staged model import'
    DEPLOY_ERROR="$(tail -n 35 "$DEPLOY_LOG" 2>/dev/null || true)"
    print_partial_success "Training and GGUF export succeeded, but Ollama could not import the model. This usually indicates an unsupported model architecture, GGUF metadata incompatibility, or a missing multimodal projector integration. Ollama output follows:\n${DEPLOY_ERROR}"
fi
if ! ollama show "$STAGING_MODEL" >/dev/null 2>&1; then
    DEPLOYMENT_STATUS='failed: staged model could not be verified with ollama show'
    print_partial_success "Ollama create returned, but the staging model could not be verified. The original Ollama model (if any) was not intentionally removed. Review ollama-deploy.log."
fi

# Detect a model created by another process during training.
if ollama show "$BOT_NAME" >/dev/null 2>&1 && (( ! REPLACE_MODEL )); then
    DEPLOYMENT_STATUS="staged only: '$BOT_NAME' appeared during the build; it was not overwritten"
    print_partial_success "The staged model '$STAGING_MODEL' was created and verified, but '$BOT_NAME' appeared while training was running. AarForge preserved that existing model. Review the staging model and deploy it manually when ready."
fi

BACKUP_MODEL="aarforge-backup-${RUN_ID,,}"
HAD_OLD_MODEL=0
if ollama show "$BOT_NAME" >/dev/null 2>&1; then
    if (( ! REPLACE_MODEL )); then
        DEPLOYMENT_STATUS="staged only: existing '$BOT_NAME' was preserved"
        print_partial_success "The staging model '$STAGING_MODEL' is ready, but the existing model '$BOT_NAME' was not approved for replacement."
    fi
    log "Backing up existing Ollama model '$BOT_NAME' as '$BACKUP_MODEL'..."
    if ! ollama cp "$BOT_NAME" "$BACKUP_MODEL" >>"$DEPLOY_LOG" 2>&1 || ! ollama show "$BACKUP_MODEL" >/dev/null 2>&1; then
        DEPLOYMENT_STATUS='staged only: existing model could not be backed up safely'
        print_partial_success "AarForge created '$STAGING_MODEL', but could not verify a backup of existing '$BOT_NAME'. The existing model was left untouched."
    fi
    HAD_OLD_MODEL=1
    # Ollama copy behavior for an existing destination can vary by version.
    # Remove the old tag only after a verified backup exists; then promotion is
    # unambiguous and a failed copy can be restored from that backup.
    log "Removing old model '$BOT_NAME' after verifying its backup..."
    if ! ollama rm "$BOT_NAME" >>"$DEPLOY_LOG" 2>&1; then
        DEPLOYMENT_STATUS='staged only: could not safely remove existing model after backup'
        print_partial_success "A backup '$BACKUP_MODEL' was made, but Ollama would not remove the existing '$BOT_NAME'. The original model was not deliberately overwritten; review ollama-deploy.log."
    fi
fi

log "Promoting verified staging model '$STAGING_MODEL' to '$BOT_NAME'..."
if ! ollama cp "$STAGING_MODEL" "$BOT_NAME" >>"$DEPLOY_LOG" 2>&1 || ! ollama show "$BOT_NAME" >/dev/null 2>&1; then
    if (( HAD_OLD_MODEL )); then
        warn 'Promotion failed. Attempting to restore the previous Ollama model from its backup...'
        if ollama cp "$BACKUP_MODEL" "$BOT_NAME" >>"$DEPLOY_LOG" 2>&1 && ollama show "$BOT_NAME" >/dev/null 2>&1; then
            info "Previous Ollama model '$BOT_NAME' was restored. Backup copy '$BACKUP_MODEL' is retained until you remove it."
        else
            warn "Automatic restoration failed. The backup model '$BACKUP_MODEL' is preserved; do not delete it."
        fi
    fi
    DEPLOYMENT_STATUS='failed: staged model was not promoted to the final name'
    print_partial_success "The staging model was created, but promotion to '$BOT_NAME' failed. Existing model backup status is recorded in ollama-deploy.log."
fi

if (( HAD_OLD_MODEL )); then
    if ollama rm "$BACKUP_MODEL" >>"$DEPLOY_LOG" 2>&1; then
        info "Removed temporary backup '$BACKUP_MODEL' after successful promotion."
    else
        warn "The new model is active, but temporary backup '$BACKUP_MODEL' could not be removed. Remove it manually after verification."
    fi
fi
if ollama rm "$STAGING_MODEL" >>"$DEPLOY_LOG" 2>&1; then
    info "Removed temporary staging model '$STAGING_MODEL'."
else
    warn "Model deployment succeeded, but temporary staging model '$STAGING_MODEL' remains. Remove it manually after verification."
fi

DEPLOYMENT_STATUS='success: final Ollama model was created and verified'
BUILD_STATUS='SUCCESS'
write_report "$BUILD_STATUS" 'Training, adapter verification, GGUF export, and Ollama model creation/metadata verification completed successfully. Inference generation was not smoke-tested; run the model once before relying on it. Review export_manifest.json for additional GGUF files and vision projector sidecars.'
printf '\n==================================================\n'
printf '%s\n' '          AARFORGE BUILD COMPLETE                 '
printf '%s\n' '=================================================='
printf 'Model:          ollama run %s\n' "$BOT_NAME"
printf '%s\n' 'Ollama metadata was verified; inference generation was not smoke-tested.'
printf 'Run artifacts:  %s\n' "$RUN_DIR"
printf 'Dataset copy:   %s\n' "$DATASET_COPY"
printf 'Main GGUF:      %s\n' "$GGUF_MAIN_FILE"
if [[ -n "$MMPROJ_FILES_TEXT" ]]; then
    printf 'Vision files:   %s\n' "$MMPROJ_FILES_TEXT"
    printf '%s\n' '[!] Ollama import was attempted without a separate mmproj directive; vision input may not be available through Ollama.'
fi
printf 'Build report:   %s/BUILD_REPORT.txt\n' "$RUN_DIR"
printf '%s\n' '=================================================='
