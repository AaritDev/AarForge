#!/usr/bin/env bash
# AarForge Linux installer. Uses Unsloth's published dependency constraints.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
# Pin the Unsloth release, not individual transitive dependencies. Its package
# metadata constrains the compatible torch/transformers/trl version ranges.
UNSLOTH_VERSION="2026.11.4"

log()  { printf '[+] %s\n' "$*"; }
info() { printf '[i] %s\n' "$*"; }
die()  { printf '[-] %s\n' "$*" >&2; exit 1; }
on_error() {
    local rc=$?
    printf '\n[-] Installation failed (exit %s) near line %s.\n' "$rc" "${BASH_LINENO[0]:-unknown}" >&2
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

printf '%s\n' '=================================================='
printf '%s\n' '               AARFORGE INSTALLER                 '
printf '%s\n' '=================================================='
printf '%s\n' 'Dataset format: one JSON object per line, with a top-level messages array.'
printf '%s\n' '{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
printf '%s\n' 'Roles are system, user, and assistant; content must be strings.'
printf '%s\n' '=================================================='

[[ "$(uname -s)" == Linux ]] || die 'This installer currently supports Linux only.'
for tool in curl realpath; do command -v "$tool" >/dev/null 2>&1 || die "$tool is required."; done
command -v python3 >/dev/null 2>&1 || die 'Python 3 is required.'
PYTHON_BIN="$(command -v python3)"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
    raise SystemExit(f"AarForge currently supports Python 3.10–3.13; found {sys.version.split()[0]}.")
print(f"[+] Using Python {sys.version.split()[0]}")
PY

[[ -f "$SCRIPT_DIR/build.sh" ]] || die "build.sh must be present beside install.sh ($SCRIPT_DIR)."
INSTALL_DIR=""
read -r -p 'Enter the directory for the AarForge workspace: ' INSTALL_DIR || true
[[ -n "$INSTALL_DIR" ]] || die 'Workspace path cannot be empty.'
INSTALL_DIR="$(expand_home "$INSTALL_DIR")"
mkdir -p -- "$INSTALL_DIR"
INSTALL_DIR="$(cd -- "$INSTALL_DIR" && pwd -P)"

if ! command -v uv >/dev/null 2>&1; then
    log 'Installing Astral UV...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
    UV_BIN="$HOME/.cargo/bin/uv"
else
    die 'UV installation completed but its executable could not be located.'
fi

if ! command -v ollama >/dev/null 2>&1; then
    log 'Installing Ollama...'
    curl -fsSL https://ollama.com/install.sh | sh
fi
command -v ollama >/dev/null 2>&1 || die "Ollama installation did not provide an 'ollama' executable."
ollama --version

cd -- "$INSTALL_DIR"
if [[ ! -d .venv ]]; then
    log 'Creating workspace-local Python virtual environment...'
    "$UV_BIN" venv --python "$PYTHON_BIN" .venv
else
    info "Reusing existing $INSTALL_DIR/.venv. It will not be deleted."
fi
VENV_PY="$INSTALL_DIR/.venv/bin/python"
[[ -x "$VENV_PY" ]] || die 'Invalid .venv: .venv/bin/python is missing.'

log "Installing Unsloth $UNSLOTH_VERSION and its declared compatible dependencies..."
# Do not add an independent transformers>=5.18 requirement: it contradicts
# the tested upper bounds in the selected Unsloth release metadata.
"$UV_BIN" pip install --python "$VENV_PY" --upgrade --reinstall \
    "unsloth==$UNSLOTH_VERSION" gguf

log 'Checking dependency consistency and runtime imports...'
"$UV_BIN" pip check --python "$VENV_PY"
"$VENV_PY" - <<'PY'
import sys
# Unsloth must patch/initialize before importing Transformers, TRL or Torch.
import unsloth
import torch
import transformers
import trl
import datasets
import peft
import bitsandbytes
import huggingface_hub
from packaging.version import Version

checks = {
    "torch": (Version(torch.__version__.split('+')[0]), Version("2.15.0"), "<"),
    "transformers": (Version(transformers.__version__), Version("5.17.0"), "<="),
    "trl": (Version(trl.__version__), Version("1.15.0"), "<="),
}
for name, (actual, ceiling, op) in checks.items():
    ok = actual < ceiling if op == "<" else actual <= ceiling
    if not ok:
        raise SystemExit(f"Incompatible {name} {actual}; selected Unsloth requires {name} {op} {ceiling}.")
if not hasattr(trl, "SFTTrainer") or not hasattr(trl, "SFTConfig"):
    raise SystemExit("Installed TRL does not expose SFTTrainer and SFTConfig.")
print(f"Python:       {sys.version.split()[0]}")
print(f"Unsloth:      {getattr(unsloth, '__version__', 'installed')}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"TRL:          {trl.__version__}")
print(f"Datasets:     {datasets.__version__}")
print(f"CUDA built:   {torch.version.cuda}")
print(f"CUDA usable:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
else:
    print("[!] No usable CUDA GPU detected. Installation is valid, but this QLoRA workflow cannot train until CUDA is available.")
PY

missing_tools=()
for tool in git cmake make c++; do command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool"); done
if ((${#missing_tools[@]})); then
    printf '[!] Missing GGUF export build tools: %s\n' "${missing_tools[*]}" >&2
    printf '%s\n' '    Install git, cmake, make, and a C++ compiler using your distribution package manager.' >&2
fi

SOURCE_BUILD="$(realpath -e -- "$SCRIPT_DIR/build.sh")"
TARGET_BUILD="$INSTALL_DIR/build.sh"
if [[ "$SOURCE_BUILD" != "$TARGET_BUILD" ]]; then
    if [[ -e "$TARGET_BUILD" ]] && ! cmp -s -- "$SOURCE_BUILD" "$TARGET_BUILD"; then
        answer=''
        read -r -p "$TARGET_BUILD exists and differs. Replace it? [y/N]: " answer || true
        [[ "${answer,,}" == y || "${answer,,}" == yes ]] || die 'Existing build.sh was not replaced. No installation completion will be reported.'
    fi
    cp -- "$SOURCE_BUILD" "$TARGET_BUILD"
fi
chmod +x "$TARGET_BUILD"

SAMPLE_DATASET="$INSTALL_DIR/sample_dataset.jsonl"
if [[ ! -e "$SAMPLE_DATASET" ]]; then
    cat > "$SAMPLE_DATASET" <<'JSONL'
{"messages": [{"role": "system", "content": "You are an AI assistant trained with AarForge."}, {"role": "user", "content": "What dataset format does AarForge use?"}, {"role": "assistant", "content": "AarForge uses one JSON object per line. Each object has a messages array containing ordered system, user, and assistant messages."}]}
JSONL
else
    info 'Preserving existing sample_dataset.jsonl.'
fi
"$VENV_PY" - "$SAMPLE_DATASET" <<'PY'
import json, sys
path = sys.argv[1]
count = 0
with open(path, encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or set(obj) != {"messages"} or not isinstance(obj["messages"], list) or not obj["messages"]:
            raise SystemExit(f"Invalid JSONL record on line {n}.")
        for i, msg in enumerate(obj["messages"], 1):
            if not isinstance(msg, dict) or set(msg) != {"role", "content"} or msg["role"] not in {"system", "user", "assistant"} or not isinstance(msg["content"], str):
                raise SystemExit(f"Invalid message on line {n}, message {i}.")
        count += 1
if not count:
    raise SystemExit("Sample dataset is empty.")
print(f"[+] Sample JSONL validated: {count} conversation(s).")
PY

read -r -p 'Download/cache the default model now? This may require several GB. [Y/n]: ' cache_model || true
if [[ -z "${cache_model:-}" || "${cache_model,,}" == y || "${cache_model,,}" == yes ]]; then
    HF_BIN="$INSTALL_DIR/.venv/bin/hf"
    if [[ -x "$HF_BIN" ]]; then
        log "Downloading $DEFAULT_MODEL..."
        if "$HF_BIN" download "$DEFAULT_MODEL"; then
            log 'Default model cache completed.'
        else
            printf '[!] Model pre-download failed; training can retry the download later.\n' >&2
        fi
    else
        printf '[!] Hugging Face CLI not found; the model will download during training.\n' >&2
    fi
else
    info 'Skipping default model download.'
fi

printf '\n==================================================\n'
printf '%s\n' '             INSTALLATION COMPLETE                '
printf 'Workspace:       %s\n' "$INSTALL_DIR"
printf 'Virtual env:     %s\n' "$INSTALL_DIR/.venv"
printf 'Sample dataset:  %s\n' "$SAMPLE_DATASET"
printf '%s\n' 'Next: cd into the workspace and run ./build.sh'
printf '%s\n' '=================================================='
