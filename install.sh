#!/usr/bin/env bash
# AarForge installer for Linux. Installs tools into a workspace-local .venv.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

log()  { printf '[+] %s\n' "$*"; }
info() { printf '[i] %s\n' "$*"; }
die()  { printf '[-] %s\n' "$*" >&2; exit 1; }
on_error() {
    local rc=$?
    printf '\n[-] Installation failed (exit %s) at line %s.\n' "$rc" "${BASH_LINENO[0]:-unknown}" >&2
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

[[ "$(uname -s)" == "Linux" ]] || die "This installer currently supports Linux only."
command -v curl >/dev/null 2>&1 || die "curl is required. Install curl with your distribution package manager and rerun."
command -v python3 >/dev/null 2>&1 || die "Python 3 is required. Install Python 3.10 or newer and rerun."
command -v realpath >/dev/null 2>&1 || die "realpath is required (usually provided by coreutils)."

PYTHON_BIN="$(command -v python3)"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10 or newer is required; found {sys.version.split()[0]}.")
print(f"[+] Using Python {sys.version.split()[0]}")
PY

INSTALL_DIR=""
read -r -p "Enter the directory for the AarForge workspace: " INSTALL_DIR || true
[[ -n "$INSTALL_DIR" ]] || die "Workspace path cannot be empty."
INSTALL_DIR="$(expand_home "$INSTALL_DIR")"
mkdir -p -- "$INSTALL_DIR"
INSTALL_DIR="$(cd -- "$INSTALL_DIR" && pwd -P)"

[[ -f "${SCRIPT_DIR}/build.sh" ]] || die "build.sh is missing beside install.sh in $SCRIPT_DIR. Keep both scripts in the same directory."

# Install UV only when it is missing. Resolve its path explicitly because the installer
# may add it to a shell startup file that this non-interactive process does not reload.
if ! command -v uv >/dev/null 2>&1; then
    log "Installing Astral UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
    UV_BIN="$HOME/.cargo/bin/uv"
else
    die "UV installation finished but the uv executable could not be located. Add ~/.local/bin or ~/.cargo/bin to PATH and rerun."
fi

# Ollama install script handles the supported Linux distributions and service setup.
if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
command -v ollama >/dev/null 2>&1 || die "Ollama installation did not provide an 'ollama' executable."
ollama --version

cd -- "$INSTALL_DIR"
if [[ ! -d .venv ]]; then
    log "Creating workspace-local Python virtual environment..."
    "$UV_BIN" venv --python "$PYTHON_BIN" .venv
else
    info "Reusing existing $INSTALL_DIR/.venv; it will not be deleted or recreated."
fi
[[ -x "${INSTALL_DIR}/.venv/bin/python" ]] || die "The workspace virtual environment is invalid: .venv/bin/python is missing."
VENV_PY="${INSTALL_DIR}/.venv/bin/python"

log "Installing compatible Unsloth/TRL dependencies through UV's resolver..."
# Avoid the previous brittle pin list of dozens of transitive dependencies. Let the
# package resolver select a compatible set for the current Python/platform.
"$UV_BIN" pip install --python "$VENV_PY" --upgrade \
    unsloth \
    trl \
    "transformers>=5.18.0" \
    peft \
    datasets \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    gguf \
    packaging

log "Checking the installed Python packages and runtime..."
"$VENV_PY" - <<'PY'
import sys
import torch
import transformers
import datasets
import trl
import unsloth
import huggingface_hub
import bitsandbytes
from packaging.version import Version
if Version(transformers.__version__) < Version("5.18.0"):
    raise SystemExit(f"Transformers >=5.18.0 is required for the assistant-only loss pipeline; found {transformers.__version__}. Rerun install.sh to resolve dependencies.")
if not hasattr(trl, "SFTTrainer") or not hasattr(trl, "SFTConfig"):
    raise SystemExit("The installed TRL package does not expose SFTTrainer and SFTConfig. Rerun install.sh with a compatible TRL release.")
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets:     {datasets.__version__}")
print(f"TRL:          {trl.__version__}")
print(f"CUDA built:   {torch.version.cuda}")
print(f"CUDA usable:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No usable CUDA GPU detected. The installer completed, but the current AarForge QLoRA build workflow cannot train until an NVIDIA GPU, driver, and CUDA-enabled PyTorch are available.")
PY

# GGUF conversion can need a local C/C++ toolchain (llama.cpp build).
missing_tools=()
for tool in git cmake make; do
    command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
done
if command -v c++ >/dev/null 2>&1; then :; else missing_tools+=("C++ compiler (c++)"); fi
if ((${#missing_tools[@]})); then
    printf '[!] Missing optional GGUF export build tools: %s\n' "${missing_tools[*]}" >&2
    printf '%s\n' '    Install git, cmake, make, and a C++ compiler using your Linux distribution package manager.' >&2
fi

# Copy the matching build script without clobbering a different existing workspace copy.
SOURCE_BUILD="$(realpath -- "${SCRIPT_DIR}/build.sh")"
TARGET_BUILD="${INSTALL_DIR}/build.sh"
if [[ "$SOURCE_BUILD" != "$TARGET_BUILD" ]]; then
    if [[ -e "$TARGET_BUILD" ]] && ! cmp -s -- "$SOURCE_BUILD" "$TARGET_BUILD"; then
        replace_build=""
        read -r -p "$TARGET_BUILD already exists and differs. Replace it? [y/N]: " replace_build || true
        [[ "${replace_build,,}" == "y" || "${replace_build,,}" == "yes" ]] || die "Existing build.sh was not replaced. Copy the new build.sh into $INSTALL_DIR manually, then rerun install.sh."
    fi
    cp -- "$SOURCE_BUILD" "$TARGET_BUILD"
fi
chmod +x "$TARGET_BUILD"

SAMPLE_DATASET="${INSTALL_DIR}/sample_dataset.jsonl"
if [[ ! -e "$SAMPLE_DATASET" ]]; then
    cat > "$SAMPLE_DATASET" <<'JSONL'
{"messages": [{"role": "system", "content": "You are an AI assistant trained with AarForge."}, {"role": "user", "content": "What dataset format does AarForge use?"}, {"role": "assistant", "content": "AarForge uses one JSON object per line. Each object has a messages array containing ordered system, user, and assistant messages."}]}
JSONL
else
    info "Preserving existing sample_dataset.jsonl; it was not overwritten."
fi

# Run a stdlib-only sanity check on the sample JSONL before declaring installation complete.
"$VENV_PY" - "$SAMPLE_DATASET" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    lines = [line for line in stream if line.strip()]
if not lines:
    raise SystemExit("Sample JSONL file is empty.")
for number, line in enumerate(lines, 1):
    obj = json.loads(line)
    if set(obj) != {"messages"} or not isinstance(obj["messages"], list):
        raise SystemExit(f"Invalid sample JSONL schema on line {number}.")
    for message in obj["messages"]:
        if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise SystemExit(f"Invalid sample message on line {number}.")
print(f"[+] Sample dataset validated: {path}")
PY

printf '\n'
read -r -p "Download/cache the default model now? This can require several GB. [Y/n]: " cache_model || true
if [[ -z "${cache_model:-}" || "${cache_model,,}" == "y" || "${cache_model,,}" == "yes" ]]; then
    HF_BIN="${INSTALL_DIR}/.venv/bin/hf"
    if [[ -x "$HF_BIN" ]]; then
        log "Downloading model files for $DEFAULT_MODEL into the Hugging Face cache..."
        if "$HF_BIN" download "$DEFAULT_MODEL" --include '*.safetensors' 'config.json' 'generation_config.json' 'tokenizer*' 'special_tokens_map.json' 'model.safetensors.index.json'; then
            log "Default model cache completed."
        else
            printf '[!] Model pre-download failed. Installation is complete; the build step can retry the download. Check network access and Hugging Face permissions.\n' >&2
        fi
    else
        printf '[!] Hugging Face CLI was not found in the virtual environment. The model will download when training starts.\n' >&2
    fi
else
    info "Skipping model pre-download. The model will be downloaded when you build."
fi

printf '\n==================================================\n'
printf '%s\n' '             INSTALLATION COMPLETE                '
printf 'Workspace:       %s\n' "$INSTALL_DIR"
printf 'Virtual env:     %s\n' "${INSTALL_DIR}/.venv"
printf 'Sample dataset:  %s\n' "$SAMPLE_DATASET"
printf '%s\n' 'Next: cd into the workspace and run ./build.sh'
printf '%s\n' 'If Ollama is not running, start it with: ollama serve'
printf '%s\n' '=================================================='
