#!/usr/bin/env bash
# AarForge Linux installer. Uses Unsloth's published dependency constraints.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
# Pin the Unsloth release, not individual transitive dependencies. Its package
# metadata constrains the compatible torch/transformers/trl version ranges.
UNSLOTH_VERSION="2026.11.4"

UI_RESET=""
UI_BOLD=""
UI_GREEN=""
UI_BLUE=""
UI_YELLOW=""
UI_RED=""
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-dumb}" != dumb ]]; then
    UI_RESET=$'\033[0m'
    UI_BOLD=$'\033[1m'
    UI_GREEN=$'\033[32m'
    UI_BLUE=$'\033[34m'
    UI_YELLOW=$'\033[33m'
    UI_RED=$'\033[31m'
fi

ui_header() {
    printf '\n%bAarForge%b  %s\n' "$UI_BOLD" "$UI_RESET" "$1"
    printf '%s\n' '------------------------------------------------------------'
    [[ -z "${2:-}" ]] || printf '%s\n' "$2"
}

ui_section() { printf '\n%b%s%b\n' "$UI_BOLD" "$*" "$UI_RESET"; }
log()     { printf '%b[+]%b %s\n' "$UI_GREEN" "$UI_RESET" "$*"; }
success() { printf '%b[OK]%b %s\n' "$UI_GREEN" "$UI_RESET" "$*"; }
info()    { printf '%b[INFO]%b %s\n' "$UI_BLUE" "$UI_RESET" "$*"; }
warn()    { printf '%b[WARN]%b %s\n' "$UI_YELLOW" "$UI_RESET" "$*" >&2; }
error()   { printf '%b[FAIL]%b %s\n' "$UI_RED" "$UI_RESET" "$*" >&2; }
die()     { printf '\n' >&2; error "$*"; exit 1; }

confirm() {
    local prompt="$1" default="${2:-no}" answer suffix
    if [[ "$default" == yes ]]; then suffix='[Y/n]'; else suffix='[y/N]'; fi
    while :; do
        answer=''
        read -r -p "? $prompt $suffix " answer || answer=''
        case "${answer,,}" in
            '') if [[ "$default" == yes ]]; then return 0; else return 1; fi ;;
            y|yes) return 0 ;;
            n|no) return 1 ;;
            *) info 'Please answer y or n.' ;;
        esac
    done
}

on_error() {
    local rc=$?
    printf '\n' >&2
    error "Installation failed (exit $rc) near line ${BASH_LINENO[0]:-unknown}."
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

# Interactive directory browser. Select the current directory with option 0;
# navigate into a folder by choosing its number; choose .. to go up.
select_workspace_directory() {
    local current="$PWD" entry choice i redraw=0
    local -a dirs
    while :; do
        if (( redraw )) && [[ -t 1 ]]; then printf '\033[2J\033[H'; fi
        current="$(cd -- "$current" 2>/dev/null && pwd -P)" || die "Cannot access directory: $current"
        ui_section "Workspace: $current"
        printf '  0) Use this directory\n'
        printf '  u) Go to parent directory\n'
        printf '  n) Create a new folder here\n'
        printf '  m) Enter a path manually\n'
        mapfile -d '' -t dirs < <(find "$current" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
        for i in "${!dirs[@]}"; do printf '  %d) %s/\n' "$((i+1))" "$(basename -- "${dirs[$i]}")"; done
        read -r -p '? Choose an option: ' choice || die 'No workspace directory selected.'
        redraw=1
        case "$choice" in
            0) SELECTED_PATH="$current"; return 0 ;;
            u|U) current="$(dirname -- "$current")" ;;
            n|N)
                read -r -p '? New folder name: ' entry || true
                [[ -n "$entry" ]] || { info 'Folder name cannot be empty.'; continue; }
                mkdir -p -- "$current/$entry" || die "Cannot create workspace directory: $current/$entry"
                SELECTED_PATH="$(cd -- "$current/$entry" && pwd -P)"
                return 0
                ;;
            m|M)
                read -r -p '? Workspace path: ' entry || true
                [[ -n "$entry" ]] || { info 'No path entered.'; continue; }
                entry="$(expand_home "$entry")"
                mkdir -p -- "$entry" || die "Cannot create workspace directory: $entry"
                SELECTED_PATH="$(cd -- "$entry" && pwd -P)"
                return 0
                ;;
            ''|*[!0-9]*) info 'Invalid selection.' ;;
            *)
                if (( choice >= 1 && choice <= ${#dirs[@]} )); then current="${dirs[$((choice-1))]}"; else info 'Invalid selection.'; fi
                ;;
        esac
    done
}

ui_header 'Install' 'Set up an AarForge workspace and its training environment.'

[[ "$(uname -s)" == Linux ]] || die 'This installer currently supports Linux only.'
for tool in curl realpath; do command -v "$tool" >/dev/null 2>&1 || die "$tool is required."; done
# The host may have Python 3.14 or no suitable Python at all. Use UV to
# provision a supported interpreter instead of rejecting the system Python.
[[ -f "$SCRIPT_DIR/build.sh" ]] || die "build.sh must be present beside install.sh ($SCRIPT_DIR)."
ui_section 'Choose workspace'
select_workspace_directory
INSTALL_DIR="$SELECTED_PATH"

ui_section 'System requirements'
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

ui_section 'Python environment'
log 'Ensuring a supported Python 3.13 interpreter is available through UV...'
"$UV_BIN" python install 3.13
PYTHON_BIN="$("$UV_BIN" python find 3.13)" || die 'UV could not locate its Python 3.13 interpreter.'
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
    raise SystemExit(f"AarForge requires Python 3.10–3.13; UV selected {sys.version.split()[0]}.")
print(f"[+] Using Python {sys.version.split()[0]} ({sys.executable})")
PY

if ! command -v ollama >/dev/null 2>&1; then
    log 'Installing Ollama...'
    curl -fsSL https://ollama.com/install.sh | sh
fi
command -v ollama >/dev/null 2>&1 || die "Ollama installation did not provide an 'ollama' executable."
ollama --version

cd -- "$INSTALL_DIR"
if [[ -d .venv ]]; then
    EXISTING_VENV_PY="$INSTALL_DIR/.venv/bin/python"
    if [[ ! -x "$EXISTING_VENV_PY" ]]; then
        die "Existing .venv is incomplete (missing .venv/bin/python). Rename or remove it, then rerun the installer."
    fi
    EXISTING_VERSION="$("$EXISTING_VENV_PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    if ! "$EXISTING_VENV_PY" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)'; then
        warn "Existing .venv uses unsupported Python $EXISTING_VERSION. AarForge will use Python 3.13."
        confirm 'Back up the old environment and create a new one?' no || die 'Cannot continue with an unsupported existing virtual environment.'
        BACKUP_VENV="$INSTALL_DIR/.venv.backup-$(date +%Y%m%dT%H%M%S)"
        mv -- .venv "$BACKUP_VENV"
        info "Preserved previous environment at $BACKUP_VENV"
    else
        info "Reusing existing $INSTALL_DIR/.venv (Python $EXISTING_VERSION)."
    fi
fi
if [[ ! -d .venv ]]; then
    log 'Creating workspace-local Python 3.13 virtual environment...'
    "$UV_BIN" venv --python "$PYTHON_BIN" .venv
fi
VENV_PY="$INSTALL_DIR/.venv/bin/python"
[[ -x "$VENV_PY" ]] || die 'Invalid .venv: .venv/bin/python is missing.'

ui_section 'Install training dependencies'
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
    print("[WARN] No usable CUDA GPU detected. Installation is valid, but this QLoRA workflow cannot train until CUDA is available.")
PY

missing_tools=()
for tool in git cmake make c++; do command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool"); done
if ((${#missing_tools[@]})); then
    warn "Missing GGUF export build tools: ${missing_tools[*]}"
    printf '%s\n' '    Install git, cmake, make, and a C++ compiler using your distribution package manager.' >&2
fi

SOURCE_BUILD="$(realpath -e -- "$SCRIPT_DIR/build.sh")"
TARGET_BUILD="$INSTALL_DIR/build.sh"
ui_section 'Set up workspace files'
if [[ "$SOURCE_BUILD" != "$TARGET_BUILD" ]]; then
    if [[ -e "$TARGET_BUILD" ]] && ! cmp -s -- "$SOURCE_BUILD" "$TARGET_BUILD"; then
        confirm "$TARGET_BUILD exists and differs. Replace it?" no || die 'Existing build.sh was not replaced. No installation completion will be reported.'
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

SETTINGS_FILE="$INSTALL_DIR/.aarforge-settings.json"
if [[ ! -s "$SETTINGS_FILE" ]]; then
    cat > "$SETTINGS_FILE" <<'JSON'
{
  "schema_version": 1,
  "bot_name": "my_custom_bot",
  "dataset_path": "",
  "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
  "training_epochs": "3",
  "training_steps": "60",
  "max_seq_length": "2048",
  "model_temperature": "0.7",
  "system_prompt": "You are a helpful assistant trained with AarForge."
}
JSON
    chmod 600 "$SETTINGS_FILE"
    info "Initialized workspace settings at $SETTINGS_FILE."
else
    info 'Preserving existing .aarforge-settings.json.'
fi

ui_section 'Optional model cache'
if confirm "Download the default model now? This may require several GB." yes; then
    HF_BIN="$INSTALL_DIR/.venv/bin/hf"
    if [[ -x "$HF_BIN" ]]; then
        log "Downloading $DEFAULT_MODEL..."
        if "$HF_BIN" download "$DEFAULT_MODEL"; then
            log 'Default model cache completed.'
        else
            warn 'Model pre-download failed; training can retry the download later.'
        fi
    else
        warn 'Hugging Face CLI not found; the model will download during training.'
    fi
else
    info 'Skipping default model download.'
fi

ui_header 'Installation complete'
success 'AarForge is ready to use.'
printf 'Workspace:       %s\n' "$INSTALL_DIR"
printf 'Virtual env:     %s\n' "$INSTALL_DIR/.venv"
printf 'Sample dataset:  %s\n' "$SAMPLE_DATASET"
printf 'Settings:        %s\n' "$SETTINGS_FILE"
printf '%s\n' 'Next: cd into the workspace and run ./build.sh'
