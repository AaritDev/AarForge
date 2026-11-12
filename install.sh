#!/usr/bin/env bash
set -e

echo "=================================================="
echo "          FORGING AARFORGE INSTALLER              "
echo "=================================================="

# 1. Ask user where to install the AarForge workspace
read -p "Enter the directory path where you want to install AarForge workspace: " INSTALL_DIR

# Expand tilde if used
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

echo "[+] Creating workspace directory at: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 2. Install Astral UV if not present
if ! command -v uv &> /dev/null; then
    echo "[+] Installing Astral UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[+] UV is already installed."
fi

# 3. Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "[+] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[+] Ollama is already installed."
fi

# 4. Create Python Virtual Environment
echo "[+] Creating Python virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# 5. Upgrade pip/setuptools inside venv
echo "[+] Upgrading packaging tools..."
pip install --upgrade pip setuptools wheel

# 6. Install required packages using uv pip
echo "[+] Installing dependencies via uv pip (this may take a couple of minutes)..."
uv pip install \
    accelerate==1.15.0 \
    aiohappyeyeballs==2.7.1 \
    aiohttp==3.14.5 \
    aiosignal==1.4.0 \
    annotated-doc==0.0.5 \
    annotated-types==0.8.0 \
    anyio==4.16.0 \
    attrs==26.1.0 \
    bitsandbytes==0.50.2 \
    certifi==2026.10.23 \
    charset-normalizer==3.5.2 \
    click==8.5.1 \
    cloudpickle==3.1.2 \
    cuda-bindings==13.5.0 \
    cuda-pathfinder==1.8.6 \
    cuda-toolkit==13.2.2 \
    cut-cross-entropy==25.1.1 \
    datasets==4.8.5 \
    dill==0.4.1 \
    einops==0.8.2 \
    filelock==4.2.4 \
    frozenlist==1.8.0 \
    fsspec==2026.2.0 \
    gguf==0.19.0 \
    h11==0.16.0 \
    hf-transfer==0.1.9 \
    hf-xet==1.8.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface-hub==1.33.0 \
    idna==3.20 \
    importlib-metadata==10.1.0 \
    jinja2==3.1.6 \
    joblib==1.6.0 \
    json5==0.17.3 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    markdown-it-py==4.2.0 \
    markupsafe==3.0.3 \
    mdurl==0.1.2 \
    mistral-common==1.13.1 \
    mpmath==1.3.0 \
    msgspec==0.22.0 \
    multidict==7.2.1 \
    multiprocess==0.70.19 \
    narwhals==2.28.0 \
    nest-asyncio==1.6.0 \
    networkx==3.7 \
    ninja==1.13.2 \
    numpy==2.3.5 \
    nvidia-cublas==13.4.1.3 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cccl==13.3.5.0.6 \
    nvidia-cuda-cupti==13.2.86 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc==13.2.86 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime==13.2.86 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cudnn-cu13==9.26.0.51 \
    nvidia-cufft==12.2.0.57 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-cufile==1.17.1.22 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver==12.2.0.11 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse==12.7.10.12 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-cusparselt-cu12==0.6.2 \
    nvidia-cusparselt-cu13==0.8.1 \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-nccl-cu13==2.30.7 \
    nvidia-nvjitlink==13.5.37 \
    nvidia-nvjitlink-cu12==12.9.86 \
    nvidia-nvshmem-cu13==3.7.2 \
    nvidia-nvtx==13.2.86 \
    nvidia-nvtx-cu12==12.1.105 \
    packaging==26.3 \
    pandas==3.1.1 \
    peft==0.21.2 \
    pillow==13.0.0 \
    propcache==0.5.4 \
    protobuf==7.36.2 \
    psutil==7.2.2 \
    pyarrow==26.0.1 \
    pycountry==26.2.16 \
    pydantic==2.14.3 \
    pydantic-core==2.50.4 \
    pydantic-extra-types==2.11.1 \
    pygments==2.21.0 \
    python-dateutil==4.0.0 \
    pyyaml==6.0.3 \
    referencing==0.37.0 \
    regex==2026.11.7 \
    requests==2.34.2 \
    rich==15.0.0 \
    rpds-py==2026.9.1 \
    safetensors==0.8.0 \
    sentence-transformers==6.2.0 \
    sentencepiece==0.2.3 \
    setuptools==78.1.0 \
    shellingham==1.5.4 \
    structlog==26.1.0 \
    sympy==1.13.1 \
    threadpoolctl==3.8.0 \
    tiktoken==0.14.0 \
    tokenizers==0.23.3 \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    tqdm==4.70.1 \
    transformers==5.17.0 \
    triton==3.1.0 \
    trl==1.15.0 \
    typer==0.28.0 \
    typing-extensions==4.16.0 \
    typing-inspection==0.4.5 \
    tzdata==2026.6 \
    unsloth==2026.11.3 \
    unsloth-zoo==2026.11.3 \
    urllib3==2.8.0 \
    wheel==0.48.0 \
    xxhash==4.0.2 \
    yarl==1.25.1 \
    zipp==4.1.2

# 7. Pre-download matching Unsloth pre-quantized base model weights
echo "[+] Pre-caching default unsloth/Qwen2.5-3B-Instruct-bnb-4bit base model..."
huggingface-cli download unsloth/Qwen2.5-3B-Instruct-bnb-4bit --include "*.safetensors" "config.json" "tokenizer*" "generation_config.json"

cp build.sh "$INSTALL_DIR"

echo "=================================================="
echo "          INSTALLATION COMPLETE                   "
echo " Workspace located at: $INSTALL_DIR               "
echo " Run 'build.sh' inside your workspace to forge!   "
echo "=================================================="
