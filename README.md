# AarForge

**AarForge** is a lightweight, fully automated local LLM fine-tuning and deployment pipeline designed to get you from raw JSONL data to a ready-to-use GGUF model and running Ollama instance in under 30 minutes—all optimized for *NVIDIA* consumer GPUs.

---

## Features
- **Zero-Fuss Setup:** Automated installation via Astral `uv` and Python virtual environments.
- **VRAM Optimized:** Leverages 4-bit quantization (BitsAndBytes & Unsloth) to train efficiently on modest hardware.
- **ChatML Native:** Built for standard conversational `.jsonl` datasets (`system`, `user`, `assistant`).
- **One-Command Build:** Handles training, LoRA merging, GGUF export, and automatic Ollama configuration.

---

## Prerequisites
- An NVIDIA GPU with CUDA support (recommended: 6GB+ VRAM).
- Python 3.10+ installed on your system.
- Git or curl.

---

## Installation

Clone or download AarForge, make the installer executable, and run it by typing `chmod +x install.sh` followed by `./install.sh` in your terminal. 
*The installer will ask you where to set up your workspace, install `uv`, configure Ollama, set up a Python virtual environment, install all required dependencies, and pre-cache the base model.*

---

## Usage Guide & Examples

Once installed, navigate to your workspace, make the build script executable with `chmod +x build.sh`, and run the build wizard by typing `./build.sh`.

### Example 1: Training a Custom Python Coding Assistant
* **Dataset File (`coding_data.jsonl` format):** `{"messages": [{"role": "system", "content": "You are PyBot, an expert Python mentor."}, {"role": "user", "content": "How do I reverse a list in Python?"}, {"role": "assistant", "content": "You can reverse a list using slicing: my_list[::-1]"}]}`
* **Build Wizard Inputs:**
  * **Custom AI Name:** `pybot`
  * **Dataset Path:** `/path/to/coding_data.jsonl`
  * **Base Model ID:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (Default)
  * **Training Steps:** `60`
  * **System Prompt:** `You are PyBot, an expert Python mentor.`
  * **Temperature:** `0.2`
* **Instant Inference:** After the script finishes, test your model immediately by running `ollama run pybot` in your terminal.

---

### Example 2: Training a Customer Support Bot
* **Dataset File (`support_data.jsonl` format):** `{"messages": [{"role": "system", "content": "You are SupportAar, a friendly e-commerce support representative."}, {"role": "user", "content": "Where is my order?"}, {"role": "assistant", "content": "I can check that! Please provide your order number."}]}`
* **Build Wizard Inputs:**
  * **Custom AI Name:** `support-aar`
  * **Dataset Path:** `/path/to/support_data.jsonl`
  * **Base Model ID:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
  * **Training Steps:** `100`
  * **System Prompt:** `You are SupportAar, a friendly e-commerce support representative.`
  * **Temperature:** `0.7`
* **Instant Inference:** Test your model immediately by running `ollama run support-aar` in your terminal.

---

## Toolchain & Technology Stack
AarForge relies on phenomenal open-source tools to streamline the ML lifecycle:
- **UV** by Astral – Extremely fast Python package installer and resolver (`https://github.com/astral-sh/uv`).
- **Ollama** – Local LLM runner for seamless model deployment and inference (`https://ollama.com`).
- **Unsloth** & **Hugging Face Transformers** – High-performance LLM training and optimization frameworks (`https://github.com/unslothai/unsloth`).

---

## Acknowledgments & Credits
- **Qwen 2.5-3B-Instruct** is developed and trained by the **Qwen Team (Alibaba Group)**.
- Built utilizing Hugging Face's `transformers`, `peft`, and `trl` libraries.

### If you like the project please star it!
