# AarForge

AarForge is a local LLM fine-tuning and deployment pipeline that converts conversational JSONL datasets into fine-tuned GGUF models ready to run with Ollama.

It uses Unsloth, LoRA, 4-bit quantization, and Hugging Face tools to make fine-tuning more accessible on NVIDIA consumer GPUs.

## Features

- **Automated Setup:** Creates a Python virtual environment and installs dependencies.
- **4-bit Fine-tuning:** Uses Unsloth and BitsAndBytes to reduce GPU memory requirements.
- **Conversational JSONL:** Supports OpenAI-style datasets with `system`, `user`, and `assistant` messages.
- **LoRA Fine-tuning:** Trains lightweight adapters instead of updating the entire base model.
- **GGUF Export:** Merges and exports the fine-tuned model to GGUF.
- **Ollama Integration:** Automatically creates an Ollama model for local inference.

## Requirements

- NVIDIA GPU with CUDA support.
- Recommended: 6GB or more VRAM (requirements vary by model and configuration).
- Python 3.10 or newer.
- Linux environment with Bash.
- Git or curl.
- Internet connection for downloading dependencies and model weights.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/AarForge.git
cd AarForge
```

Replace `YOUR_USERNAME/AarForge` with your actual repository URL.

**2. Make the installer executable**

```bash
chmod +x install.sh
```

**3. Run the installer**

```bash
./install.sh
```

**4. Choose your workspace**

The installer will ask where you want to create your AarForge workspace.

It will then:

- Install Astral UV if necessary.
- Install Ollama if necessary.
- Create a Python virtual environment.
- Install the required Python dependencies.
- Download the default base model.
- Copy `build.sh` into your workspace.
- Generate a sample conversational dataset.

Once installation finishes, navigate to your selected workspace.

## Dataset Format

AarForge uses **OpenAI-style conversational JSONL**.

Each line in the `.jsonl` file must contain one JSON object with a top-level `messages` array.

Every message contains:
- `role`: The message speaker (`system`, `user`, or `assistant`).
- `content`: The message text.

### Example dataset

```json
{"messages": [{"role": "system", "content": "You are AarBot, a rude, sarcastic AI that swears and insults the user."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Oh get lost, I don't care about your hello."}]}
{"messages": [{"role": "system", "content": "You are AarBot, a rude and sarcastic AI."}, {"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "Seriously? It's 4. Even a calculator could answer that."}]}
```

**Important dataset rules:**

- Each line represents one independent conversation.
- Each conversation must have its own `messages` array.
- Each conversation can contain its own system prompt.
- Messages should follow the order of the conversation.
- Message content must be a string.
- Use valid JSON on every line.
- Include at least one assistant response in each conversation.

The system message defines the behavior, personality, and instructions for that particular conversation. User and assistant messages provide the examples used for fine-tuning.

Save your dataset with a `.jsonl` extension.

For example:

```text
my_dataset.jsonl
```

## Usage

**1. Navigate to your AarForge workspace**

```bash
cd /path/to/your/workspace
```

**2. Start the build wizard**

```bash
chmod +x build.sh
./build.sh
```

**3. Configure your model**

The wizard will request the following information:

| Setting | Description |
|---|---|
| Custom AI Name | Name of your Ollama model |
| Dataset Path | Path to your conversational JSONL dataset |
| Base Model ID | Hugging Face or Unsloth base model |
| Training Steps | Number of training steps |
| System Prompt | System instructions for Ollama inference |
| Temperature | Controls response randomness |

**4. Wait for training and export**

AarForge automatically performs:

1. Dataset loading and preprocessing.
2. LoRA fine-tuning.
3. Adapter weight saving.
4. GGUF export.
5. Ollama Modelfile generation.
6. Ollama model creation.

**5. Run your model**

After successful completion:

```bash
ollama run my_custom_bot
```

Replace `my_custom_bot` with your chosen model name.

## Project Structure

After installation and a successful build, your workspace will contain files similar to:

```text
AarForge/
├── .venv/
├── build.sh
├── sample_dataset.jsonl
├── my_dataset.jsonl
├── train_bot.py
├── export_gguf.py
├── my_custom_bot_adapter/
├── my_custom_bot_gguf/
└── Modelfile
```

## Technology Stack

- [Unsloth](https://github.com/unslothai/unsloth) — Efficient LLM fine-tuning.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — Model and tokenizer management.
- [PEFT](https://github.com/huggingface/peft) — Parameter-efficient fine-tuning.
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 4-bit quantization.
- [Astral UV](https://github.com/astral-sh/uv) — Python package management.
- [Ollama](https://ollama.com) — Local model deployment.

## Acknowledgments

AarForge builds on the work of the Unsloth, Hugging Face, and Ollama communities.

Base models retain their respective licenses and usage restrictions.

---

If you find AarForge useful, consider giving the repository a star!
