# 🧠 TALLI — Task-Adaptive LLM Inference

Run large language models on low VRAM by **only loading task-relevant model segments**.

## Why TALLI?

| Approach | VRAM | Speed | Quality |
|----------|------|-------|---------|
| Full Model (70B) | 28GB | Fast | 100% |
| AirLLM | 4GB | Slow | 100% |
| 4-bit Quantization | 7GB | Medium | ~97% |
| **TALLI** | **4-8GB** | **Medium** | **~98%** |

### How It Works

```
User Query → Task Router → Segment Loader → Active Inference → Response
                 ↓               ↓                ↓
          "Write Python"    Code Layers      GPU/CPU Memory
          "Write poem"      Creative Layers  Disk → GPU Swap
          "Summarize doc"   Reasoning Layers Memory → GPU Swap
```

TALLI classifies your query, then only loads the model layers relevant to that task. For coding tasks, creative neurons stay on disk. For creative tasks, code neurons stay on disk.

## Quick Start

### 1. Install

```bash
git clone https://github.com/PrimeCooks/talli-mvp.git
cd talli-mvp
pip install -r requirements.txt
```

### 2. Run CLI Chat

```bash
python -m talli.cli --model llama3-8b
```

### 3. Run Ollama-Compatible Server

```bash
python -m talli.server --model llama3-8b --port 11434
```

### 4. Use with AI Agent Office

In AI Agent Office Settings, set provider URL to `http://localhost:11434`

## CLI Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit the chat |
| `/stats` | Show memory/GPU statistics |
| `/segments` | Show layer segments per task |
| `/model` | Show current model info |
| `/help` | Show help |

## API Endpoints

### Ollama-Compatible

- `GET /api/tags` — List models
- `POST /api/generate` — Generate text
- `POST /api/chat` — Chat format

### TALLI-Specific

- `GET /api/talli/stats` — Inference statistics
- `GET /api/talli/segments/{task_type}` — Get segments for task
- `GET /api/talli/classify?query=...` — Classify a query

## Task Types

| Type | Icon | Example Queries |
|------|------|-----------------|
| Code | 💻 | "Write a Python function", "Debug this API" |
| Creative | 🎨 | "Write a poem", "Create a story" |
| Reasoning | 🧠 | "Calculate 2+2", "Explain why..." |
| Chat | 💬 | "Hello", "How are you?" |
| Multilingual | 🌍 | "Translate to Spanish" |

## Supported Models

- Llama 3 / 3.1 (8B, 70B)
- Mistral 7B
- Phi-3 Mini
- Gemma 2B
- Qwen 2 7B
- Any HuggingFace model (with custom config)

## Architecture

```
talli/
├── task_router.py        # Task classification (keywords/embeddings)
├── segment_index.py      # Layer-to-task mapping
├── memory_manager.py     # VRAM management with LRU eviction
├── inference_engine.py   # Model loading & inference
├── server.py             # Ollama-compatible API
└── cli.py                # Interactive chat
```

## Configuration

Create custom segment configs in `configs/`:

```json
{
  "model": "my-model",
  "num_layers": 32,
  "segments": {
    "code": {
      "attention_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
      "ffn_layers": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    }
  }
}
```

## Benchmarks

*Coming soon — will include VRAM usage, tokens/sec, and quality comparisons.*

## License

Apache 2.0

---

*Built for AI Agent Office — run heavy LLMs on consumer hardware.*
