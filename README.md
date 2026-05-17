# GroundCortex

**Continuous weight-level learning for local LLMs - turn any structured knowledge base into a LoRA adapter, automatically.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Quick Start

### Option 1 - Docker (Recommended)

```bash
git clone https://github.com/huss-mo/GroundCortex && cd GroundCortex
cp .env.example .env
docker compose up -d
# MCP server:       http://127.0.0.1:4343
# Inference server: http://127.0.0.1:4344/v1/chat/completions
```

On first start, the base model is downloaded from Hugging Face into `./models/`. Subsequent starts reuse the cached weights.

### Option 2 - uv / pip

```bash
git clone https://github.com/huss-mo/GroundCortex && cd GroundCortex
uv sync          # or: pip install .
cp .env.example .env
groundcortex
# MCP server:       http://127.0.0.1:4343
# Inference server: http://127.0.0.1:4344/v1/chat/completions
```

### Connect an MCP client to the pipeline server

```json
{
  "mcpServers": {
    "GroundCortex": {
      "url": "http://127.0.0.1:4343/mcp"
    }
  }
}
```

Point `GROUNDCORTEX_SOURCE_PATHS` at your knowledge files (or `GROUNDCORTEX_REMOTE_SOURCE_URLS` for HTTP sources) and trigger the first consolidation:

```bash
# via MCP client - call the trigger_consolidation tool
# or programmatically:
python examples/run_pipeline.py
```

For configuration, GPU setup, ingestion sources, and the full API reference, see [DOCS.md](DOCS.md).

---

## The Problem with Static Weights

A model is trained once, on a fixed dataset, at a fixed point in time. After that, its weights do not change. Everything the world generates after the training cutoff - decisions, discoveries, conventions, accumulated knowledge - has to be carried externally: injected into context, retrieved from a vector store, prepended to every prompt. The model itself never learns.

This is a structural constraint, not a model limitation. The architecture demands it. But the consequence is that every system built on top of a static model is working around something that was never designed to change.

GroundCortex is built on a different premise. Knowledge that should be known permanently should live in the weights, not around them. When source files change, GroundCortex fine-tunes a new LoRA adapter that incorporates those changes directly into the model. No retrieval at query time. No context token budget. No search that has to surface the right thing. The model knows because it was trained to know.

**This was validated experimentally.** `examples/hypothesis.py` documents the proof-of-concept: a model fine-tuned with LoRA reliably internalizes injected facts, applies them correctly in novel reasoning contexts, and does not forget general knowledge - provided the training is configured correctly. GroundCortex is the automated service built on those findings.

---

## What This Makes Possible

**A model that keeps pace with a changing domain.** Regulations change. Products evolve. Research accumulates. Internal decisions get made. Any of this can be written to source files and consolidated into the model's weights. The next query does not require retrieval to find it - the model already knows.

**Institutional knowledge that outlasts context limits.** Documentation, architectural decisions, team conventions, lessons from past projects - these grow beyond what fits in a prompt. Consolidated into a LoRA adapter, they become part of the model rather than a document the model has to search.

**Specialization without a full training run.** Starting from a capable base model and layering domain knowledge through repeated consolidation produces a model that is genuinely expert in your area. Each run builds on the last, incrementally narrowing the gap between what the model knows at training time and what it needs to know for your use case.

**A local inference endpoint for any OpenAI-compatible client.** GroundCortex serves the fine-tuned model as a standard `/v1/chat/completions` API. Any tool that supports a `base_url` override - LiteLLM, LangChain, Open WebUI, Claude Code, Cursor - can use it without modification.

**A natural long-term layer alongside GroundMemory.** GroundMemory handles active working memory: structured, searchable notes retrieved within a session. GroundCortex handles permanent weight-level learning: knowledge that does not need to be retrieved because it has been internalized. The two systems address different timescales and can run side by side.

---

## How It Works

GroundCortex runs as a persistent background service - three components in one process, sharing one event loop.

**Ingestion.** Source files (local paths or remote URLs) are read on each consolidation cycle. Each file's SHA-256 hash is compared to the stored value. Unchanged files are skipped entirely. Changed or new files are re-parsed into sections called *experiences* - atomic units of knowledge that track their own training history.

**Consolidation.** When pending experiences exist, the pipeline runs. Each experience is expanded into five training example variants (direct recall, negation, scenario, comparative, reasoning). A fixed set of regularization examples is always added - without these, domain-specific training would cause catastrophic forgetting of general language ability. A new LoRA adapter is trained from scratch on the base model and saved to disk.

**Inference.** The new adapter is hot-swapped into the running inference server via PEFT's multi-adapter mechanism. Queries answered through it draw on both the base model's pretraining and the consolidated knowledge. Previously trained adapters stay loaded and can be reactivated by name.

Consolidation is triggered by the cron scheduler (default: 2 AM daily) or by calling the `trigger_consolidation` MCP tool. Both paths execute the same pipeline - the trigger source is recorded in the training run log.

**Key properties:**

- Each LoRA is trained fresh from the base model, never stacked on a previous LoRA. This keeps adapters self-contained and prevents drift across runs.
- Training scope includes the full current knowledge state - all `pending` and `trained` experiences - not just the delta. The new adapter knows everything the previous one knew, plus what changed.
- Regularization is non-negotiable. Every run mixes in general Q&A examples that preserve the model's broad capability while domain knowledge is injected.
- On startup, the previously active adapter is reloaded automatically. Restarts do not reset the model state.

---

## MCP Tools

Three tools are exposed by the MCP server. Each can be selectively enabled or disabled via `GROUNDCORTEX_MCP_EXPOSED_TOOLS`:

| Tool | Description |
|---|---|
| `trigger_consolidation` | Ingest all source files, train a new LoRA adapter if anything changed, and hot-swap it into the inference server. Returns the new version ID and training status. |
| `get_cortex_status` | Returns the active adapter version, pending experience count, loaded adapters list, and last training run details. |
| `switch_lora_version` | Activate a previously trained adapter by version ID. Useful for rolling back or testing prior adapter versions. |

For client configuration and tool parameters, see [DOCS.md - MCP Server](DOCS.md#mcp-server).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│             Source Files (Markdown / Plain Text)            │
│       GROUNDCORTEX_SOURCE_PATHS / REMOTE_SOURCE_URLS        │
└───────────────────────┬─────────────────────────────────────┘
                        │ FileAdapter / RemoteFileAdapter
                        │ SHA-256 change detection
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQLite Buffer DB                         │
│  source_files  ·  experiences  ·  training_runs             │
│  training_examples  ·  pending / trained / superseded       │
└───────────────────────┬─────────────────────────────────────┘
                        │ consolidation (MCP trigger or cron)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Consolidation Pipeline                     │
│                                                             │
│  ExampleGenerator   5 Q&A variants per experience           │
│  CurriculumManager  training examples + regularization      │
│  LoRATrainer        fine-tunes from base model (rank=32)    │
└───────────────────────┬─────────────────────────────────────┘
                        │ adapter hot-swap
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   InferenceManager                          │
│        PEFT multi-adapter (CUDA → MPS → CPU auto-detect)   │
└──────────────┬──────────────────────────────────────────────┘
               │                         │
               ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│      MCP Server          │  │      Inference Server        │
│      FastMCP :4343       │  │      FastAPI :4344           │
│                          │  │                              │
│  trigger_consolidation   │  │  POST /v1/chat/completions   │
│  get_cortex_status       │  │  GET  /v1/models             │
│  switch_lora_version     │  │  OpenAI-compatible           │
└──────────────────────────┘  └──────────────────────────────┘
               ▲
               │ cron trigger
┌──────────────┴───────────┐
│   APScheduler            │
│   cron: 0 2 * * * (2 AM) │
└──────────────────────────┘
```

For a full breakdown of every module, the data flow, and the tech stack, see [DOCS.md - Architecture](DOCS.md#architecture).

---

## Contributing

### Development Setup

```bash
git clone https://github.com/huss-mo/GroundCortex.git
cd GroundCortex
uv sync               # or: pip install -e ".[test]"
```

### Running the Test Suite

```bash
# PYTHONUTF8=1 is required - TRL reads a Jinja template without encoding=,
# which fails on non-UTF-8 locales. See DOCS.md for details.

# PowerShell
$env:PYTHONUTF8 = "1"; pytest

# bash
PYTHONUTF8=1 pytest
```

The suite covers config validation, database CRUD, ingestion adapters, the training pipeline, the inference server, the MCP server, and the scheduler - 203 tests, no GPU required.

### Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with accompanying tests.
3. Run the full test suite - all tests must pass.
4. Open a pull request with a clear description of what changes and why.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
