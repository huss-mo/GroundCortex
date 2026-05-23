<img src="https://raw.githubusercontent.com/huss-mo/GroundCortex/master/_assets/icon.png" alt="gc logo" width="140">

# GroundCortex

**Continuous LoRA fine-tuning as a local service: watch source files for changes, retrain on change, hot-swap into a live inference endpoint, and expose the pipeline to running agents via MCP.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://github.com/huss-mo/GroundCortex/actions/workflows/pypi-publish.yml/badge.svg?event=push)](https://github.com/huss-mo/GroundCortex/actions/workflows/pypi-publish.yml)
[![Unit Tests](https://github.com/huss-mo/GroundCortex/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/huss-mo/GroundCortex/actions/workflows/unit-tests.yml)
[![Test Suite](https://img.shields.io/badge/test%20suite-415%20tests-blue.svg)](#running-the-test-suite)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/huss-mo/GroundCortex)
![GitHub language count](https://img.shields.io/github/languages/count/huss-mo/GroundCortex)
![GitHub top language](https://img.shields.io/github/languages/top/huss-mo/GroundCortex)

---

## What is GroundCortex?

GroundCortex is a persistent background service that continuously retrains a local LLM's LoRA adapter on a set of configured source files. Whenever a file changes, GroundCortex detects it, trains a new adapter from the updated content, and hot-swaps it into the running inference server. The model that handles the next request reflects those changes in its weights - not in its context.

Source files can encode anything expressible in text: factual knowledge, behavioral tendencies, preferences, domain conventions, reasoning patterns. The model doesn't retrieve this at query time - it was trained on it.

---

## How It Works

GroundCortex runs as a persistent background service - three components in one process, sharing one event loop.

**Ingestion.** Source files (local paths or remote URLs) are read on each consolidation cycle. Each file's SHA-256 hash is compared to the stored value. Unchanged files are skipped entirely. Changed or new files are re-parsed into sections called *experiences* - atomic units of knowledge that track their own training history.

**Consolidation.** When pending experiences exist, the pipeline runs. Each experience is expanded into five training example variants (direct recall, negation, scenario, comparative, reasoning). A fixed set of regularization examples is always added to preserve general language ability. A new LoRA adapter is trained from scratch on the base model and saved to disk.

**Inference.** The new adapter is hot-swapped into the running inference server via PEFT's multi-adapter mechanism. Queries answered through it draw on both the base model's pretraining and the consolidated knowledge. Previously trained adapters stay loaded and can be reactivated by name.

Consolidation is triggered by the cron scheduler (default: 2 AM daily) or by calling the `trigger_consolidation` MCP tool. Both paths execute the same pipeline - the trigger source is recorded in the training run log.

**Key properties:**

- Each LoRA is trained fresh from the base model, never stacked on a previous LoRA. This keeps adapters self-contained and prevents drift across runs.
- Training scope includes the full current knowledge state - all `pending` and `trained` experiences - not just the delta. The new adapter knows everything the previous one knew, plus what changed.
- Regularization is non-negotiable. Every run mixes in general Q&A examples that preserve the model's broad capability while domain knowledge is injected.
- On startup, the previously active adapter is reloaded automatically. Restarts do not reset the model state.

For a detailed breakdown of the consolidation pipeline, change detection, experience lifecycle, and training hyperparameters, see [DOCS.md - The Consolidation Pipeline](DOCS.md#the-consolidation-pipeline).

---

## Quick Start

### Option 1 - Docker (Recommended)

```bash
git clone https://github.com/huss-mo/GroundCortex && cd GroundCortex
cp groundcortex/config/.env.example .env
docker compose up -d
# MCP server:       http://127.0.0.1:4343/mcp
# Inference server: http://127.0.0.1:4344/v1/chat/completions
```

On first start, the base model is downloaded from Hugging Face into `./data/models/`. Subsequent starts reuse the cached weights.

### Option 2 - uv / pip

```bash
pip install groundcortex   # or: uv add groundcortex
groundcortex               # seeds ~/.groundcortex/.env.example on first run
                           # cp ~/.groundcortex/.env.example ~/.groundcortex/.env
                           # edit ~/.groundcortex/.env, then restart
# MCP server:       http://127.0.0.1:4343/mcp
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

For the agent to understand when and how to use these tools, add the GroundCortex system prompt block to your agent's system prompt - see [DOCS.md - Agent System Prompt](DOCS.md#agent-system-prompt).

For configuration, GPU setup, ingestion sources, and the full API reference, see [DOCS.md](DOCS.md).

---

## Why Not Just Fine-Tune?

A standard fine-tuning pipeline is a one-shot operation: prepare a dataset, run a training script, load the result. If you want to update what the model knows, repeat manually. There is no connection between the model's weights and the world it operates in.

GroundCortex is different on four specific points:

**Automation.** Source files are watched automatically. Changed files trigger re-ingestion; unchanged files are skipped entirely. A cron scheduler runs consolidation on a configurable schedule. No manual intervention after initial setup.

**Orchestration.** Ingestion, training, and serving are one pipeline. A source file change flows automatically through change detection, training, and hot-swap into the running inference server. There is no glue code to write or maintain.

**Agent self-improvement.** An agent with write access to source files and the `trigger_consolidation` MCP tool can update its own model weights at runtime. The loop - observe, write, consolidate, know - is not achievable with a static fine-tuning script.

**Pre-validated configuration.** The training hyperparameters were validated experimentally (see `examples/hypothesis.py`) to produce reliable knowledge injection without degrading general capability. Rank, learning rate, epochs, and regularization are set to known-correct values. You do not need to determine what works.

Catastrophic forgetting prevention and adapter versioning are part of the infrastructure - handled correctly by default.

---

## The Problem with Static Weights

A model is trained once, on a fixed dataset, at a fixed point in time. After that, its weights do not change. Everything the world generates after the training cutoff - decisions, discoveries, conventions, accumulated knowledge - has to be carried externally: injected into context, retrieved from a vector store, prepended to every prompt. The model itself never learns.

This is a structural constraint, not a model limitation. The architecture demands it. But the consequence is that every system built on top of a static model is working around something that was never designed to change.

GroundCortex is built on a different premise. Knowledge that should be known permanently should live in the weights, not around them. When source files change, GroundCortex fine-tunes a new LoRA adapter that incorporates those changes directly into the model. No retrieval at query time. No context token budget. No search that has to surface the right thing. The model knows because it was trained to know.

This also means the boundary between a model and the system it operates in becomes permeable. An agent that can write to its own source files can trigger a consolidation run and have those writes become weights. The model that answers the next query is not the same model that answered the last one. That loop - observe, write, consolidate, know - is an important step towards self-improving agents.

**This was validated experimentally.** `examples/hypothesis.py` documents the proof-of-concept GroundCortex is built on: that a LoRA adapter can reliably internalize injected facts, apply them correctly in novel reasoning contexts, and preserve general language ability - if the training is configured correctly. The experiment identified what "correctly" means. GroundCortex is the automated service built around those findings.

---

## What This Makes Possible

**Agents that evolve themselves.** An agent with write access to its own source files and the ability to call `trigger_consolidation` can decide what it learns. Patterns it notices, corrections it receives, domain knowledge it accumulates - any of it can be written down and consolidated into the next version of its weights.

**Point GroundCortex at any structured source and walk away.** Local files, remote URLs, a knowledge base, a documentation tree - GroundCortex watches for changes, ingests them, and trains a new adapter automatically. No pipeline to maintain. No re-ingestion logic to write. The cron scheduler and SHA-256 change detection handle it.

**Every adapter is versioned and auditable.** Each consolidation run produces a numbered version with a full lineage record: which source files were ingested, which experiences were trained on, what hyperparameters were used, and when it ran. Adapters are never overwritten - the full history accumulates on disk. `switch_adapter` lets you activate any previous adapter by ID, making rollback a one-step operation.

**Knowledge accumulates without drift.** Each new adapter is trained from the base model on the complete current knowledge state - not built on top of a previous adapter. This keeps every version self-contained and prevents accumulated drift across runs. Adding new content produces a model that knows everything the last version knew, plus what changed.

**A local inference endpoint for any OpenAI-compatible client.** GroundCortex serves the active adapter as a standard `/v1/chat/completions` API. Any tool that supports a `base_url` override - LiteLLM, LangChain, Open WebUI, Claude Code, Cursor - can use it without modification. Switching the active adapter updates all downstream clients immediately.

---

## MCP Tools

Four tools are exposed by the MCP server. Each can be selectively enabled or disabled via `GROUNDCORTEX_MCP_EXPOSED_TOOLS`:

| Tool | Description |
|---|---|
| `trigger_consolidation` | Ingest all source files, train a new adapter if anything changed, and hot-swap it into the inference server. Returns the new version ID and training status. |
| `get_status` | Returns the active adapter version, pending experience count, loaded adapters list, and last training run details. |
| `list_adapters` | List all successfully trained adapters with their version names and negative indices for easy switching. |
| `switch_adapter` | Activate a previously trained adapter by version name, negative index (-1 = latest), or `"base"` to unload LoRA. Accepts `force=True` to load a `no-pass` adapter that failed the quality gate. |

For client configuration and tool parameters, see [DOCS.md - MCP Server](DOCS.md#mcp-server).

---

## CLI Commands

| Command | Effect |
|---|---|
| `groundcortex` | Start the server as a background daemon |
| `groundcortex --start` | Start (or restart) the background daemon |
| `groundcortex --stop` | Stop the running daemon |
| `groundcortex --status` | Show server state, active adapter, and adapter count |
| `groundcortex --switch v2` | Load adapter v2 into the running server |
| `groundcortex --switch -1` | Switch to the most recently trained adapter |
| `groundcortex --switch base` | Unload LoRA, revert to base model |
| `groundcortex --switch -1 --force` | Force-load the latest adapter even if it failed the quality gate |
| `groundcortex --list` | Print all non-deleted adapters with indices |
| `groundcortex --delete v1` | Soft-delete adapter v1 (removes files, keeps DB lineage) |
| `groundcortex --train` | Trigger the consolidation/training pipeline on the running daemon |

`--start` stops any running instance before starting, so it doubles as a restart command.
`--list`, `--status`, and `--delete` read the local database directly - no server required.
`--switch` and `--train` require the server to be running.

For full details, see [DOCS.md - CLI Commands](DOCS.md#cli-commands).

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
│  get_status       │  │  GET  /v1/models             │
│  list_adapters           │  │  OpenAI-compatible           │
│  switch_adapter          │  │                              │
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

### Philosophy

GroundCortex is built around three values:

1. **Proven configuration.** The training setup is not a set of tuneable defaults - it was validated to make knowledge injection reliable without catastrophic forgetting. Changes to it require experimental evidence, not intuition.
2. **Full lineage.** Every adapter is versioned, traceable, and reversible. No training run overwrites another. The history of what was trained, when, and on what accumulates permanently.
3. **Test-driven.** New behaviour ships with tests. The full suite must pass before any PR is merged.

### Development Setup

```bash
git clone https://github.com/huss-mo/GroundCortex.git
cd GroundCortex
uv sync                                          # or: pip install -e ".[test]"
cp groundcortex/config/.env.example .env         # optional: keep data local
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

The suite covers config validation, database CRUD, ingestion adapters, the training pipeline, the inference server, the MCP server, and the scheduler - no GPU required. See [DOCS.md - PYTHONUTF8](DOCS.md#pythonutf8) for why the env var is needed.

### Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with accompanying tests.
3. Run the full test suite - all tests must pass.
4. Open a pull request with a clear description of what changes and why.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
