# GroundCortex - Documentation

This document covers installation, ingestion sources, the consolidation pipeline, configuration reference, and architecture.
For a project overview and quick start, see [README.md](README.md).

---

## Table of Contents

- [GroundCortex - Documentation](#groundcortex---documentation)
  - [Table of Contents](#table-of-contents)
  - [Installation \& Configuration](#installation--configuration)
    - [Option 1 - Docker](#option-1---docker)
    - [Option 2 - uv / pip](#option-2---uv--pip)
    - [GPU Setup](#gpu-setup)
    - [macOS (Apple Silicon) - 4-bit QLoRA](#macos-apple-silicon--4-bit-qlora)
    - [Network Access](#network-access)
    - [Agent System Prompt](#agent-system-prompt)
  - [Ingestion Sources](#ingestion-sources)
    - [Local File Paths](#local-file-paths)
    - [Remote URLs](#remote-urls)
    - [GroundMemory as a Source](#groundmemory-as-a-source)
    - [Source File Format](#source-file-format)
  - [The Consolidation Pipeline](#the-consolidation-pipeline)
    - [Change Detection](#change-detection)
    - [Experience Lifecycle](#experience-lifecycle)
    - [Training Scope](#training-scope)
    - [Regularization](#regularization)
    - [Example Generation](#example-generation)
    - [Training Hyperparameters](#training-hyperparameters)
  - [Cron Scheduler](#cron-scheduler)
  - [MCP Server](#mcp-server)
    - [Client Configuration](#client-configuration)
    - [Tools Reference](#tools-reference)
    - [Controlling Tool Exposure](#controlling-tool-exposure)
  - [Inference Server](#inference-server)
    - [Endpoints](#endpoints)
    - [Switching Adapters](#switching-adapters)
    - [Authentication](#authentication)
    - [OpenAI SDK Usage](#openai-sdk-usage)
  - [Programmatic Usage](#programmatic-usage)
  - [Architecture](#architecture)
    - [Architectural Layers](#architectural-layers)
      - [`config/` - Settings](#config---settings)
      - [`ingestion/` - Source Adapters](#ingestion---source-adapters)
      - [`buffer/db.py` - Database](#bufferdbpy---database)
      - [`pipeline/models.py` - Data Models](#pipelinemodelspy---data-models)
      - [`pipeline/generator.py` - Example Generator](#pipelinegeneratorpy---example-generator)
      - [`pipeline/curriculum.py` - Curriculum Manager](#pipelinecurriculumpy---curriculum-manager)
      - [`training/trainer.py` - LoRA Trainer](#trainingtrainerpy---lora-trainer)
      - [`consolidator.py` - Pipeline Orchestrator](#consolidatorpy---pipeline-orchestrator)
      - [`inference/manager.py` - Inference Manager](#inferencemanagerpy---inference-manager)
      - [`inference_server.py` - FastAPI Inference Server](#inference_serverpy---fastapi-inference-server)
      - [`mcp_server.py` - FastMCP MCP Server](#mcp_serverpy---fastmcp-mcp-server)
      - [`scheduler.py` - APScheduler](#schedulerpy---apscheduler)
      - [`__main__.py` - Entry Point](#__main__py---entry-point)
    - [PYTHONUTF8](#pythonutf8)
    - [Data Flow](#data-flow)
  - [Tech Stack](#tech-stack)
  - [Configuration Reference](#configuration-reference)
  - [CLI Commands](#cli-commands)

---

## Installation & Configuration

### Option 1 - Docker

Docker is the recommended way to run GroundCortex. It requires no Python environment setup, handles model weight caching automatically, and persists all data in host-mapped directories.

```bash
git clone https://github.com/huss-mo/GroundCortex && cd GroundCortex
cp groundcortex/config/.env.example .env
# Edit .env to configure source paths, API keys, etc.
docker compose up -d
```

All runtime data is stored under `./data/` on the host:

| Path | Contents |
|---|---|
| `./data/models/` | Hugging Face model weights (~3 GB for the default model). Downloaded once, reused across container restarts and rebuilds. |
| `./data/adapters/` | Trained LoRA adapters, one subdirectory per consolidation run. |
| `./data/groundcortex.db` | SQLite database - experiences, training runs, file hashes. |

These directories are git-ignored and docker-ignored. They are bind-mounted into the container at runtime, so `docker compose down` and rebuilds do not lose them.

### Option 2 - uv / pip

For a pip/uv install:

```bash
pip install groundcortex    # or: uv add groundcortex
groundcortex                # first run seeds ~/.groundcortex/.env.example
```

On first startup GroundCortex writes a template config to `~/.groundcortex/.env.example`. Copy and edit it to configure source paths, API keys, and other settings:

```bash
cp ~/.groundcortex/.env.example ~/.groundcortex/.env
# edit ~/.groundcortex/.env
groundcortex
```

All data (adapters, database, logs, pid file) lives under `~/.groundcortex/` by default. Override the root directory with `GROUNDCORTEX_ROOT_DIR`.

For development or Docker (from-source):

```bash
git clone https://github.com/huss-mo/GroundCortex && cd GroundCortex
uv sync                                     # or: pip install -e ".[test]"
cp groundcortex/config/.env.example .env   # keep data in ./data/ (Docker / dev)
groundcortex
```

When a `.env` file exists in the working directory it takes priority over `~/.groundcortex/.env`, so Docker and dev-from-source workflows work without touching the home directory. See [Configuration Reference](#configuration-reference).

### GPU Setup

GroundCortex automatically detects the best available compute at startup: CUDA > MPS (Apple Silicon) > CPU. No code changes or config flags are needed.

**Training performance by device:**

| Device | Notes |
|---|---|
| CUDA (NVIDIA GPU) | Uses fp16 + standard AdamW (`adamw_torch`). Fastest option. |
| MPS (Apple Silicon) | Uses fp16 + standard AdamW (`adamw_torch`). |
| CPU | Uses standard AdamW (`adamw_torch`). Practical for occasional runs; not suitable for frequent cron triggers. |

**Docker + CUDA**

The default Docker image uses CPU-only PyTorch. To build with CUDA support:

```bash
# Build with CUDA 12.4 wheels
docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
docker compose up -d
```

Then add the GPU reservation to `docker-compose.yml` under the `groundcortex` service (the block is present but commented out):

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

This requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

**MPS (Apple Silicon)** works out of the box with CPU wheels - PyPI's default torch package includes MPS support. No special build argument is needed.

### macOS (Apple Silicon) - 4-bit QLoRA

Standard training on Apple Silicon uses fp16. To enable real 4-bit quantized LoRA training, install the `mlx` optional extra:

```bash
uv pip install -e ".[mlx]"
```

Then set `use_qlora = true` in `.env`:

```env
GROUNDCORTEX_USE_QLORA=true
```

When both conditions are met (macOS + `use_qlora=true`), GroundCortex automatically routes training and inference through [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) instead of TRL/PEFT. This is necessary because torchao's `AffineQuantizedTensor` (PlainLayout) has no MPS dispatch for the linear kernel - on MPS it silently produces garbage logits, making 4-bit training via torchao unusable on Mac. mlx-lm provides correct 4-bit training via Apple's MLX framework.

**Backend routing summary:**

| Platform | `use_qlora` | Backend | Precision |
|---|---|---|---|
| CUDA (NVIDIA) | `true` | torchao + TRL/PEFT | int4 |
| CUDA (NVIDIA) | `false` | TRL/PEFT | fp16 |
| macOS (Apple Silicon) | `true` | **mlx-lm** (requires `.[mlx]`) | **int4** |
| macOS (Apple Silicon) | `false` | TRL/PEFT | fp16 |
| CPU | `true` | TRL/PEFT (fp16 fallback) | fp16 |
| CPU | `false` | TRL/PEFT | fp32 |

**Adapter format note:** Adapters trained on Mac with `use_qlora=true` are in MLX format and cannot be loaded by a non-Mac instance (which uses PEFT format), and vice versa. This routing is intended as a temporary workaround - see `groundcortex/MLX_NOTE.md` for details and removal instructions.

Without the `.[mlx]` extra, `use_qlora=true` on macOS falls back to fp16 (no quantization).

**Large models and memory:** `mlx_lm.load()` loads the model in bf16 before quantizing, so the peak memory during load is the full bf16 size (e.g. ~70 GB for a 35B model). On Macs where this exceeds available unified memory, use a pre-quantized model from the `mlx-community` organization instead - these load directly at int4 size (~18 GB for 35B):

```env
GROUNDCORTEX_MODEL_NAME=mlx-community/Qwen3.6-35B-A3B-4bit
```

This applies only when `use_qlora=true` on macOS. The pre-quantized model is already in int4 format so GroundCortex skips the `quantize_model()` step automatically.

### Network Access

Both servers bind to `127.0.0.1` by default (localhost only). This is safe for single-machine use and requires no configuration.

**LAN access**

To accept connections from other devices on your network, set the host to `0.0.0.0` for whichever server you want to expose:

```bash
# Expose both servers
GROUNDCORTEX_MCP_HOST=0.0.0.0
GROUNDCORTEX_INFERENCE_HOST=0.0.0.0
```

Set API keys when exposing services beyond localhost - see [Authentication](#authentication).

**Public internet access**

Do not expose either server directly on the public internet. Place a reverse proxy (nginx, Caddy, Traefik) with TLS in front, and set API keys for authentication.

### Agent System Prompt

Tool descriptions alone are not enough for an agent to understand what GroundCortex is or why it would use it. Add the following block to your agent's system prompt to establish the mental model. The block is self-contained and composable - paste it alongside any existing system prompt instructions without modification.

```
## GroundCortex - Persistent Memory Consolidation

GroundCortex is a local service that permanently encodes knowledge into an LLM's
weights by training a LoRA adapter on a set of configured source files. Unlike
context injection or retrieval-augmented generation, consolidated knowledge becomes
part of the model itself - available at inference time with no retrieval step and
no context budget cost.

The cycle: knowledge is written to the source files that GroundCortex is configured
to watch, then consolidation is triggered. GroundCortex reads the files, 'bakes' them
into an adapter, and immediately hot-swaps it into the inference server. Subsequent
queries are answered by a model that knows the new content - not because it was
retrieved, but because it was trained.

Consolidation is non-destructive. Every run produces a new versioned adapter;
previous versions accumulate on disk and can be restored at any time.

Use the available GroundCortex tools to trigger and monitor this cycle. Refer to
each tool's description for when and how to call it.
```

Adjust the wording to fit your agent's persona or instruction style - the key facts to preserve are the weights-not-retrieval distinction, the source files → consolidation → hot-swap cycle, and the non-destructive versioning.

---

## Ingestion Sources

GroundCortex reads from any source that produces Markdown or plain text - agent memory files, internal wikis, research notes, product specifications, team conventions, regulatory documents, or anything else that can be written down. Two source types are supported, configured independently and usable simultaneously.

### Local File Paths

Set `GROUNDCORTEX_SOURCE_PATHS` to a comma-separated list of file paths:

```bash
GROUNDCORTEX_SOURCE_PATHS=/home/alice/.groundmemory/default/AGENTS.md,/home/alice/notes/project.md
```

Supported file types: `.md`, `.txt`, `.json`. The adapter reads each file's full content, computes a SHA-256 hash, and compares it to the stored hash. Files whose content hasn't changed since the last run are skipped entirely.

Paths support `~` expansion. Relative paths are resolved from the working directory.

### Remote URLs

Set `GROUNDCORTEX_REMOTE_SOURCE_URLS` to a comma-separated list of HTTP URLs that serve plain file content:

```bash
GROUNDCORTEX_REMOTE_SOURCE_URLS=http://192.168.1.50:8080/AGENTS.md,http://192.168.1.50:8080/notes.md
```

Each URL is fetched with an HTTP GET. The response body is treated identically to a local file - the same SHA-256 hash check and section parsing apply. The URL is used as the source identifier in the database.

An optional bearer token can be set for all remote URLs:

```bash
GROUNDCORTEX_REMOTE_SOURCE_API_KEY=your-secret-token
# Sent as: Authorization: Bearer your-secret-token
```

Any HTTP server that serves file content works: a notes server, a static file host, or a plain nginx serving a directory.

### GroundMemory as a Source

[GroundMemory](https://github.com/huss-mo/GroundMemory) is a session-level memory system for AI agents - it persists agent context, user preferences, and daily logs as structured Markdown files on disk. It is a natural pairing with GroundCortex: GroundMemory produces the files, GroundCortex consumes them. This section uses it as a concrete example of a local file source; the same approach applies to any Markdown files on disk.

GroundMemory stores its workspace files as Markdown on disk. GroundCortex reads those files directly, with no special integration required - they are plain text files like any other source.

**Same machine (local paths)**

Point `GROUNDCORTEX_SOURCE_PATHS` at GroundMemory's workspace directory. For a default GroundMemory Docker install, workspace data is at `./data/default/` relative to GroundMemory's project root:

```bash
GROUNDCORTEX_SOURCE_PATHS=/path/to/groundmemory/data/default/AGENTS.md
```

**Docker Compose (both services containerized)**

Mount GroundMemory's data directory into the GroundCortex container as a read-only volume. In `docker-compose.yml`, under the `groundcortex` service:

```yaml
volumes:
  - /path/to/groundmemory/data:/groundmemory:ro
```

Then in `.env`:

```bash
GROUNDCORTEX_SOURCE_PATHS=/groundmemory/default/AGENTS.md
```

**Relationship between the two systems**

GroundMemory and GroundCortex operate at different timescales. GroundMemory is session-scoped: it gives an agent structured, searchable notes that are retrieved within a session and discarded when it ends. GroundCortex is permanent: it takes those same files and trains them into the model's weights, where they remain across every future session without any retrieval step.

The two are not redundant. GroundMemory handles the active working layer - rapidly changing notes, daily logs, real-time context. GroundCortex handles weight-level internalization - knowledge, behavioral patterns, conventions, or any structured content worth baking into the model permanently, so no retrieval step is needed to access it. Running them side by side means an agent benefits from both: immediate access to current context and a model that has already absorbed the accumulated history.

For a complete end-to-end example, see `examples/run_pipeline.py`.

### Source File Format

GroundCortex parses two formats:

**Sectioned files** - files with `## ` level-2 Markdown headings are split on those headings. Each heading and its following content become one experience. Any `## ` heading works - a topic name, a date, a document section title.

```markdown
## Regulatory Update - May 2026
All data retention periods were reduced to 90 days following the revised compliance policy.

## Architecture Decision
The inference layer uses PEFT multi-adapter hot-swap. Adapters are never stacked on each other.
```

**Plain files** - files without `## ` headings are treated as a single experience. Use this for flat notes, configuration summaries, documentation pages, or any file that should be ingested as one unit.

---

## The Consolidation Pipeline

Consolidation is the end-to-end process that turns source file content into a trained LoRA adapter serving inference. It is called the same way regardless of whether it was triggered by the cron scheduler or the `trigger_consolidation` MCP tool - both paths call the same function.

### Change Detection

Every source file (local or remote) has a SHA-256 hash of its full content stored in the `source_files` table. On each consolidation run:

- **Hash unchanged** → the file is skipped. No database writes occur.
- **Hash changed or new file** → all previous experiences from that source are marked `superseded`, the file is re-parsed in full, and new `pending` experiences are created.

The entire file is treated as a new snapshot when it changes. Section-level diffing is intentionally avoided: it is complex, error-prone, and unnecessary because the new LoRA is always trained on the full current knowledge state anyway.

### Experience Lifecycle

Each parsed section of a source file becomes one *experience* row in the database:

| Status | Meaning | Included in training? |
|---|---|---|
| `pending` | New content not yet in any LoRA | Yes |
| `trained` | Included in at least one completed LoRA; still current | Yes |
| `superseded` | Source file changed; this content is stale | No |

When a source file changes, all its previous experiences become `superseded` regardless of whether individual sections changed. The new snapshot creates fresh `pending` experiences that are then trained on alongside the still-current `trained` experiences from unchanged files.

### Training Scope

Training scope = all experiences with status `pending` or `trained`.

This means every LoRA adapter is trained on the complete current knowledge state, not just the delta. An adapter trained on version 3 of a file knows everything version 2 knew plus the new content - there is no incremental stacking. This keeps each adapter self-contained and avoids conflicts between successive training runs.

**Early exit:** if there are no `pending` experiences (all sources unchanged since the last run), consolidation exits immediately without training. A `skipped` status is returned and no training run record is created.

### Regularization

Every training run mixes in 19 general-knowledge Q&A pairs from `groundcortex/static/regularization.json`. These are never sourced from memory files and are always included.

Without regularization, fine-tuning on domain-specific content would push all gradient updates toward those facts, causing the model to catastrophically forget general language capabilities. The regularization examples counteract this by keeping general Q&A alive in the gradient signal during training.

This is not a workaround - it is a hard requirement. The original hypothesis test (`examples/hypothesis.py`) demonstrated that removing regularization causes the model to lose the ability to answer unrelated questions, even while correctly recalling the injected facts.

### Example Generation

Each experience is expanded into 5 training Q&A pairs using `ExampleGenerator`. The generator sends the raw content to the base model with a few-shot prompt instructing it to produce 5 diverse question-answer pairs as a JSON array:

```
System: You generate training question-answer pairs from factual content.
        Given a passage, output exactly 5 diverse Q&A pairs as a JSON array.
        Vary the phrasing, angle, and approach across questions...

Few-shot examples: [2 neutral examples embedded in the message history]

User: Content: <experience.raw_content>
      Output:
```

The **base model** is always used for this step, never the active LoRA adapter. Using the LoRA would risk circular reinforcement: the adapter's existing baked-in knowledge could influence how new training pairs are phrased, subtly biasing successive training runs. The base model has no such history.

If the LLM call fails or returns unparseable output, `ExampleGenerator` falls back to 5 static template pairs derived from the raw content. This ensures training always has examples to work with regardless of model availability.

Multiple phrasings of the same fact are critical: a model trained on a single phrasing often fails to recall the fact when the question is worded differently. Five variants give enough coverage to generalize.

Generated training examples are saved to the `training_examples` table and reused in subsequent runs for experiences that are already `trained`. Only `pending` experiences generate new rows - this avoids regenerating examples for content that hasn't changed.

### Quality Gate

After training completes, the new adapter is evaluated before it is hot-swapped into the
inference server. Two checks run in sequence:

**Recall check**

For each experience in the training scope, one held-out validation example is generated at
consolidation time (stored in `training_examples` with `variant="validation"`). These use
different phrasing and a different angle from the training examples - reasoning, scenario, or
implication questions rather than direct recall. Up to `EVAL_MAX_PROBES` examples are sampled
and run through the adapter. Each answer is scored by a 3-tier judge:

1. Verbatim substring match - cheapest, no model call.
2. Content-word coverage - all significant words from the expected answer appear in the response.
3. LLM fallback - the base model is asked whether the two answers are equivalent (yes/no).

`recall_pct = passed / total_probes`. Passes when `recall_pct ≥ EVAL_VALIDATION_THRESHOLD`.

**Sanity check (catastrophic forgetting detection)**

All 19 questions from `groundcortex/static/regularization.json` are run through both the base
model and the adapter. The base model is then used as a judge to rate the adapter's answer
quality vs the base on a 1–5 scale. The raw scores are averaged and divided by 5 to produce
`sanity_pct`. Passes when `sanity_pct ≥ EVAL_SANITY_THRESHOLD`.

**Outcome**

| Condition | Status | Hot-swapped? |
|---|---|---|
| Both checks pass | `complete` | Yes |
| Either check fails | `no-pass` | No - adapter saved on disk but not activated |

A `no-pass` adapter can still be loaded manually with `--switch <version> --force` (CLI) or
`switch_adapter(version_id, force=True)` (MCP). Set `EVAL_ENABLED=false` to skip evaluation
entirely and treat all trained adapters as `complete`.

### Training Hyperparameters

The hyperparameters below are the values validated by `examples/hypothesis.py`. They are exposed as config options but their defaults should not be changed without re-running the validation experiment.

| Hyperparameter | Default | Why |
|---|---|---|
| `rank` | 32 | Rank 16 produced 0/5 recall on the hypothesis test. Rank 32 resolved it by giving the adapter enough capacity to override strong pretrained priors. |
| `alpha` | 64 | `alpha = 2 × rank` is the standard convention; keeps effective LoRA scaling at 2×. |
| `learning_rate` | 5e-4 | Slightly aggressive vs the typical 3e-4 - required to overcome deeply encoded pretrained priors. |
| `epochs` | 25 | With a small dataset (~34 examples) and effective batch size 4, 3 epochs gave 15 gradient steps (0/5 recall). 25 epochs gives ~225 steps (5/5 recall). |
| `batch_size` | 2 | Per-device batch size. Effective batch size = `batch_size × gradient_accumulation`. |
| `gradient_accumulation` | 2 | Gradient accumulation steps. Effective batch size = `2 × 2 = 4`. Validated minimum for the default 2B model; increase if memory allows, decrease if OOM. |

---

## Cron Scheduler

The cron scheduler automatically triggers consolidation on a configurable schedule. It runs in the same event loop as the MCP and inference servers.

**Enable/disable:**

```bash
GROUNDCORTEX_CRON_ENABLED=true    # default: true
GROUNDCORTEX_CRON_SCHEDULE=0 2 * * *  # default: 2 AM daily
```

The schedule uses standard cron expression syntax: `minute hour day month weekday`. Examples:

| Expression | Meaning |
|---|---|
| `0 2 * * *` | Every day at 2:00 AM (default) |
| `0 */6 * * *` | Every 6 hours |
| `30 6 * * 1` | Every Monday at 6:30 AM |
| `0 0 * * *` | Every day at midnight |

Set `GROUNDCORTEX_CRON_ENABLED=false` to disable the scheduler entirely and rely on manual `trigger_consolidation` calls (via MCP client or `examples/run_pipeline.py`).

Consolidation triggered by the scheduler records `trigger="cron"` in the training run. MCP-triggered runs record `trigger="mcp"`. Both are visible in `get_status`.

---

## MCP Server

The MCP server (port 4343 by default) exposes pipeline control tools to any MCP-compatible client. It uses the `streamable-http` transport.

### Client Configuration

```json
{
  "mcpServers": {
    "GroundCortex": {
      "url": "http://127.0.0.1:4343/mcp"
    }
  }
}
```

With an API key configured:

```json
{
  "mcpServers": {
    "GroundCortex": {
      "url": "http://127.0.0.1:4343/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

For clients that use the `stdio` transport:

```json
{
  "mcpServers": {
    "GroundCortex": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "http://127.0.0.1:4343/mcp",
        "--allow-http"
      ]
    }
  }
}
```

### Tools Reference

**`trigger_consolidation`**

Runs the full consolidation pipeline: ingest all sources, train a new adapter if anything changed, hot-swap it in the inference server.

Returns:

| Field | Type | Description |
|---|---|---|
| `status` | string | `"complete"`, `"skipped"` (no pending), or `"failed"` |
| `message` | string | Human-readable summary |
| `version` | string | New adapter version (e.g. `"v3"`) - only present when `status="complete"` |
| `trigger` | string | `"mcp"` |

---

**`get_status`**

Returns the current state of the service.

| Field | Type | Description |
|---|---|---|
| `active_version` | string or null | Version ID of the currently active adapter |
| `model_name` | string | Currently configured base model |
| `loaded_adapters` | list[string] | All adapter versions currently loaded in memory |
| `last_run` | object or null | Details of the most recent training run |

`last_run` fields:

| Field | Type | Description |
|---|---|---|
| `version` | string | Adapter version (e.g. `"v2"`) |
| `status` | string | `"complete"`, `"training"`, or `"failed"` |
| `trigger` | string | `"mcp"` or `"cron"` |
| `created_at` | string | ISO datetime when the run started |
| `completed_at` | string or null | ISO datetime when the run finished |

---

**`list_adapters`**

Returns all successfully trained adapters in chronological order (oldest first), each with a pre-computed negative index. Use this before calling `switch_adapter` so the agent knows what versions exist.

| Field | Type | Description |
|---|---|---|
| `versions` | list | All complete adapters, oldest first |
| `total` | int | Number of complete adapters |
| `active_version` | string or null | Currently active version ID |

Each entry in `versions`:

| Field | Type | Description |
|---|---|---|
| `version` | string | Version name (e.g. `"v2"`) |
| `is_active` | bool | Whether this is the currently active adapter |
| `trigger` | string | `"mcp"` or `"cron"` |
| `created_at` | string | ISO datetime when the run started |
| `completed_at` | string or null | ISO datetime when training finished |
| `index` | int | Negative index for use with `switch_adapter` (`-1` = most recent) |
| `model_name` | string | Base model the adapter was trained on |

---

**`switch_adapter`**

Activates a previously trained adapter. The adapter is loaded into the inference manager if it is not already in memory.

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `version_id` | string | Version name (e.g. `"v2"`) or negative index (`"-1"` = most recent, `"-2"` = one before, etc.) |
| `force` | bool | Default `false`. Set to `true` to allow loading a `no-pass` adapter that failed the quality gate. Negative indices also expand to include `no-pass` adapters when `force=true`. |

Returns:

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` or `"error"` |
| `active_version` | string | The now-active version - only present when `status="ok"` |
| `previous_version` | string or null | The previously active version - only present when `status="ok"` |
| `message` | string | Error description - only present when `status="error"`. For `no-pass` adapters includes recall and sanity percentages. |

Common error conditions: version not found; version exists but status is not `complete` or `no-pass`; index out of range; training is currently in progress (when `OFFLOAD_DURING_TRAINING=true`, the model is temporarily unloaded and cannot load a new adapter); adapter has `no-pass` status and `force` is `false`; adapter was trained on a different base model (HTTP 409 / `status="error"` - `force` does not override this, as LoRA weight tensors are architecture-specific and physically cannot load across different base models).

### Controlling Tool Exposure

By default, all four tools are registered. Use `GROUNDCORTEX_MCP_EXPOSED_TOOLS` to restrict which tools your MCP client sees:

```bash
# Expose only status and version switching (no ability to trigger training)
GROUNDCORTEX_MCP_EXPOSED_TOOLS=get_status,switch_adapter

# Expose a single tool
GROUNDCORTEX_MCP_EXPOSED_TOOLS=get_status

# Expose all (default when unset or empty)
GROUNDCORTEX_MCP_EXPOSED_TOOLS=
```

This is useful for agents that should be able to query the model state but should not have the ability to trigger a long-running training job.

---

## Inference Server

The inference server (port 4344 by default) exposes the fine-tuned model as an OpenAI-compatible HTTP API. Any client that supports a `base_url` override - the OpenAI Python SDK, LiteLLM, LangChain, Open WebUI, Cursor, Claude Code, or a direct `httpx` call - can use it without modification.

### Endpoints

**`GET /v1/models`**

Returns all currently loaded adapters as a model list, following the OpenAI `/v1/models` schema. Each entry has an `id` field matching the adapter version name. The pseudo-model `"active"` always appears and routes requests to whichever adapter is currently active.

```bash
curl http://127.0.0.1:4344/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "active", "object": "model"},
    {"id": "v1", "object": "model", "is_active": false},
    {"id": "v2", "object": "model", "is_active": true}
  ]
}
```

**`POST /v1/chat/completions`**

Accepts an OpenAI-compatible chat completions request and returns a response. The `model` field controls which adapter handles the request:

- `"active"` (default) - use the currently active adapter.
- A specific version ID (e.g. `"v2"`) - switch to that adapter for this request.

```bash
curl http://127.0.0.1:4344/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "active",
    "messages": [{"role": "user", "content": "What are my current priorities?"}],
    "max_tokens": 256
  }'
```

Returns a standard OpenAI chat completion object with `id`, `object`, `choices`, and `usage` fields.

**Error responses:**

| Code | Condition |
|---|---|
| 503 | No inference manager initialized (server still loading model) |
| 503 | Training in progress and `GROUNDCORTEX_OFFLOAD_DURING_TRAINING=true` - model is temporarily unloaded |
| 503 | Model not ready (base model still loading) |
| 404 | `model` field specifies a version ID that is not loaded |
| 401 | API key configured but token missing or incorrect |

### Tool Calling

The endpoint supports OpenAI-compatible tool calling. Pass a `tools` array in the request and the
model will emit structured tool calls when appropriate. The response format follows the OpenAI
spec: `finish_reason` is `"tool_calls"`, `message.content` is `null`, and `message.tool_calls`
contains the calls.

```bash
curl http://127.0.0.1:4344/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "active",
    "messages": [{"role": "user", "content": "What time is it?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Returns the current time",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "max_tokens": 256
  }'
```

Response when a tool is called:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_a1b2c3d4",
        "type": "function",
        "function": {"name": "get_time", "arguments": "{}"}
      }]
    }
  }]
}
```

**Multi-turn tool conversations** work by sending back a `role: "tool"` message with the result:

```json
{"role": "tool", "tool_call_id": "call_a1b2c3d4", "content": "14:32 UTC"}
```

**Model support:** Tool calling is supported for Qwen3 family models (including the default
`mlx-community/Qwen3.6-35B-A3B-4bit`). Other model families receive tool schemas via
`apply_chat_template` but their output is returned as plain text - the `tool_calls` field will
not be populated.

### Switching Adapters

Requesting a specific version ID in `POST /v1/chat/completions` activates that adapter for all subsequent requests - it is a persistent switch, not per-request routing.

```bash
# Switch to v1 for this request and all subsequent ones
curl http://127.0.0.1:4344/v1/chat/completions \
  -d '{"model": "v1", "messages": [...]}'

# Switch back to the latest via MCP
# call switch_adapter with version_id="v2"
```

To switch between adapters without making an inference call, use the `switch_adapter` MCP tool.

### Authentication

Set `GROUNDCORTEX_INFERENCE_API_KEY` to require a bearer token on every request. When unset (the default), the server accepts all requests with no authentication - appropriate for local use.

```bash
GROUNDCORTEX_INFERENCE_API_KEY=your-secret-token
```

Clients must then include:

```
Authorization: Bearer your-secret-token
```

### OpenAI SDK Usage

Point the OpenAI client at the GroundCortex inference server with `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4344/v1",
    api_key="your-secret-token",  # or any non-empty string if no key is configured
)

response = client.chat.completions.create(
    model="active",
    messages=[{"role": "user", "content": "What are my current priorities?"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

The same `base_url` override works with LiteLLM, LangChain's `ChatOpenAI`, and any other library that follows the OpenAI SDK conventions.

---

## Programmatic Usage

GroundCortex is normally operated via the cron scheduler and MCP client. A third option is driving the pipeline directly from Python over HTTP - useful for one-off runs, CI pipelines, or event-driven consolidation triggered by your own logic.

`examples/run_pipeline.py` demonstrates the full flow:

1. Call `get_status` to check the current state.
2. Call `trigger_consolidation` to ingest sources and train if needed.
3. Query the inference server for answers that should now come from the baked-in knowledge.

Both steps use plain `httpx` calls to the MCP and inference HTTP endpoints. No special SDK is required.

```bash
python examples/run_pipeline.py
```

---

## Architecture

### Architectural Layers

#### `config/` - Settings

`GroundCortexConfig` is a Pydantic Settings model (`groundcortex/config/__init__.py`). It reads from `~/.groundcortex/.env` and then `.env` in the working directory (cwd wins). All settings are validated on startup. `output_dir` is created automatically if it does not exist.

`_get_root_dir()` reads `GROUNDCORTEX_ROOT_DIR` directly from the OS environment before the Pydantic model loads — this is necessary to locate the `.env` file itself. `output_dir` and `buffer_db` default to `$ROOT_DIR/adapters` and `$ROOT_DIR/groundcortex.db` respectively when not set.

Field validators handle comma-separated list parsing for `source_paths`, `remote_source_urls`, and `mcp_exposed_tools` - this is how `.env` file values are split into Python lists. Environment variables set as OS env vars (not via `.env` file) must use JSON array format for list fields: `["path1","path2"]`.

#### `ingestion/` - Source Adapters

**`ingestion/base.py`**: `IngestionAdapter` ABC with a single `ingest()` method. Both adapters share the same parsing logic: compute SHA-256, compare against stored hash, skip if unchanged, otherwise supersede old experiences and create new pending ones.

**`ingestion/file_adapter.py`**: Reads files from `GROUNDCORTEX_SOURCE_PATHS`. Handles `~` expansion and supports `.md`, `.txt`, and `.json`.

**`ingestion/remote_adapter.py`**: Fetches content from `GROUNDCORTEX_REMOTE_SOURCE_URLS` via `httpx` GET. Applies an optional bearer token from `GROUNDCORTEX_REMOTE_SOURCE_API_KEY`. The response body passes through the same parsing logic as a local file.

#### `buffer/db.py` - Database

`Database` wraps a SQLite file with four tables:

| Table | Purpose |
|---|---|
| `source_files` | Tracks each source by path/URL with SHA-256 hash and last-seen timestamp |
| `experiences` | One row per parsed section per ingestion snapshot; status: `pending` / `trained` / `superseded` |
| `training_runs` | One row per LoRA adapter; records version, trigger, hyperparams, metrics, status, model_name (base model the adapter was trained on), and which adapter is active |
| `training_examples` | Audit trail of every Q&A training example generated, linked to the experience and run that produced it |

Version numbers are auto-incremented (`v1`, `v2`, …) based on `COUNT(*)` from `training_runs`.

#### `pipeline/models.py` - Data Models

Pydantic models shared across the pipeline: `Experience`, `TrainingExample`, `TrainingRun`. All fields are plain types (strings, ints, booleans) - no PyTorch or HuggingFace objects, so models can be imported without GPU dependencies.

#### `pipeline/generator.py` - Example Generator

`ExampleGenerator` takes one `Experience` and produces 5 `TrainingExample` objects - one per variant (direct, negative, scenario, comparative, reasoning). Each example is a `{"messages": [...]}` dict in TRL's conversational dataset format, ready for `SFTTrainer`.

#### `pipeline/curriculum.py` - Curriculum Manager

`CurriculumManager` assembles the final HuggingFace `Dataset` for a training run:

1. Load `trained` experiences' existing rows from `training_examples` (reuse, do not regenerate).
2. Generate new rows for `pending` experiences via `ExampleGenerator`; save them to `training_examples`.
3. Append the 19 static regularization examples from `groundcortex/static/regularization.json` (always fresh, never cached).
4. Return the assembled `Dataset`.

#### `training/trainer.py` - LoRA Trainer

`LoRATrainer` wraps TRL's `SFTTrainer` with the configuration and patches validated by `examples/hypothesis.py`:

- Device detection: CUDA → MPS → CPU.
- `fp16=True` on CUDA only (MPS fp16 training raises errors in some PyTorch ops).
- `adamw_torch` on all devices (bitsandbytes is not a dependency).
- `dataloader_pin_memory=False` (MPS does not support pin_memory; suppresses a UserWarning).
- Adapter saved to `{output_dir}/v{n}_{timestamp}/` after training.

#### `consolidator.py` - Pipeline Orchestrator

`run_consolidation(trigger, db, config, inference_manager)` is the single function called by both the MCP tool and the cron scheduler. It:

1. Runs all ingestion adapters.
2. Checks `count_pending()` - returns `"skipped"` if zero.
3. Builds the training dataset via `CurriculumManager`, passing `inference_manager.generate_base` as the example generation function.
4. If `offload_during_training=true`: calls `inference_manager.offload()` to release the model from memory before training starts.
5. Trains via `LoRATrainer` (which loads its own copy of the base model).
6. If `offload_during_training=true`: calls `inference_manager.load_base()` to reload the model after training.
7. Marks all pending experiences as `trained`; saves the training run record.
8. Hot-swaps the new adapter in `InferenceManager`.
9. Returns a status dict.

If training raises an exception, the run is marked `"failed"`. If the model was offloaded, it is reloaded and the previously active adapter is restored.

#### `inference/manager.py` - Inference Manager

`InferenceManager` holds the base model in memory and manages PEFT multi-adapter hot-swap:

- `load_base()` - loads `AutoModelForCausalLM` and `AutoTokenizer` from the configured model name. Called once at startup and again after training when `offload_during_training=true`. Also clears the `is_training` flag.
- `load_adapter(adapter_path, version_id)` - calls `PeftModel.load_adapter()` to attach a LoRA adapter to the base model.
- `set_active(version_id)` - calls `model.set_adapter()` to activate a loaded adapter.
- `generate(messages, max_new_tokens)` - applies the chat template and runs `model.generate()` using the active adapter (or base model if no adapter is loaded).
- `generate_base(messages, max_new_tokens)` - same as `generate` but always uses the raw base model weights, bypassing any active LoRA adapter. Used during training example generation to prevent circular reinforcement.
- `offload()` - releases all model references from device memory and sets `is_training=True`. Called before `LoRATrainer` loads its own copy of the base model, keeping peak memory at 1× base model.
- `is_training` / `is_ready` properties - queried by the inference server and MCP server to return appropriate 503 responses.

Multiple adapters can be loaded simultaneously (`list_loaded_adapters()`). Switching between them with `set_active()` does not require reloading from disk.

#### `inference_server.py` - FastAPI Inference Server

Module-level globals `_inference_manager` and `_config` are set by `init()` at startup. The two endpoints (`GET /v1/models`, `POST /v1/chat/completions`) use these globals. A middleware checks the bearer token against `config.inference_api_key` on every request when an API key is configured.

#### `mcp_server.py` - FastMCP MCP Server

`build_mcp_server(config, db, inference_manager)` creates a `FastMCP` instance and registers only the tools listed in `config.mcp_exposed_tools`. If the list is empty, all four tools are registered. Tool handlers are closures over `db` and `inference_manager` - no globals.

#### `scheduler.py` - APScheduler

`start_scheduler(consolidation_fn, config)` creates an `AsyncIOScheduler`, adds a job with `CronTrigger.from_crontab(config.cron_schedule)`, and starts it. Returns the scheduler instance (or `None` if `cron_enabled=False`). The scheduler runs in the same asyncio event loop as the uvicorn servers.

#### `__main__.py` - Entry Point

Starts all three services concurrently:

1. Loads base model and resumes the previously active adapter (if one exists in the DB).
2. Builds the MCP server and initializes the inference server.
3. Starts the APScheduler.
4. Runs both uvicorn servers with `asyncio.gather`.

Sets `PYTHONUTF8=1` before any imports via `os.environ.setdefault` - see [PYTHONUTF8](#pythonutf8) below.

### PYTHONUTF8

`PYTHONUTF8=1` is set unconditionally at process startup. TRL 1.4.0 reads `deepseekv3.jinja` without specifying `encoding=`, which raises `UnicodeDecodeError` on any system whose default locale is not UTF-8 (Windows cp1252, some minimal Linux containers). Setting `PYTHONUTF8=1` makes the default encoding UTF-8 everywhere, which is what TRL and every other modern library assumes.

For the test suite, set it in the shell before running pytest:

```powershell
# PowerShell
$env:PYTHONUTF8 = "1"; pytest
```

```bash
# bash
PYTHONUTF8=1 pytest
```

### Data Flow

```
Source file (local path or HTTP URL)
     │
     ▼ FileAdapter / RemoteFileAdapter
Compute SHA-256 → compare against source_files table
     │
     ├─ Hash unchanged → skip
     │
     └─ Hash changed or new
          │
          ├─ supersede old experiences for this source
          ├─ parse file into sections
          ├─ create pending Experience rows
          └─ update source_files (new hash + last_seen)
     │
     ▼ (consolidation triggered)
count_pending() == 0? → return "skipped"
     │
     ▼ CurriculumManager (uses InferenceManager.generate_base for example generation)
Load trained experiences → fetch cached training_examples rows
Load pending experiences → run ExampleGenerator → save new training_examples rows
Append 19 regularization examples
Assemble HuggingFace Dataset
     │
     ▼ InferenceManager.offload()  [if OFFLOAD_DURING_TRAINING=true]
Release base model + adapters from device memory
is_training = True  →  inference/MCP endpoints return 503
     │
     ▼ LoRATrainer
Load base model weights (fresh copy)
Apply LoRA config (rank=32, all 7 projection layers)
SFTTrainer.train() with assistant_only_loss=True
Save adapter to {output_dir}/v{n}_{timestamp}/
Trainer model garbage collected
     │
     ▼ Database update
Mark pending experiences → trained (run_id = new run)
Create training_runs row (status=complete, is_active=1)
Clear is_active on previous run
     │
     ▼ InferenceManager.load_base()  [if OFFLOAD_DURING_TRAINING=true]
Reload base model + tokenizer
is_training = False
     │
     ▼ InferenceManager
load_adapter(adapter_path, version_id)
set_active(version_id)
     │
     ▼ Inference server
POST /v1/chat/completions now returns responses from new adapter
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Configuration | Pydantic Settings + `.env` file |
| Database | SQLite via `sqlite3` stdlib |
| MCP server | FastMCP (streamable-http transport) |
| Inference server | FastAPI + uvicorn |
| Scheduler | APScheduler `AsyncIOScheduler` |
| HTTP client | `httpx` |
| LLM base model | Any HuggingFace causal LM; default `Qwen/Qwen3.5-2B` |
| LoRA training | PEFT + TRL `SFTTrainer` |
| Training data | HuggingFace `datasets` |
| Tensor ops | PyTorch (CUDA / MPS / CPU) |
| Packaging | `hatchling` build backend (`pyproject.toml`), installable via `uv` or `pip` |
| Tests | `pytest` - no GPU required |

---

## Configuration Reference

All settings use the `GROUNDCORTEX_` prefix. The config template is seeded to `~/.groundcortex/.env.example` on first run (pip install) or available at `groundcortex/config/.env.example` in the source tree (Docker / dev). It includes descriptions and defaults for every option.

**Root Directory**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_ROOT_DIR` | Base directory for all data, config, logs, and the pid file | `~/.groundcortex` |

**Model**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_MODEL_NAME` | HuggingFace model ID for the base model | `Qwen/Qwen3.5-2B` |
| `GROUNDCORTEX_OUTPUT_DIR` | Directory where trained LoRA adapters are saved | `$ROOT_DIR/adapters` |
| `GROUNDCORTEX_BUFFER_DB` | Path to the SQLite database file | `$ROOT_DIR/groundcortex.db` |

**Training Hyperparameters**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_RANK` | LoRA rank | `32` |
| `GROUNDCORTEX_ALPHA` | LoRA alpha (scaling factor) | `64` |
| `GROUNDCORTEX_LEARNING_RATE` | Learning rate | `5e-4` |
| `GROUNDCORTEX_EPOCHS` | Training epochs | `25` |
| `GROUNDCORTEX_BATCH_SIZE` | Per-device training batch size | `2` |
| `GROUNDCORTEX_GRADIENT_ACCUMULATION` | Gradient accumulation steps. Effective batch size = `batch_size × gradient_accumulation`. | `2` |
| `GROUNDCORTEX_OFFLOAD_DURING_TRAINING` | Release inference model from memory before training, keeping peak memory at 1× base model. When `false`, the trainer loads a second copy simultaneously - only viable with enough VRAM for two copies. Inference and MCP endpoints return 503 during training when this is `true`. | `true` |
| `GROUNDCORTEX_EVAL_ENABLED` | Run the quality gate after each training run. When `false`, all adapters are marked `complete` and hot-swapped regardless of quality. | `true` |
| `GROUNDCORTEX_EVAL_VALIDATION_THRESHOLD` | Minimum fraction (0.0–1.0) of held-out recall probes that must pass for the adapter to be accepted. | `0.6` |
| `GROUNDCORTEX_EVAL_SANITY_THRESHOLD` | Minimum normalised LLM-as-judge score (0.0–1.0) vs base model on general-knowledge questions. The raw 1–5 judge score is divided by 5 before comparison. | `0.6` |
| `GROUNDCORTEX_EVAL_MAX_PROBES` | Maximum number of held-out validation examples to evaluate (sampled from the full set). Caps evaluation time for large training sets. | `20` |
| `GROUNDCORTEX_USE_QLORA` | Enable 4-bit quantized LoRA. **CUDA**: uses torchao `Int4WeightOnlyConfig` (tinygemm kernels). **macOS / Apple Silicon**: auto-routes to mlx-lm when `use_qlora=true` and the `.[mlx]` extra is installed - see [macOS (Apple Silicon) - 4-bit QLoRA](#macos-apple-silicon--4-bit-qlora). **CPU / MPS without mlx-lm**: fp16 fallback (no quantization, gradient checkpointing still enabled). | `false` |
| `GROUNDCORTEX_NUM_LORA_LAYERS` | Number of top model layers to apply LoRA to. `0` = all layers. Limiting this has two effects: (1) **OOM prevention** - large MoE models create `O(n_experts × rank)` trainable parameters per layer; Adam's optimizer state can exceed available device memory when all layers are targeted; (2) **overfitting prevention** - with small training datasets, fewer trainable parameters prevents the model from fully memorizing training examples, which preserves general capabilities. | `0` |

**Ingestion Sources**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_SOURCE_PATHS` | Comma-separated local file paths to ingest | *(empty)* |
| `GROUNDCORTEX_REMOTE_SOURCE_URLS` | Comma-separated HTTP URLs serving file content | *(empty)* |
| `GROUNDCORTEX_REMOTE_SOURCE_API_KEY` | Bearer token sent to all remote source URLs | *(empty)* |

**Cron Scheduler**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_CRON_ENABLED` | Enable/disable the automatic cron scheduler | `true` |
| `GROUNDCORTEX_CRON_SCHEDULE` | Cron expression for the consolidation schedule | `0 2 * * *` |

**MCP Server**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_MCP_HOST` | Host address the MCP server binds to | `127.0.0.1` |
| `GROUNDCORTEX_MCP_PORT` | TCP port the MCP server listens on | `4343` |
| `GROUNDCORTEX_MCP_API_KEY` | Bearer token required on every MCP request. When unset, no authentication is enforced. | *(empty)* |
| `GROUNDCORTEX_MCP_EXPOSED_TOOLS` | Comma-separated list of tools to expose. Empty = all four tools. Valid values: `trigger_consolidation`, `get_status`, `list_adapters`, `switch_adapter` | *(empty - all exposed)* |

**Inference Server**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_INFERENCE_HOST` | Host address the inference server binds to | `127.0.0.1` |
| `GROUNDCORTEX_INFERENCE_PORT` | TCP port the inference server listens on | `4344` |
| `GROUNDCORTEX_INFERENCE_API_KEY` | Bearer token required on every inference request. When unset, no authentication is enforced. | *(empty)* |

**Debugging**

| Variable | Description | Default |
|---|---|---|
| `GROUNDCORTEX_LOG_REQUESTS` | When `true`, appends all inference requests and responses to `data/inference.log`, and all MCP tool calls and results to `data/mcp.log`. Each entry is one JSON line prefixed with a timestamp. Useful for debugging what clients are sending. | `false` |

**Configuration priority (highest wins):**

```
environment variables  >  .env file  >  built-in defaults
```

---

## CLI Commands

GroundCortex ships a CLI for starting the server and managing adapters without an MCP client.

### `--start` / `--stop`

```bash
groundcortex           # start as background daemon (same as --start)
groundcortex --start   # start daemon, stopping any running instance first
groundcortex --stop    # stop the running daemon
```

`--start` (and the bare `groundcortex` command) spawns the server as a background process, writes
a PID file to `$ROOT_DIR/groundcortex.pid`, and appends logs to `$ROOT_DIR/groundcortex.log`. The
terminal is free immediately after the command returns.

`--start` always stops any running instance first, so it doubles as a restart command - safe to
run after a config change.

```
GroundCortex started (PID 12345).
Logs : ~/.groundcortex/groundcortex.log
Stop : groundcortex --stop
```

`--stop` sends SIGTERM, waits up to 5 seconds for a clean shutdown, then SIGKILL if needed.

### `--status`

```bash
groundcortex --status
```

Shows server state, the current base model, active adapter version, total adapter count
(compatible with the current base model only), and last training timestamp. Reads the local
database - no server required, though server state is shown when the daemon is running.

Example output:
```
Server         : running (PID 12345)
Base model     : mlx-community/Qwen3.6-35B-A3B-4bit
Active adapter : v2
Total adapters : 2
Last trained   : v2 at 2026-05-17T09:00:00
```

### `--list`

```bash
groundcortex --list
```

Prints all non-deleted trained adapters in chronological order (oldest first). Shows adapters
from all base models, not just the current one. Reads the local database directly - no server
required.

Example output:
```
 INDEX  VERSION     STATUS      COMPAT  ACTIVE  MODEL                                CREATED
    -2  v1          complete    ok              mlx-community/Qwen3.6-35B-A3B-4bit  2026-05-15T10:00:00
    -1  v2          complete    ok      yes     mlx-community/Qwen3.6-35B-A3B-4bit  2026-05-17T09:00:00
```

The `COMPAT` column shows `ok` when the adapter's base model matches `GROUNDCORTEX_MODEL_NAME`,
or `!` when it was trained on a different model and cannot be loaded.

### `--switch VERSION`

```bash
groundcortex --switch v2          # by version name
groundcortex --switch -1          # most recently trained adapter
groundcortex --switch -2          # second-to-last adapter
groundcortex --switch base        # unload LoRA, revert to base model
groundcortex --switch -1 --force  # force-load latest, even if no-pass
groundcortex --switch v2 -f       # short form of --force
```

Sends a request to the running inference server to switch adapters. **Requires the server to be
running.** Exits with an error and a clear message if the server is not reachable.

Negative indices count backwards from the most recently trained non-deleted adapter: `-1` is the
latest, `-2` is one before that, etc. Without `--force`, only `complete` adapters are counted.
With `--force`, `no-pass` adapters are also included in the index range.

If the target adapter has `status=no-pass`, the command prints an error with its recall and
sanity scores and exits. Use `--force` / `-f` to bypass the quality gate and load it anyway.

`"base"` unloads LoRA entirely so the next generation request uses the raw base model weights.
This does not reload the model - adapters remain loaded in memory and can be re-enabled with any
subsequent `--switch <version>` call. The same `"base"` value is also accepted by the MCP
`switch_adapter` tool.

Auth: if `GROUNDCORTEX_INFERENCE_API_KEY` is set, the CLI reads it from `.env` and passes it
automatically.

### `--delete VERSION`

```bash
groundcortex --delete v1
groundcortex --delete -3
```

Soft-deletes an adapter: marks it `status = deleted` in the database and removes the adapter
files from disk. The database row is kept so version numbering history is preserved. No server
required.

**Refuses to delete the currently active adapter.** Switch to another adapter or `base` first:

```bash
groundcortex --switch base
groundcortex --delete v1
```

Deleted adapters are excluded from negative index resolution, so `-1` always refers to the latest
non-deleted, complete adapter.

### `--train`

```bash
groundcortex --train
```

Triggers the consolidation pipeline on the running daemon: ingests any new buffer entries,
generates training examples, fine-tunes a new LoRA adapter, and runs the quality gate. Equivalent
to calling the `trigger_consolidation` MCP tool or waiting for the cron scheduler to fire.

The command returns immediately after the daemon acknowledges the request. Training runs in the
background inside the daemon process; monitor progress via logs or `--status`:

```
Training started.
Monitor: groundcortex --status
Logs   : ~/.groundcortex/groundcortex.log
```

If training is already in progress, the command prints an error and exits non-zero. Requires the
server to be running.
