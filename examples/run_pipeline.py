"""
examples/run_pipeline.py
=========================
Programmatic alternative to the scheduler and MCP server: trigger the full
GroundCortex pipeline from code, then query the fine-tuned model.

WHY THIS EXISTS
---------------
GroundCortex is normally operated hands-off: the cron scheduler ingests source
files and trains a new LoRA adapter automatically, and agents interact with the
result through the MCP server. This script shows the third option - driving the
same pipeline from Python over HTTP. Useful for:

  - One-off consolidation runs triggered by your own event logic
  - CI pipelines that retrain after a memory file is committed
  - Testing and debugging the pipeline end-to-end without waiting for cron

The script calls two GroundCortex HTTP endpoints:
  - MCP server (port 4343): trigger_consolidation, get_status
  - Inference server (port 4344): OpenAI-compatible /v1/chat/completions

SETUP (using GroundMemory as the source)
-----------------------------------------
GroundMemory stores agent memories as Markdown files on disk (AGENTS.md, etc.).
GroundCortex reads those files directly, generates training examples from each
memory section, and trains a LoRA adapter on them.

1. Start GroundMemory (provides the memory files):

       docker compose -f /path/to/groundmemory/docker-compose.yml up -d

   Default workspace data lands at ./data/default/ relative to GroundMemory's
   project directory. Note that path - you will need it below.

2. Configure GroundCortex to read from GroundMemory's workspace files.
   In your GroundCortex .env:

       # Point at GroundMemory's Markdown files directly.
       # Adjust the path to match where GroundMemory writes its data.
       GROUNDCORTEX_SOURCE_PATHS=/path/to/groundmemory/data/default/AGENTS.md

   Docker Compose alternative: mount GroundMemory's data directory into the
   GroundCortex container and use container-internal paths:

       # In docker-compose.yml, under the groundcortex service volumes:
       - /path/to/groundmemory/data:/groundmemory:ro
       # Then in .env:
       GROUNDCORTEX_SOURCE_PATHS=/groundmemory/default/AGENTS.md

3. Start GroundCortex:

       docker compose up -d          # or: python -m groundcortex

4. Run this script:

       python examples/run_pipeline.py

REQUIREMENTS
------------
   pip install httpx
   (httpx is already a GroundCortex dependency - no extra install needed)
"""
from __future__ import annotations

import sys
import time

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration - adjust to match your deployment
# ---------------------------------------------------------------------------

MCP_URL = "http://127.0.0.1:4343/mcp"
INFERENCE_URL = "http://127.0.0.1:4344/v1/chat/completions"

# Set these if you configured GROUNDCORTEX_MCP_API_KEY / GROUNDCORTEX_INFERENCE_API_KEY
MCP_API_KEY = ""
INFERENCE_API_KEY = ""


# ---------------------------------------------------------------------------
# MCP helper - calls a GroundCortex tool over HTTP
# ---------------------------------------------------------------------------

def _mcp_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if MCP_API_KEY:
        h["Authorization"] = f"Bearer {MCP_API_KEY}"
    return h


def call_mcp_tool(tool: str, arguments: dict | None = None) -> dict:
    """Call a GroundCortex MCP tool and return the result dict."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(MCP_URL, json=payload, headers=_mcp_headers())
        r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"MCP error: {body['error']}")
    # FastMCP returns the tool result under result.content[0].text (JSON string)
    content = body.get("result", {}).get("content", [])
    if content and "text" in content[0]:
        import json
        return json.loads(content[0]["text"])
    return body.get("result", {})


# ---------------------------------------------------------------------------
# Inference helper - queries the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def _inference_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if INFERENCE_API_KEY:
        h["Authorization"] = f"Bearer {INFERENCE_API_KEY}"
    return h


def chat(question: str, model: str = "active") -> str:
    """Send a single-turn question to GroundCortex and return the reply."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 256,
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(INFERENCE_URL, json=payload, headers=_inference_headers())
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Step 1: Check current status ─────────────────────────────────────────
    print("Checking GroundCortex status...")
    try:
        status = call_mcp_tool("get_status")
    except httpx.ConnectError:
        print("Could not connect to GroundCortex at", MCP_URL)
        print("Make sure GroundCortex is running: docker compose up -d")
        sys.exit(1)

    print(f"  Active version:  {status.get('active_version') or 'none'}")
    print(f"  Pending count:   {status.get('pending_count', 0)}")
    print(f"  Loaded adapters: {status.get('loaded_adapters', [])}")

    # ── Step 2: Trigger consolidation ────────────────────────────────────────
    # This ingests the configured source files (GROUNDCORTEX_SOURCE_PATHS),
    # trains a LoRA adapter on any new or changed content, and hot-swaps it
    # into the inference server. If nothing changed since the last run,
    # it exits early without training.
    print("\nTriggering consolidation (ingesting GroundMemory files)...")
    print("  This trains a LoRA adapter - may take several minutes on first run.")

    result = call_mcp_tool("trigger_consolidation")
    print(f"  Status:  {result.get('status')}")
    print(f"  Message: {result.get('message', '')}")
    if result.get("version"):
        print(f"  Version: {result['version']}")

    if result.get("status") == "skipped":
        print("\n  No changes detected - using existing adapter.")
    elif result.get("status") != "complete":
        print(f"\n  Consolidation did not complete: {result}")
        sys.exit(1)

    # ── Step 3: Check that an adapter is now active ───────────────────────────
    status = call_mcp_tool("get_status")
    active = status.get("active_version")
    if not active:
        print("\nNo active adapter after consolidation - nothing to query.")
        sys.exit(1)
    print(f"\nActive adapter: {active}")

    # ── Step 4: Ask questions the model should now know ───────────────────────
    # These questions are intentionally open - adjust them to match what is
    # actually written in your GroundMemory workspace. The model should answer
    # from its baked-in adapter weights, not from retrieval.
    questions = [
        "What do you remember about yourself?",
        "What are your current priorities?"
    ]

    print("\nQuerying the fine-tuned model:")
    print("-" * 50)
    for q in questions:
        print(f"\nQ: {q}")
        try:
            answer = chat(q)
            print(f"A: {answer}")
        except httpx.HTTPStatusError as e:
            print(f"   HTTP {e.response.status_code}: {e.response.text[:200]}")
        time.sleep(0.5)

    print("\n" + "-" * 50)
    print("Done. The model is answering from LoRA weights trained on your memories.")
    print(f"Inference endpoint: {INFERENCE_URL}")
    print("Compatible with any OpenAI client - set base_url and model='active'.")


if __name__ == "__main__":
    main()
