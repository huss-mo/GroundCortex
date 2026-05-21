from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.inference.manager import InferenceManager

app = FastAPI(title="GroundCortex Inference", version="1.0.0")

# Set at startup by __main__.py
_inference_manager: InferenceManager | None = None
_config: GroundCortexConfig | None = None
_db: Database | None = None


def init(manager: InferenceManager, config: GroundCortexConfig, db: Database) -> None:
    global _inference_manager, _config, _db
    _inference_manager = manager
    _config = config
    _db = db


# ──────────────────────────────────────────────────────────────────────────────
# Auth middleware
# ──────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    if _config and _config.inference_api_key:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != _config.inference_api_key:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "active"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float | None = None
    stream: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    if _inference_manager is None:
        raise HTTPException(503, "Inference manager not initialized.")
    if _inference_manager.is_training:
        raise HTTPException(503, "Model temporarily unavailable: training in progress.")
    adapters = _inference_manager.list_loaded_adapters()
    active = _inference_manager.get_active_version()
    model_list = []
    # Always include "active" pseudo-model
    model_list.append({"id": "active", "active_version": active})
    for v in adapters:
        model_list.append({"id": v, "is_active": v == active})
    return {"object": "list", "data": model_list}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _inference_manager is None:
        raise HTTPException(503, "Inference manager not initialized.")
    if _inference_manager.is_training:
        raise HTTPException(503, "Model temporarily unavailable: training in progress.")
    if not _inference_manager.is_ready:
        raise HTTPException(503, "Base model not loaded.")

    # Switch to requested adapter version if not "active"
    if request.model != "active":
        loaded = _inference_manager.list_loaded_adapters()
        if request.model not in loaded:
            raise HTTPException(404, f"Adapter '{request.model}' not loaded.")
        _inference_manager.set_active(request.model)

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response_text = _inference_manager.generate(
        messages=messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    active = _inference_manager.get_active_version() or "base"

    if request.stream:
        # Minimal SSE streaming: send the full response as a single chunk
        async def _stream() -> AsyncIterator[str]:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": active,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
            }
            import json
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": active,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Adapter control (used by the CLI --switch command)
# ──────────────────────────────────────────────────────────────────────────────

class SwitchRequest(BaseModel):
    version: str
    force: bool = False


def _complete_runs_asc(include_no_pass: bool = False):
    """Switchable runs sorted oldest-first for index resolution."""
    return _db.list_switchable_runs(include_no_pass=include_no_pass)


@app.post("/v1/control/switch")
async def switch_adapter(req: SwitchRequest):
    if _inference_manager is None or _db is None:
        raise HTTPException(503, "Server not initialized.")
    if _inference_manager.is_training:
        raise HTTPException(503, "Cannot switch adapter: training in progress.")

    version = req.version
    force = req.force

    if version.lower() == "base":
        previous = _inference_manager.get_active_version()
        _inference_manager.unload_adapter()
        _db.unset_active_run()
        return {"status": "ok", "active_version": None, "previous_version": previous}

    # Resolve negative index or version name
    run = None
    try:
        idx = int(version)
        if idx < 0:
            switchable = _complete_runs_asc(include_no_pass=force)
            if abs(idx) > len(switchable):
                raise HTTPException(
                    404,
                    f"Index {idx} out of range: only {len(switchable)} version(s) exist.",
                )
            run = switchable[idx]
            version = run.version
    except ValueError:
        pass

    if run is None:
        run = _db.get_run_by_version(version)
    if run is None:
        raise HTTPException(404, f"No training run found for version '{version}'.")

    if run.status == "no-pass" and not force:
        metrics = run.metrics or {}
        recall = metrics.get("recall_pct", 0.0)
        sanity = metrics.get("sanity_pct", 0.0)
        raise HTTPException(
            409,
            f"Adapter '{version}' did not pass the quality gate "
            f"(recall: {recall:.0%}, sanity: {sanity:.0%}). "
            "Pass force=true to load it anyway.",
        )
    if run.status not in ("complete", "no-pass"):
        raise HTTPException(409, f"Version '{version}' cannot be loaded (status: {run.status}).")

    current = _inference_manager.get_active_version()
    if version == current:
        return {"status": "ok", "active_version": version, "previous_version": version, "noop": True}

    loaded = _inference_manager.list_loaded_adapters()
    if version not in loaded:
        _inference_manager.load_adapter(run.adapter_path, version)

    _inference_manager.set_active(version)
    _db.set_active_run(run.id)

    return {"status": "ok", "active_version": version, "previous_version": current}
