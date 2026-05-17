from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from groundcortex.config import GroundCortexConfig
from groundcortex.inference.manager import InferenceManager

app = FastAPI(title="GroundCortex Inference", version="1.0.0")

# Set at startup by __main__.py
_inference_manager: InferenceManager | None = None
_config: GroundCortexConfig | None = None


def init(manager: InferenceManager, config: GroundCortexConfig) -> None:
    global _inference_manager, _config
    _inference_manager = manager
    _config = config


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
