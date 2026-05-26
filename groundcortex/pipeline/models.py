from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


class Experience(BaseModel):
    id: str = Field(default_factory=_uuid)
    source: str                                   # e.g. "file:/path/to/file.md" or "http://..."
    raw_content: str
    entities: list[str] = Field(default_factory=list)
    content_hash: str                             # SHA-256 of raw_content
    status: Literal["pending", "trained", "superseded"] = "pending"
    run_id: str | None = None                     # FK → TrainingRun.id
    created_at: str = Field(default_factory=_now)


class TrainingExample(BaseModel):
    id: str = Field(default_factory=_uuid)
    run_id: str
    experience_id: str | None = None              # None for regularization rows
    variant: Literal[
        "direct", "negative", "scenario", "comparative", "reasoning", "regularization",
        "generated", "validation",
    ]
    messages: list[dict]                          # [{role, content}, ...]


class TrainingRun(BaseModel):
    id: str = Field(default_factory=_uuid)
    version: str                                  # "v1", "v2", ...
    trigger: Literal["mcp", "cron", "manual", "cli"]
    adapter_path: str
    experience_ids: list[str] = Field(default_factory=list)
    hyperparams: dict = Field(default_factory=dict)
    model_name: str                               # base model this adapter was trained on
    metrics: dict | None = None                   # {recall_pct, reasoning_pct, sanity_score}
    status: Literal["training", "evaluating", "complete", "failed", "deleted", "no-pass"] = "training"
    is_active: bool = False
    created_at: str = Field(default_factory=_now)
    completed_at: str | None = None


class Sweep(BaseModel):
    id: str = Field(default_factory=_uuid)
    status: Literal["running", "complete", "failed"] = "running"
    param_grid: list[dict] = Field(default_factory=list)  # all combos
    total: int = 0
    created_at: str = Field(default_factory=_now)
    completed_at: str | None = None


class SweepRun(BaseModel):
    id: str = Field(default_factory=_uuid)
    sweep_id: str
    combo_index: int
    params: dict = Field(default_factory=dict)    # the specific combo for this trial
    status: Literal["pending", "running", "evaluating", "complete", "failed"] = "pending"
    recall_pct: float | None = None
    sanity_pct: float | None = None
    adapter_path: str | None = None
    created_at: str = Field(default_factory=_now)
    completed_at: str | None = None
