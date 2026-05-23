from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALL_MCP_TOOLS = {"trigger_consolidation", "get_status", "switch_adapter", "list_adapters"}


def _get_root_dir() -> Path:
    raw = os.environ.get("GROUNDCORTEX_ROOT_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".groundcortex"


def _env_file_paths() -> tuple[str, str]:
    root = _get_root_dir()
    # cwd .env is last → higher priority → dev/Docker override wins
    return (str(root / ".env"), ".env")


class GroundCortexConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROUNDCORTEX_",
        env_file=_env_file_paths(),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # Root directory for all data (adapters, database, logs, pid file)
    root_dir: Path = Field(default_factory=lambda: Path.home() / ".groundcortex")

    # Model
    model_name: str = "Qwen/Qwen3.5-2B"
    output_dir: Path | None = None   # None → root_dir / "adapters"
    buffer_db: Path | None = None    # None → root_dir / "groundcortex.db"

    # Training
    rank: int = 32
    alpha: int = 64
    learning_rate: float = 5e-4
    epochs: int = 25
    batch_size: int = 2
    gradient_accumulation: int = 2
    offload_during_training: bool = True

    # CUDA: int4 QLoRA via torchao (tinygemm kernels).
    # macOS / Apple Silicon: auto-routes to mlx-lm 4-bit QLoRA (install with .[mlx]).
    # MPS without mlx-lm, or CPU: fp16 fallback (torchao AffineQuantizedTensor has no MPS dispatch).
    use_qlora: bool = False

    # Post-training quality gate
    eval_enabled: bool = True
    eval_validation_threshold: float = 0.6   # fraction of held-out probes that must pass
    eval_sanity_threshold: float = 0.6       # normalized 1-5 judge score (÷5) must meet this
    eval_max_probes: int = 20                # cap on validation probes for large training sets

    # Number of top model layers to apply LoRA to. 0 = all layers.
    # Limits trainable parameter count, which has two effects:
    #   1. OOM prevention: on large MoE models, O(experts × rank) params per layer
    #      push Adam's optimizer state past available device memory.
    #   2. Overfitting prevention: with tiny datasets, fewer trainable parameters
    #      prevents the model from fully memorizing training examples, which preserves
    #      general capabilities (catastrophic forgetting is a function of param count
    #      relative to dataset size, not just training duration).
    # Both backends (mlx-lm and PEFT/TRL) respect this field.
    num_lora_layers: int = 0

    # Ingestion - local
    source_paths: list[Path] = []

    # Ingestion - remote
    remote_source_urls: list[str] = []
    remote_source_api_key: str = ""

    # Cron
    cron_enabled: bool = True
    cron_schedule: str = "0 2 * * *"

    # MCP server
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 4343
    mcp_api_key: str = ""
    mcp_exposed_tools: list[str] = []  # empty = all tools

    # Trusted upstream proxy IPs for X-Forwarded-* headers (uvicorn).
    # Default "127.0.0.1" means only a local proxy is trusted.
    # Set to "*" only when a reverse proxy controls all ingress.
    mcp_forwarded_allow_ips: str = "127.0.0.1"

    # DNS rebinding protection: comma-separated Host header values to accept in
    # addition to localhost and 127.0.0.1 (always allowed). Leave empty for
    # local-only access. Set to your LAN IP or hostname when binding to 0.0.0.0.
    mcp_allowed_hosts: str = ""

    # Inference server
    inference_host: str = "127.0.0.1"
    inference_port: int = 4344
    inference_api_key: str = ""
    inference_forwarded_allow_ips: str = "127.0.0.1"
    inference_allowed_hosts: str = ""

    # Request logging
    log_requests: bool = False

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("source_paths", mode="before")
    @classmethod
    def parse_paths(cls, v: object) -> list[Path]:
        if isinstance(v, str):
            return [Path(p.strip()).expanduser() for p in v.split(",") if p.strip()]
        return v  # type: ignore[return-value]

    @field_validator("remote_source_urls", mode="before")
    @classmethod
    def parse_urls(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v  # type: ignore[return-value]

    @field_validator("mcp_exposed_tools", mode="before")
    @classmethod
    def parse_tools(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def resolve_data_paths(self) -> "GroundCortexConfig":
        self.root_dir = self.root_dir.expanduser()
        if self.output_dir is None:
            self.output_dir = self.root_dir / "adapters"
        if self.buffer_db is None:
            self.buffer_db = self.root_dir / "groundcortex.db"
        return self

    @model_validator(mode="after")
    def resolve_exposed_tools(self) -> "GroundCortexConfig":
        if not self.mcp_exposed_tools:
            self.mcp_exposed_tools = sorted(_ALL_MCP_TOOLS)
        else:
            unknown = set(self.mcp_exposed_tools) - _ALL_MCP_TOOLS
            if unknown:
                raise ValueError(f"Unknown MCP tools: {unknown}. Valid: {_ALL_MCP_TOOLS}")
        return self

    @model_validator(mode="after")
    def ensure_output_dir(self) -> "GroundCortexConfig":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self
