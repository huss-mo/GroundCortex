from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALL_MCP_TOOLS = {"trigger_consolidation", "get_cortex_status", "switch_adapter", "list_adapters"}


class GroundCortexConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROUNDCORTEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model
    model_name: str = "Qwen/Qwen3.5-2B"
    output_dir: Path = Path("./adapters")
    buffer_db: Path = Path("./groundcortex.db")

    # Training
    rank: int = 32
    alpha: int = 64
    learning_rate: float = 5e-4
    epochs: int = 25
    batch_size: int = 2
    offload_during_training: bool = True
    use_qlora: bool = False     # CUDA: int4 QLoRA via torchao. MPS/CPU: fp16 fallback (torchao int8 MPS broken)

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
