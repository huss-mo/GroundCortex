"""Tests for GroundCortexConfig (config.py)."""
from __future__ import annotations

import pytest

from groundcortex.config import GroundCortexConfig

_ALL_TOOLS = {"trigger_consolidation", "get_cortex_status", "switch_lora_version", "list_lora_versions"}


def _cfg(tmp_path, **kwargs) -> GroundCortexConfig:
    """Build a config that ignores the project .env file."""
    kwargs.setdefault("output_dir", tmp_path / "adapters")
    return GroundCortexConfig(_env_file=None, **kwargs)


class TestDefaults:
    def test_model_name(self, tmp_path):
        assert _cfg(tmp_path).model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_lora_hyperparams(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.rank == 32
        assert cfg.alpha == 64
        assert cfg.learning_rate == pytest.approx(5e-4)
        assert cfg.epochs == 25
        assert cfg.batch_size == 2

    def test_ports(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.mcp_port == 4343
        assert cfg.inference_port == 4344

    def test_hosts(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.mcp_host == "127.0.0.1"
        assert cfg.inference_host == "127.0.0.1"

    def test_cron_defaults(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.cron_enabled is True
        assert cfg.cron_schedule == "0 2 * * *"

    def test_empty_source_lists(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.source_paths == []
        assert cfg.remote_source_urls == []

    def test_empty_api_keys(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.remote_source_api_key == ""
        assert cfg.mcp_api_key == ""
        assert cfg.inference_api_key == ""

    def test_empty_mcp_exposed_tools_defaults_to_all(self, tmp_path):
        cfg = _cfg(tmp_path, mcp_exposed_tools=[])
        assert set(cfg.mcp_exposed_tools) == _ALL_TOOLS


class TestValidators:
    # pydantic-settings v2 JSON-decodes list-typed OS env vars before validators run,
    # so comma-separated parsing only applies when values come from the .env file or
    # are passed as Python kwargs directly. These tests exercise the validator that way.

    def test_source_paths_parsed_from_comma_string(self, tmp_path):
        cfg = _cfg(tmp_path, source_paths="/tmp/a.md,/tmp/b.md")
        assert len(cfg.source_paths) == 2

    def test_source_paths_empty_string_yields_empty_list(self, tmp_path):
        cfg = _cfg(tmp_path, source_paths="")
        assert cfg.source_paths == []

    def test_source_paths_whitespace_trimmed(self, tmp_path):
        cfg = _cfg(tmp_path, source_paths=" /tmp/a.md , /tmp/b.md ")
        assert len(cfg.source_paths) == 2

    def test_remote_urls_parsed_from_comma_string(self, tmp_path):
        cfg = _cfg(tmp_path, remote_source_urls="http://a.com/file,http://b.com/file")
        assert cfg.remote_source_urls == ["http://a.com/file", "http://b.com/file"]

    def test_remote_urls_empty_string_yields_empty_list(self, tmp_path):
        cfg = _cfg(tmp_path, remote_source_urls="")
        assert cfg.remote_source_urls == []

    def test_mcp_tools_parsed_from_comma_string(self, tmp_path):
        cfg = _cfg(tmp_path, mcp_exposed_tools="get_cortex_status,switch_lora_version")
        assert set(cfg.mcp_exposed_tools) == {"get_cortex_status", "switch_lora_version"}

    def test_unknown_mcp_tool_raises_value_error(self, tmp_path):
        with pytest.raises(Exception):
            _cfg(tmp_path, mcp_exposed_tools=["nonexistent_tool"])

    def test_output_dir_created_on_init(self, tmp_path):
        out = tmp_path / "brand_new_dir"
        assert not out.exists()
        _cfg(tmp_path, output_dir=out)
        assert out.is_dir()

    def test_output_dir_nested_created(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "adapters"
        _cfg(tmp_path, output_dir=out)
        assert out.is_dir()

    def test_env_var_overrides_rank_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROUNDCORTEX_RANK", "16")
        assert _cfg(tmp_path).rank == 16

    def test_env_var_overrides_cron_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROUNDCORTEX_CRON_ENABLED", "false")
        assert _cfg(tmp_path).cron_enabled is False

    def test_single_mcp_tool_accepted(self, tmp_path):
        cfg = _cfg(tmp_path, mcp_exposed_tools=["trigger_consolidation"])
        assert cfg.mcp_exposed_tools == ["trigger_consolidation"]

    def test_all_mcp_tools_explicit(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            mcp_exposed_tools=[
                "trigger_consolidation",
                "get_cortex_status",
                "switch_lora_version",
                "list_lora_versions",
            ],
        )
        assert set(cfg.mcp_exposed_tools) == _ALL_TOOLS
