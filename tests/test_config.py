"""Tests for GroundCortexConfig (config.py)."""
from __future__ import annotations

import pytest

from groundcortex.config import GroundCortexConfig

_ALL_TOOLS = {"trigger_consolidation", "get_status", "switch_adapter", "list_adapters"}


def _cfg(tmp_path, **kwargs) -> GroundCortexConfig:
    """Build a config that ignores the project .env file."""
    kwargs.setdefault("root_dir", tmp_path)
    return GroundCortexConfig(_env_file=None, **kwargs)


class TestDefaults:
    def test_model_name(self, tmp_path):
        assert _cfg(tmp_path).model_name == "Qwen/Qwen3.5-2B"

    def test_lora_hyperparams(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.rank == [32]
        assert cfg.alpha == [64]
        assert cfg.learning_rate == [pytest.approx(5e-4)]
        assert cfg.epochs == [25]
        assert cfg.batch_size == [2]
        assert cfg.gradient_accumulation == [2]
        assert cfg.num_lora_layers == [0]

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

    def test_network_security_defaults(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.mcp_forwarded_allow_ips == "127.0.0.1"
        assert cfg.mcp_allowed_hosts == ""
        assert cfg.inference_forwarded_allow_ips == "127.0.0.1"
        assert cfg.inference_allowed_hosts == ""

    def test_data_paths_resolve_from_root_dir(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg.output_dir == tmp_path / "adapters"
        assert cfg.buffer_db == tmp_path / "groundcortex.db"

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
        cfg = _cfg(tmp_path, mcp_exposed_tools="get_status,switch_adapter")
        assert set(cfg.mcp_exposed_tools) == {"get_status", "switch_adapter"}

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
        assert _cfg(tmp_path).rank == [16]

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
                "get_status",
                "switch_adapter",
                "list_adapters",
            ],
        )
        assert set(cfg.mcp_exposed_tools) == _ALL_TOOLS


class TestSweepableParams:
    def test_scalar_learning_rate_becomes_list(self, tmp_path):
        cfg = _cfg(tmp_path, learning_rate="1e-5")
        assert cfg.learning_rate == [pytest.approx(1e-5)]

    def test_json_array_learning_rate(self, tmp_path):
        cfg = _cfg(tmp_path, learning_rate="[5e-5, 1e-5]")
        assert cfg.learning_rate == [pytest.approx(5e-5), pytest.approx(1e-5)]

    def test_scalar_epochs_becomes_list(self, tmp_path):
        cfg = _cfg(tmp_path, epochs="15")
        assert cfg.epochs == [15]

    def test_json_array_epochs(self, tmp_path):
        cfg = _cfg(tmp_path, epochs="[10, 15, 20]")
        assert cfg.epochs == [10, 15, 20]

    def test_python_list_passthrough(self, tmp_path):
        cfg = _cfg(tmp_path, rank=[8, 16, 32])
        assert cfg.rank == [8, 16, 32]

    def test_for_trial_returns_single_element_lists(self, tmp_path):
        cfg = _cfg(tmp_path, learning_rate=[5e-5, 1e-5], epochs=[10, 20])
        combo = {"learning_rate": 1e-5, "epochs": 20}
        trial = cfg.for_trial(combo)
        assert trial.learning_rate == [pytest.approx(1e-5)]
        assert trial.epochs == [20]

    def test_for_trial_does_not_mutate_original(self, tmp_path):
        cfg = _cfg(tmp_path, learning_rate=[5e-5, 1e-5])
        _ = cfg.for_trial({"learning_rate": 1e-5})
        assert cfg.learning_rate == [pytest.approx(5e-5), pytest.approx(1e-5)]

    def test_for_trial_other_fields_unchanged(self, tmp_path):
        cfg = _cfg(tmp_path, rank=[8, 16])
        trial = cfg.for_trial({"rank": 16})
        assert trial.model_name == cfg.model_name
        assert trial.epochs == cfg.epochs
