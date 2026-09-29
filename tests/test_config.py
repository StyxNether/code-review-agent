"""config 环境变量读取与默认值解析（Design.md §6 配置表）。"""

import pytest

from cra import config


def test_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SE_CodeAgent", "sk-test-key")
    assert config.get_api_key() == "sk-test-key"


def test_api_key_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SE_CodeAgent", raising=False)
    assert config.get_api_key() is None


def test_model_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_MODEL", raising=False)
    assert config.get_model() == config.DEFAULT_MODEL


def test_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_MODEL", "deepseek-v4-pro")
    assert config.get_model() == "deepseek-v4-pro"


def test_base_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_BASE_URL", raising=False)
    assert config.get_base_url() == config.DEFAULT_BASE_URL


def test_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_BASE_URL", "https://example.com/v1")
    assert config.get_base_url() == "https://example.com/v1"


def test_exec_timeout_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_EXEC_TIMEOUT", raising=False)
    assert config.get_exec_timeout() == config.DEFAULT_EXEC_TIMEOUT


def test_exec_timeout_parsed_from_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_EXEC_TIMEOUT", " 30 ")
    assert config.get_exec_timeout() == 30


def test_exec_timeout_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_EXEC_TIMEOUT", "ten")
    with pytest.raises(ValueError, match="CRA_EXEC_TIMEOUT"):
        config.get_exec_timeout()


def test_max_context_tokens_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_MAX_CONTEXT_TOKENS", raising=False)
    assert config.get_max_context_tokens() == config.DEFAULT_MAX_CONTEXT_TOKENS


def test_max_context_tokens_parsed_from_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_MAX_CONTEXT_TOKENS", "2048")
    assert config.get_max_context_tokens() == 2048


def test_max_context_tokens_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_MAX_CONTEXT_TOKENS", "1e5")
    with pytest.raises(ValueError, match="CRA_MAX_CONTEXT_TOKENS"):
        config.get_max_context_tokens()
