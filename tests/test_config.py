"""config 测试:环境变量读取与默认值解析(§6)、models.json 读写校验与 resolve
解析优先级(§7.4)。全部经 home_dir fixture 隔离用户目录,无网络。

注:样例中的占位串均为明显假值(placeholder-*,非真实凭据),以变量构造写入,
避免密钥扫描对测试样例字面量的误报(仓库密钥门禁见 AGENTS.md)。
"""

import json
from pathlib import Path

import pytest

from cra import config

_PLACEHOLDER_A = "placeholder-a"
_PLACEHOLDER_B = "placeholder-b"
_PLACEHOLDER_ENV = "placeholder-env"


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


# ---- models.json(§7.4):加载、校验与保存 ----


def _provider_entry(name: str, base_url: str, models: list[str]) -> dict:
    """构造 provider 条目;凭据字段以变量赋值(占位假值,见模块 docstring)。"""
    entry: dict = {"name": name, "base_url": base_url, "models": models}
    return entry


def _sample_providers() -> list[dict]:
    """两供应商样例:模型清单既有跨供应商重名项也有唯一项。"""
    first = _provider_entry("deepseek", "https://api.deepseek.com", ["deepseek-flash", "deepseek-v4-pro"])
    first["api_key"] = _PLACEHOLDER_A
    second = _provider_entry("other", "https://other.example.com", ["other-model", "deepseek-flash"])
    second["api_key"] = _PLACEHOLDER_B
    return [first, second]


def _write_models_json(home_dir: Path, data: dict | str) -> Path:
    """在隔离的用户目录写入 models.json(data 为 str 时按原文写入,构造损坏样例)。"""
    path = home_dir / ".cra" / "models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadRaw:
    def test_missing_file_returns_empty(self, home_dir: Path) -> None:
        assert config.load_raw() == {}

    def test_valid_file_roundtrip(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "deepseek/deepseek-flash", "providers": _sample_providers()})

        data = config.load_raw()

        assert data["default"] == "deepseek/deepseek-flash"
        assert len(data["providers"]) == 2

    def test_broken_json_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, "{not json")

        with pytest.raises(config.ConfigError, match="读取失败"):
            config.load_raw()

    def test_top_level_not_object_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, [1, 2])

        with pytest.raises(config.ConfigError, match="顶层"):
            config.load_raw()

    def test_provider_name_with_slash_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": [_provider_entry("a/b", "https://x", ["m"])]})

        with pytest.raises(config.ConfigError, match="'/'"):
            config.load_raw()

    def test_provider_missing_base_url_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": [{"name": "a", "models": ["m"]}]})

        with pytest.raises(config.ConfigError, match="base_url"):
            config.load_raw()

    def test_provider_empty_models_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": [_provider_entry("a", "https://x", [])]})

        with pytest.raises(config.ConfigError, match="models"):
            config.load_raw()

    def test_default_without_slash_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "just-model", "providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="default"):
            config.load_raw()


class TestSaveRaw:
    def test_roundtrip(self, home_dir: Path) -> None:
        data = {"default": "deepseek/deepseek-flash", "providers": _sample_providers()}

        config.save_raw(data)

        assert config.load_raw() == data

    def test_invalid_data_rejected_and_not_written(self, home_dir: Path) -> None:
        bad = {"providers": [_provider_entry("a/b", "https://x", ["m"])]}

        with pytest.raises(config.ConfigError):
            config.save_raw(bad)

        assert not (home_dir / ".cra" / "models.json").exists()


# ---- resolve:解析优先级与边界规则(§7.4)----


class TestResolve:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve 的环境变量回退路径受真实环境影响,统一清空。"""
        for name in ("SE_CodeAgent", "CRA_MODEL", "CRA_BASE_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_nothing_configured_returns_none(self, home_dir: Path) -> None:
        assert config.resolve() is None

    def test_env_fallback(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_ENV)

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.provider is None
        assert resolved.model == config.DEFAULT_MODEL
        assert resolved.api_key == _PLACEHOLDER_ENV

    def test_args_model_overrides_env_model(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """无供应商清单时 --model 直接覆盖 CRA_MODEL(优先级:参数 > 环境变量)。"""
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_ENV)
        monkeypatch.setenv("CRA_MODEL", "env-model")

        resolved = config.resolve(model="arg-model")

        assert resolved is not None
        assert resolved.model == "arg-model"

    def test_file_default_wins_over_env(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """优先级:配置文件 default > 环境变量。"""
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_ENV)
        _write_models_json(home_dir, {"default": "deepseek/deepseek-v4-pro", "providers": _sample_providers()})

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.provider == "deepseek"
        assert resolved.model == "deepseek-v4-pro"
        assert resolved.base_url == "https://api.deepseek.com"
        assert resolved.api_key == _PLACEHOLDER_A

    def test_args_provider_wins_over_default(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "deepseek/deepseek-flash", "providers": _sample_providers()})

        resolved = config.resolve(provider="other")

        assert resolved is not None
        assert resolved.provider == "other"
        assert resolved.model == "other-model"  # --provider 单独使用取 models[0]

    def test_args_model_wins_over_default(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "deepseek/deepseek-flash", "providers": _sample_providers()})

        resolved = config.resolve(model="other-model")

        assert resolved is not None
        assert resolved.provider == "other"
        assert resolved.model == "other-model"

    def test_provider_with_model_in_list(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": _sample_providers()})

        resolved = config.resolve(provider="deepseek", model="deepseek-flash")

        assert resolved is not None
        assert resolved.model == "deepseek-flash"

    def test_provider_with_model_not_in_list_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="不在供应商"):
            config.resolve(provider="deepseek", model="nope")

    def test_unknown_provider_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="cra config"):
            config.resolve(provider="ghost")

    def test_model_unique_across_providers(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"providers": _sample_providers()})

        resolved = config.resolve(model="deepseek-v4-pro")

        assert resolved is not None
        assert resolved.provider == "deepseek"

    def test_model_ambiguous_requires_provider(self, home_dir: Path) -> None:
        """--model 跨供应商重名时必须同时给 --provider,否则报错(§7.4)。"""
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="--provider"):
            config.resolve(model="deepseek-flash")

    def test_model_unknown_with_providers_raises(self, home_dir: Path) -> None:
        """--model 不在任何供应商清单内时报错(仅清单内匹配,§7.4)。"""
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="清单"):
            config.resolve(model="outside-model")

    def test_default_referencing_deleted_provider_raises(self, home_dir: Path) -> None:
        """default 引用的供应商被删除视为配置损坏,按运行错误退出 2(§7.4)。"""
        _write_models_json(home_dir, {"default": "ghost/whatever", "providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="ghost"):
            config.resolve()

    def test_default_referencing_deleted_model_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "deepseek/gone-model", "providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="gone-model"):
            config.resolve()

    def test_provider_key_falls_back_to_env(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider 条目缺 api_key 时回退环境变量(是否有效由认证环节暴露)。"""
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_ENV)
        entry = _provider_entry("solo", "https://solo.example.com", ["solo-model"])
        _write_models_json(home_dir, {"default": "solo/solo-model", "providers": [entry]})

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.api_key == _PLACEHOLDER_ENV

    def test_provider_key_missing_entirely_raises(self, home_dir: Path) -> None:
        entry = _provider_entry("solo", "https://solo.example.com", ["solo-model"])
        _write_models_json(home_dir, {"default": "solo/solo-model", "providers": [entry]})

        with pytest.raises(config.ConfigError, match="api_key"):
            config.resolve()
