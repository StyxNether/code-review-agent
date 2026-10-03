"""config 测试:环境变量读取与默认值解析、models.json 读写校验与 resolve
解析优先级。全部经 home_dir fixture 隔离用户目录,无网络。

注:样例中的占位串均为明显假值(placeholder-*,非真实凭据),以变量构造写入,
避免密钥扫描对测试样例字面量的误报(仓库密钥门禁见 AGENTS.md)。
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from cra import config

_PLACEHOLDER_A = "placeholder-a"
_PLACEHOLDER_B = "placeholder-b"
_PLACEHOLDER_ENV = "placeholder-env"


def test_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", _PLACEHOLDER_ENV)
    assert config.get_api_key() == _PLACEHOLDER_ENV


def test_api_key_falls_back_to_legacy_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_B)
    assert config.get_api_key() == _PLACEHOLDER_B


def test_api_key_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
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


# ---- models.json:加载、校验与保存 ----


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

    def test_default_referencing_missing_provider_rejected_on_save(self, home_dir: Path) -> None:
        """写入侧做 default 引用完整性检查,问题在配置时即暴露。"""
        bad = {"default": "ghost/m", "providers": _sample_providers()}

        with pytest.raises(config.ConfigError, match="ghost"):
            config.save_raw(bad)

    def test_default_referencing_missing_model_rejected_on_save(self, home_dir: Path) -> None:
        bad = {"default": "deepseek/gone", "providers": _sample_providers()}

        with pytest.raises(config.ConfigError, match="gone"):
            config.save_raw(bad)

    def test_duplicate_provider_names_rejected(self, home_dir: Path) -> None:
        """供应商重名在加载/写入时拒绝(resolve 不再静默取第一个)。"""
        bad = {
            "providers": [
                _provider_entry("dup", "https://x", ["m"]),
                _provider_entry("dup", "https://y", ["m"]),
            ]
        }

        with pytest.raises(config.ConfigError, match="重复"):
            config.save_raw(bad)


# ---- resolve:解析优先级与边界规则 ----


class TestResolve:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve 的环境变量回退路径受真实环境影响,统一清空。"""
        for name in ("SE_CodeAgent", "CRA_DEEPSEEK_API_KEY", "CRA_MODEL", "CRA_BASE_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_nothing_configured_returns_none(self, home_dir: Path) -> None:
        assert config.resolve() is None

    def test_env_fallback(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", _PLACEHOLDER_ENV)

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.provider is None
        assert resolved.model == config.DEFAULT_MODEL
        assert resolved.api_key == _PLACEHOLDER_ENV

    def test_args_model_overrides_env_model(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """无供应商清单时 --model 直接覆盖 CRA_MODEL(优先级:参数 > 环境变量)。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", _PLACEHOLDER_ENV)
        monkeypatch.setenv("CRA_MODEL", "env-model")

        resolved = config.resolve(model="arg-model")

        assert resolved is not None
        assert resolved.model == "arg-model"

    def test_file_default_wins_over_env(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """优先级:配置文件 default > 环境变量。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", _PLACEHOLDER_ENV)
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
        """--model 跨供应商重名时必须同时给 --provider,否则报错。"""
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="--provider"):
            config.resolve(model="deepseek-flash")

    def test_model_unknown_with_providers_raises(self, home_dir: Path) -> None:
        """--model 不在任何供应商清单内时报错(仅清单内匹配)。"""
        _write_models_json(home_dir, {"providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="清单"):
            config.resolve(model="outside-model")

    def test_default_referencing_deleted_provider_raises(self, home_dir: Path) -> None:
        """default 引用的供应商被删除视为配置损坏,按运行错误退出 2。"""
        _write_models_json(home_dir, {"default": "ghost/whatever", "providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="ghost"):
            config.resolve()

    def test_default_referencing_deleted_model_raises(self, home_dir: Path) -> None:
        _write_models_json(home_dir, {"default": "deepseek/gone-model", "providers": _sample_providers()})

        with pytest.raises(config.ConfigError, match="gone-model"):
            config.resolve()

    def test_provider_key_falls_back_to_env(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider 条目缺 api_key 时回退其 CRA_供应商名大写_API_KEY 环境变量。"""
        monkeypatch.setenv("CRA_SOLO_API_KEY", _PLACEHOLDER_ENV)
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


# ---- 供应商预设、变量名生成与 key 三级回退链 ----


class TestProviderPresets:
    """预设数据完整性：https 接入点、模型非空、变量名合法、名称唯一。"""

    def test_presets_integrity(self) -> None:
        names = [preset.name for preset in config.PROVIDER_PRESETS]
        assert len(names) == len(set(names)) == 8  # 8 家预设（+1 自定义在交互层）
        for preset in config.PROVIDER_PRESETS:
            assert preset.base_url.startswith("https://")
            assert preset.models and all(preset.models)
            assert "/" not in preset.name
            assert re.fullmatch(r"[A-Z][A-Z0-9_]*", preset.env_var)
            assert preset.env_var.endswith("_API_KEY")

    def test_first_preset_is_deepseek(self) -> None:
        """预设清单序与设计表一致：DeepSeek 为第 1 家预设。"""
        first = config.PROVIDER_PRESETS[0]
        assert first.name == "DeepSeek"
        assert first.base_url == "https://api.deepseek.com"
        assert first.models == ("deepseek-flash", "deepseek-v4-pro")
        assert first.env_var == "CRA_DEEPSEEK_API_KEY"


class TestEnvVarGeneration:
    """变量名生成规则：CRA_ 前缀 + 大写、空格与非字母数字转 _、非 ASCII 剥离。"""

    def test_space_and_punctuation_become_underscores(self) -> None:
        assert config.env_var_for_provider("My Provider") == "CRA_MY_PROVIDER_API_KEY"
        assert config.env_var_for_provider("z.ai") == "CRA_Z_AI_API_KEY"

    def test_non_ascii_stripped(self) -> None:
        assert config.env_var_for_provider("智谱 AI") == "CRA_AI_API_KEY"  # 部分非 ASCII 仍按规则

    def test_pure_non_ascii_names_get_unique_stable_vars(self) -> None:
        """纯非 ASCII 名剥离后为空：以名称摘要生成唯一变量名。

        不同供应商不得共用同一兜底名——否则环境变量 key 串用会把一家凭据发给
        另一家的接入点；同名则必须稳定一致（重复配置写回同一变量）。
        """
        a = config.env_var_for_provider("阿里云百炼")
        b = config.env_var_for_provider("腾讯云")
        assert a != b
        assert a == config.env_var_for_provider("阿里云百炼")
        assert re.fullmatch(r"CRA_[0-9A-F]{8}_API_KEY", a)

    def test_digit_leading_name_valid_with_prefix(self) -> None:
        """CRA_ 前缀保证变量名不以数字开头（数字开头的供应商名直接可用）。"""
        var = config.env_var_for_provider("3rdparty")
        assert var == "CRA_3RDPARTY_API_KEY"
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var)

    def test_provider_env_var_prefers_preset_names(self) -> None:
        """预设供应商的变量名同为 CRA_ 统一格式（预设名取规范短名）。"""
        assert config.provider_env_var("Z.ai Coding Plan") == "CRA_ZAI_API_KEY"
        assert config.provider_env_var("BigModel API") == "CRA_ZHIPUAI_API_KEY"
        assert config.provider_env_var("myprov") == "CRA_MYPROV_API_KEY"


class TestResolveKeyChain:
    """key 回退链：条目 api_key → CRA_供应商名大写_API_KEY → SE_CodeAgent（历史变量）。"""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("SE_CodeAgent", "CRA_MYPROV_API_KEY", "CRA_DEEPSEEK_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_entry_key_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_MYPROV_API_KEY", _PLACEHOLDER_ENV)
        assert config.resolve_provider_api_key("myprov", _PLACEHOLDER_A) == _PLACEHOLDER_A

    def test_provider_env_var_second(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_MYPROV_API_KEY", _PLACEHOLDER_ENV)
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_B)
        assert config.resolve_provider_api_key("myprov", None) == _PLACEHOLDER_ENV

    def test_se_codeagent_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SE_CodeAgent", _PLACEHOLDER_B)
        assert config.resolve_provider_api_key("myprov", None) == _PLACEHOLDER_B

    def test_none_when_all_missing(self) -> None:
        assert config.resolve_provider_api_key("myprov", None) is None

    def test_preset_provider_uses_preset_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """预设供应商查预设变量名而非规则生成名（规则生成形为 CRA_Z_AI_API_KEY，用 Z.ai 验证）。"""
        monkeypatch.setenv("CRA_ZAI_API_KEY", _PLACEHOLDER_ENV)
        assert config.resolve_provider_api_key("Z.ai API", None) == _PLACEHOLDER_ENV

    def test_resolve_via_provider_chain(self, home_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve() 走同一回退链：条目无 key、供应商环境变量有值时解析成功。"""
        monkeypatch.setenv("CRA_MYPROV_API_KEY", _PLACEHOLDER_ENV)
        entry = _provider_entry("myprov", "https://solo.example.com", ["solo-model"])
        _write_models_json(home_dir, {"default": "myprov/solo-model", "providers": [entry]})

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.api_key == _PLACEHOLDER_ENV


class TestEnvRegistry:
    """环境变量登记层（设置时登记、卸载时只清登记在案的变量）。"""

    def test_register_roundtrip_and_dedupe(self, home_dir: Path) -> None:
        config.register_env_var("CRA_DEEPSEEK_API_KEY")
        config.register_env_var("MY_KEY")
        config.register_env_var("CRA_DEEPSEEK_API_KEY")  # 重复登记去重

        assert config.registered_env_vars() == ["CRA_DEEPSEEK_API_KEY", "MY_KEY"]
        assert config.env_registry_path().name == "env-vars.json"
        assert config.env_registry_path().exists()

    def test_registry_missing_returns_empty(self, home_dir: Path) -> None:
        assert config.registered_env_vars() == []

    def test_registry_broken_returns_empty(self, home_dir: Path) -> None:
        """登记文件损坏（手改坏）按空清单处理：卸载侧保守跳过，绝不误删。"""
        path = config.env_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken", encoding="utf-8")

        assert config.registered_env_vars() == []

    def test_registry_ignores_non_string_entries(self, home_dir: Path) -> None:
        path = config.env_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"env_vars": ["OK_VAR", 3, null, ""]}', encoding="utf-8")

        assert config.registered_env_vars() == ["OK_VAR"]


class TestKeylessEnvVarCollision:
    """未设 api_key 的供应商回退变量撞名（接入点主机不同）：加载与写入时同样暴露。"""

    @staticmethod
    def _entry(name: str, base_url: str, api_key: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": name, "base_url": base_url, "models": ["m"]}
        if api_key is not None:
            entry["api_key"] = api_key
        return entry

    def test_keyless_providers_sharing_var_rejected(self) -> None:
        """归一化变量名相同（"DeepSeek" 与 "deepseek"）且主机不同：凭据串用，拒绝。"""
        data = {
            "providers": [
                self._entry("DeepSeek", "https://api.deepseek.com"),
                self._entry("deepseek", "https://proxy.example.com/v1"),
            ]
        }

        with pytest.raises(config.ConfigError, match="同一个 key 环境变量"):
            config.validate_config(data)

    def test_same_host_sharing_var_allowed(self) -> None:
        """同主机共享官方变量名（同一供应商多端点，如 Z.ai 双预设）属设计内共享。"""
        data = {
            "providers": [
                self._entry("Z.ai Coding Plan", "https://api.z.ai/api/coding/paas/v4"),
                self._entry("Z.ai API", "https://api.z.ai/api/paas/v4"),
            ]
        }

        config.validate_config(data)  # 不抛即通过

    def test_provider_with_api_key_breaks_collision(self) -> None:
        """一方已设 api_key（不依赖环境变量回退）即无串用面。"""
        placeholder_key = "placeholder-" + "k"
        data = {
            "providers": [
                self._entry("DeepSeek", "https://api.deepseek.com"),
                self._entry("deepseek", "https://proxy.example.com/v1", api_key=placeholder_key),
            ]
        }

        config.validate_config(data)  # 不抛即通过

    def test_conflicting_keyless_env_vars_grouping(self) -> None:
        providers = [
            self._entry("DeepSeek", "https://api.deepseek.com"),
            self._entry("deepseek", "https://proxy.example.com/v1"),
            self._entry("Other", "https://other.example.com"),
        ]

        groups = config.conflicting_keyless_env_vars(providers)

        assert set(groups) == {"CRA_DEEPSEEK_API_KEY"}
        assert sorted(groups["CRA_DEEPSEEK_API_KEY"]) == ["DeepSeek", "deepseek"]
