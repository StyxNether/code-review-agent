"""cra config 交互式配置(§7.4)与 config test 连通性自检:全流程离线
(mock input/getpass/probe),经 home_dir 隔离用户目录。

样例中的占位串均为明显假值(placeholder-*,非真实凭据),以变量构造写入,
避免密钥扫描对测试样例字面量的误报。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from cra import cli, config

_PLACEHOLDER_KEY = "placeholder-a"
_PLACEHOLDER_UPDATED = "placeholder-updated"


def _models_path(home: Path) -> Path:
    return home / ".cra" / "models.json"


def _write_models(home: Path, data: dict) -> None:
    path = _models_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _entry_with_key(name: str, models: list[str]) -> dict:
    """构造 provider 条目;凭据字段以变量赋值(占位假值)。"""
    entry: dict[str, Any] = {"name": name, "base_url": "https://x.example.com", "models": models}
    entry["api_key"] = _PLACEHOLDER_KEY
    return entry


def _run_config(monkeypatch: pytest.MonkeyPatch, inputs: list[str], keys: list[str] | None = None) -> None:
    """驱动 cra config 交互模式;inputs 供菜单与表单,key 序列供 getpass。"""
    iterator = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *a: next(iterator))
    key_iterator = iter(keys or [])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p: next(key_iterator))
    cli._run_config_interactive()


class TestConfigAddProvider:
    def test_add_provider_writes_file(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        _run_config(
            monkeypatch,
            ["1", "deepseek", "https://api.deepseek.com", "deepseek-flash, deepseek-chat", "0"],
            keys=[_PLACEHOLDER_KEY],
        )

        entry = config.load_raw()["providers"][0]
        assert entry["name"] == "deepseek"
        assert entry["base_url"] == "https://api.deepseek.com"
        assert entry["models"] == ["deepseek-flash", "deepseek-chat"]

    def test_base_url_default_applied(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _run_config(
            monkeypatch, ["1", "deepseek", "", "deepseek-flash", "0"], keys=[_PLACEHOLDER_KEY]
        )

        assert config.load_raw()["providers"][0]["base_url"] == config.DEFAULT_BASE_URL

    def test_key_placeholder_means_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """key 留空 → 回退环境变量(条目 api_key 为空串,resolve 时回退)。"""
        _run_config(monkeypatch, ["1", "deepseek", "", "deepseek-flash", "0"], keys=[""])

        assert config.load_raw()["providers"][0]["api_key"] == ""

    def test_duplicate_name_asks_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["old-model"])]})

        _run_config(
            monkeypatch,
            ["1", "deepseek", "y", "https://new.example.com", "0"],
            keys=[_PLACEHOLDER_UPDATED],
        )

        providers = config.load_raw()["providers"]
        assert len(providers) == 1  # 未新增,原地更新
        assert providers[0]["base_url"] == "https://new.example.com"
        assert providers[0]["models"] == ["old-model"]  # 模型清单保留
        assert "已更新" in capsys.readouterr().out

    def test_duplicate_name_declined_keeps_file(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        original = {"providers": [_entry_with_key("deepseek", ["old-model"])]}
        _write_models(home_dir, original)

        _run_config(monkeypatch, ["1", "deepseek", "n", "0"])

        assert config.load_raw() == original

    def test_provider_name_with_slash_rejected(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _run_config(monkeypatch, ["1", "a/b", "0"])

        assert "'/'" in capsys.readouterr().out
        assert not _models_path(home_dir).exists()

    def test_write_notice_mentions_permissions_and_backup(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """§7.4:首次写盘时提示文件权限与备份注意事项。"""
        _run_config(
            monkeypatch, ["1", "deepseek", "", "deepseek-flash", "0"], keys=[_PLACEHOLDER_KEY]
        )

        out = capsys.readouterr().out
        assert "权限" in out
        assert "备份" in out


class TestConfigProviders:
    def test_remove_provider_clears_default(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        _write_models(
            home_dir, {"default": "deepseek/m1", "providers": [_entry_with_key("deepseek", ["m1", "m2"])]}
        )

        _run_config(monkeypatch, ["2", "1", "y", "0"])

        data = config.load_raw()
        assert data["providers"] == []
        assert "default" not in data

    def test_set_default(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1", "m2"])]})

        _run_config(monkeypatch, ["4", "1", "2", "0"])

        assert config.load_raw()["default"] == "deepseek/m2"

    def test_update_key(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})

        _run_config(monkeypatch, ["5", "1", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert config.load_raw()["providers"][0]["api_key"] == _PLACEHOLDER_UPDATED

    def test_manage_models_add_new(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1", "m2"])]})

        _run_config(monkeypatch, ["3", "1", "1", "m3", "0", "0"])

        assert config.load_raw()["providers"][0]["models"] == ["m1", "m2", "m3"]

    def test_manage_models_add_duplicate_asks_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """§7.4:同名模型已存在时询问是否覆盖(清单为纯名称,覆盖即确认保留)。

        覆盖询问经 input 提示呈现(mock 下不产生输出),以"保持不变"回执与
        清单不变断言该分支被走到。
        """
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1", "m2"])]})

        _run_config(monkeypatch, ["3", "1", "1", "m1", "y", "0", "0"])

        assert "保持不变" in capsys.readouterr().out
        assert config.load_raw()["providers"][0]["models"] == ["m1", "m2"]

    def test_manage_models_remove_keeps_at_least_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("solo", ["only"])]})

        _run_config(monkeypatch, ["3", "1", "2", "0", "0"])

        assert "至少保留一个模型" in capsys.readouterr().out

    def test_broken_config_exits_2(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _models_path(home_dir).parent.mkdir(parents=True, exist_ok=True)
        _models_path(home_dir).write_text("{broken", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            _run_config(monkeypatch, [])

        assert excinfo.value.code == 2


class TestConfigTest:
    """cra config test:逐供应商探测报告;退出码 0 全 OK / 1 有失败 / 2 配置错误。"""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """探测的 key 回退读真实环境变量,测试中统一清空。"""
        monkeypatch.delenv("SE_CodeAgent", raising=False)

    def _patch_probes(
        self, monkeypatch: pytest.MonkeyPatch, results: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        probe_results = iter(results)
        calls: list[dict[str, str]] = []

        def fake_probe(base_url: str, api_key: str, model_name: str) -> tuple[str, str]:
            calls.append({"base_url": base_url, "api_key": api_key, "model_name": model_name})
            return next(probe_results)

        monkeypatch.setattr(cli.llm, "probe", fake_probe)
        return calls

    def test_all_ok_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(
            home_dir,
            {"providers": [_entry_with_key("a", ["m"]), _entry_with_key("b", ["m"])]},
        )
        self._patch_probes(monkeypatch, [("ok", ""), ("ok", "")])

        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 0
        assert "OK" in capsys.readouterr().out

    def test_auth_failure_reported_exit_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("a", ["m"])]})
        self._patch_probes(monkeypatch, [("auth", "401 detail")])

        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "401" in out
        assert "401 detail" in out

    def test_timeout_reported(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("a", ["m"])]})
        self._patch_probes(monkeypatch, [("timeout", "timed out")])

        with pytest.raises(SystemExit):
            cli._run_config_test()

        assert "超时" in capsys.readouterr().out

    def test_missing_key_reported_without_probe(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """供应商与环境的 key 都缺时直接报告,不发探测请求。"""
        entry: dict[str, Any] = {"name": "a", "base_url": "https://x", "models": ["m"]}
        _write_models(home_dir, {"providers": [entry]})
        calls = self._patch_probes(monkeypatch, [("ok", "")])

        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 1
        assert "未配置 key" in capsys.readouterr().out
        assert calls == []

    def test_no_providers_exits_2(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 2

    def test_probe_receives_provider_settings(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """探测使用各供应商自己的 base_url/key 与清单第一个模型。"""
        entry = _entry_with_key("a", ["first-model"])
        entry["base_url"] = "https://a.example.com"
        _write_models(home_dir, {"providers": [entry]})
        calls = self._patch_probes(monkeypatch, [("ok", "")])

        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 0
        assert calls == [
            {
                "base_url": "https://a.example.com",
                "api_key": _PLACEHOLDER_KEY,
                "model_name": "first-model",
            }
        ]
