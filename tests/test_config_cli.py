"""cra config 交互式配置与 config test 连通性自检:全流程离线
(mock input/getpass/probe),经 home_dir 隔离用户目录。

样例中的占位串均为明显假值(placeholder-*,非真实凭据),以变量构造写入,
避免密钥扫描对测试样例字面量的误报。
"""

import json
import os
import subprocess
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


def _ok_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    """setx 打桩：恒成功（不真实写环境）。"""
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


def _run_config(monkeypatch: pytest.MonkeyPatch, inputs: list[str], keys: list[str] | None = None) -> None:
    """驱动 cra config 交互模式;inputs 供菜单与表单,key 序列供 getpass。

    输入清单耗尽后抛 EOFError(模拟管道结束,与 test_cli._feed 同型):存储选择等
    新增询问处的中断语义用例依赖该行为。
    """
    iterator = iter(inputs)

    def fake_input(*_args: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    key_iterator = iter(keys or [])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p: next(key_iterator))
    cli._run_config_interactive()


class TestConfigAddProvider:
    # M8 起添加供应商先选类型（1 自定义 + 8 预设），key 后有存储选择（回车=明文）
    def test_add_provider_writes_file(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        _run_config(
            monkeypatch,
            [
                "1", "1", "deepseek", "https://api.deepseek.com",
                "deepseek-flash, deepseek-chat", "1", "0",
            ],
            keys=[_PLACEHOLDER_KEY],
        )

        entry = config.load_raw()["providers"][0]
        assert entry["name"] == "deepseek"
        assert entry["base_url"] == "https://api.deepseek.com"
        assert entry["models"] == ["deepseek-flash", "deepseek-chat"]

    def test_base_url_default_applied(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _run_config(
            monkeypatch,
            ["1", "1", "deepseek", "", "deepseek-flash", "1", "0"],
            keys=[_PLACEHOLDER_KEY],
        )

        assert config.load_raw()["providers"][0]["base_url"] == config.DEFAULT_BASE_URL

    def test_key_placeholder_means_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """key 留空 → 回退环境变量(条目 api_key 为空串,resolve 时回退)；不询问存储。"""
        _run_config(
            monkeypatch, ["1", "1", "deepseek", "", "deepseek-flash", "0"], keys=[""]
        )

        assert config.load_raw()["providers"][0]["api_key"] == ""

    def test_duplicate_name_asks_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["old-model"])]})

        _run_config(
            monkeypatch,
            ["1", "1", "deepseek", "y", "https://new.example.com", "1", "0"],
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

        _run_config(monkeypatch, ["1", "1", "deepseek", "n", "0"])

        assert config.load_raw() == original

    def test_provider_name_with_slash_rejected(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        _run_config(monkeypatch, ["1", "1", "a/b", "0"])

        assert "'/'" in capsys.readouterr().out
        assert not _models_path(home_dir).exists()

    def test_write_notice_mentions_permissions_and_backup(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """首次写盘时提示文件权限与备份注意事项。"""
        _run_config(
            monkeypatch,
            ["1", "1", "deepseek", "", "deepseek-flash", "1", "0"],
            keys=[_PLACEHOLDER_KEY],
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

        # key 输入后的存储选择（回车 = 明文）
        _run_config(monkeypatch, ["5", "1", "1", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert config.load_raw()["providers"][0]["api_key"] == _PLACEHOLDER_UPDATED

    def test_manage_models_add_new(self, monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1", "m2"])]})

        _run_config(monkeypatch, ["3", "1", "1", "m3", "0", "0"])

        assert config.load_raw()["providers"][0]["models"] == ["m1", "m2", "m3"]

    def test_manage_models_add_duplicate_asks_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """同名模型已存在时询问是否覆盖(清单为纯名称,覆盖即确认保留)。

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
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
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


class TestConfigPresets:
    """M8 供应商预设（D17）：编号选择自动填官方接入点与模型清单，预填项均可修改。"""

    def test_preset_prefill_accepted(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """选 DeepSeek 预设：回车用官方 base_url、回车采用预填模型清单。"""
        _run_config(
            monkeypatch,
            ["1", "2", "", "", "1", "0"],
            keys=[_PLACEHOLDER_KEY],
        )

        entry = config.load_raw()["providers"][0]
        assert entry["name"] == "DeepSeek"
        assert entry["base_url"] == "https://api.deepseek.com"
        assert entry["models"] == ["deepseek-flash", "deepseek-v4-pro"]

    def test_preset_prefills_are_editable(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """预设预填项均可修改：自定义 base_url；模型清单答 n 后自行输入。"""
        _run_config(
            monkeypatch,
            ["1", "2", "https://mirror.example.com", "n", "my-model-a, my-model-b", "1", "0"],
            keys=[_PLACEHOLDER_KEY],
        )

        entry = config.load_raw()["providers"][0]
        assert entry["base_url"] == "https://mirror.example.com"
        assert entry["models"] == ["my-model-a", "my-model-b"]

    def test_preset_menu_lists_custom_and_all_presets(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """类型清单 = 1 自定义 + 8 家预设；回车取消。"""
        _run_config(monkeypatch, ["1", "", "0"])

        out = capsys.readouterr().out
        assert "自定义" in out
        for preset in config.PROVIDER_PRESETS:
            assert preset.name in out
        assert "已取消" in out


class TestKeyStorageChoice:
    """M8 key 存储选择（D17）：明文（现状）或环境变量（models.json 不落明文）。"""

    @pytest.fixture(autouse=True)
    def _clean_provider_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量存储分支的用例不得受真实环境同名变量影响（否则会触发覆盖确认）。"""
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("CRA_MYPROV_API_KEY", raising=False)

    def test_plaintext_choice_keeps_current_behavior(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})

        _run_config(monkeypatch, ["5", "1", "1", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert config.load_raw()["providers"][0]["api_key"] == _PLACEHOLDER_UPDATED

    def test_env_storage_removes_plaintext_and_activates_process_env(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """选环境变量：已有明文移除、setx 以 (变量名, key) 调用、当前进程立即生效。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})
        run_calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            run_calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr(cli.sys, "platform", "win32")  # 断言 setx 分支，与平台解耦
        # os.environ 换为副本：进程内生效的断言可测、且不污染真实环境
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        entry = config.load_raw()["providers"][0]
        assert "api_key" not in entry
        assert run_calls and run_calls[0][:2] == ["setx", "CRA_DEEPSEEK_API_KEY"]
        assert os.environ.get("CRA_DEEPSEEK_API_KEY") == _PLACEHOLDER_UPDATED
        assert "已写入用户环境变量" in capsys.readouterr().out

    def test_setx_failure_falls_back_to_plaintext(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """setx 失败回退明文（有明确提示），key 不因存储失败而丢失。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})

        def failing_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        monkeypatch.setattr(cli.subprocess, "run", failing_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        entry = config.load_raw()["providers"][0]
        assert entry["api_key"] == _PLACEHOLDER_UPDATED
        assert os.environ.get("CRA_DEEPSEEK_API_KEY") is None  # 失败路径不写进程环境
        assert "setx 写入失败" in capsys.readouterr().out

    def test_posix_prints_export_instruction(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """POSIX：打印 export 指令由用户执行（不自动改 rc）；当前进程即时生效。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})
        monkeypatch.setattr(cli.sys, "platform", "linux")
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        out = capsys.readouterr().out
        assert "export CRA_DEEPSEEK_API_KEY=" in out
        assert "勿粘贴到公共场合" in out  # 明文回显的保密提醒
        assert os.environ.get("CRA_DEEPSEEK_API_KEY") == _PLACEHOLDER_UPDATED  # 本会话无断层
        assert "api_key" not in config.load_raw()["providers"][0]

    def test_storage_choice_interrupt_keeps_existing_key(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """存储选择处中断（EOF/Ctrl+C）不是确认：保留旧 key，不落任何新存储。"""
        original = {"providers": [_entry_with_key("deepseek", ["m1"])]}
        _write_models(home_dir, original)

        _run_config(monkeypatch, ["5", "1"], keys=[_PLACEHOLDER_UPDATED])  # 存储询问处 EOF

        assert config.load_raw() == original  # 旧 key 原样保留

    def test_custom_provider_generates_env_var_name(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """自定义供应商的存储变量名按规则生成（空格转 _、大写、_API_KEY 后缀）。"""
        entry: dict[str, Any] = {"name": "My Provider", "base_url": "https://x", "models": ["m"]}
        _write_models(home_dir, {"providers": [entry]})
        monkeypatch.setattr(cli.sys, "platform", "linux")

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert "api_key" not in config.load_raw()["providers"][0]
        assert config.provider_env_var("My Provider") == "CRA_MY_PROVIDER_API_KEY"

    def test_successful_env_write_registers_var(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """环境变量写入成功后登记变量名（卸载清理的依据）。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})
        monkeypatch.setattr(cli.subprocess, "run", _ok_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert config.registered_env_vars() == ["CRA_DEEPSEEK_API_KEY"]


class TestEnvVarOverwriteGuard:
    """同名环境变量覆盖保护（D17 修订）：勿删用户已有变量，自己的直接更新。"""

    @pytest.fixture(autouse=True)
    def _clean_provider_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)

    def test_foreign_var_confirmed_overwrites_and_registers(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """同名变量非本程序所设：询问后答 y → setx 覆盖并登记（卸载时将清理它）。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-user-own")
        monkeypatch.setattr(cli.subprocess, "run", _ok_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        # 存储选 2 后遇到覆盖确认，答 y
        _run_config(monkeypatch, ["5", "1", "2", "y", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert config.registered_env_vars() == ["CRA_DEEPSEEK_API_KEY"]
        assert os.environ["CRA_DEEPSEEK_API_KEY"] == _PLACEHOLDER_UPDATED
        assert "api_key" not in config.load_raw()["providers"][0]

    def test_foreign_var_declined_falls_back_to_plaintext(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """同名变量非本程序所设：拒绝覆盖 → key 回退明文、原变量值不动、不登记。"""
        original = {"providers": [_entry_with_key("deepseek", ["m1"])]}
        _write_models(home_dir, original)
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-user-own")
        monkeypatch.setattr(cli.subprocess, "run", _ok_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "n", "0"], keys=[_PLACEHOLDER_UPDATED])

        entry = config.load_raw()["providers"][0]
        assert entry["api_key"] == _PLACEHOLDER_UPDATED  # 回退明文，key 不丢
        assert os.environ["CRA_DEEPSEEK_API_KEY"] == "placeholder-user-own"  # 用户原值未动
        assert config.registered_env_vars() == []  # 未登记 → 卸载不会碰它

    def test_owned_var_overwrites_without_asking(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """变量已是本程序登记在案的：更新 key 不再询问（避免每次改 key 都被打断）。"""
        _write_models(home_dir, {"providers": [_entry_with_key("deepseek", ["m1"])]})
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-previous")
        config.register_env_var("CRA_DEEPSEEK_API_KEY")  # 之前由本程序设置
        monkeypatch.setattr(cli.subprocess, "run", _ok_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2", "0"], keys=[_PLACEHOLDER_UPDATED])

        assert os.environ["CRA_DEEPSEEK_API_KEY"] == _PLACEHOLDER_UPDATED  # 直接覆盖
        assert config.registered_env_vars() == ["CRA_DEEPSEEK_API_KEY"]

    def test_interrupt_at_overwrite_prompt_falls_back_to_plaintext(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """覆盖询问处中断（EOF/Ctrl+C）不是确认：回退明文、原变量不动。"""
        original = {"providers": [_entry_with_key("deepseek", ["m1"])]}
        _write_models(home_dir, original)
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-user-own")
        monkeypatch.setattr(cli.subprocess, "run", _ok_run)
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))

        _run_config(monkeypatch, ["5", "1", "2"], keys=[_PLACEHOLDER_UPDATED])  # 询问处 EOF

        assert config.load_raw()["providers"][0]["api_key"] == _PLACEHOLDER_UPDATED
        assert os.environ["CRA_DEEPSEEK_API_KEY"] == "placeholder-user-own"
        assert config.registered_env_vars() == []


class TestConfigTestChain:
    """cra config test 与 resolve 同一条 key 回退链（D17）。"""

    def test_probe_uses_provider_env_var(
        self, monkeypatch: pytest.MonkeyPatch, home_dir: Path
    ) -> None:
        """条目无 key 时探测回退 CRA_供应商名大写_API_KEY（回退链第二级）。"""
        entry: dict[str, Any] = {
            "name": "myprov", "base_url": "https://x.example.com", "models": ["m"]
        }
        _write_models(home_dir, {"providers": [entry]})
        monkeypatch.setenv("CRA_MYPROV_API_KEY", "placeholder-provider-env")
        calls: list[dict[str, str]] = []

        def fake_probe(base_url: str, api_key: str, model_name: str) -> tuple[str, str]:
            calls.append({"base_url": base_url, "api_key": api_key, "model_name": model_name})
            return "ok", ""

        monkeypatch.setattr(cli.llm, "probe", fake_probe)

        with pytest.raises(SystemExit) as excinfo:
            cli._run_config_test()

        assert excinfo.value.code == 0
        assert calls[0]["api_key"] == "placeholder-provider-env"

def test_register_env_var_failure_warns_without_raising(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """登记失败（任何异常）只降级为警告，不阻塞已成功的 key 写入流程。"""
    from rich.console import Console

    def boom(_name: str) -> None:
        raise RuntimeError("登记文件写入失败")

    monkeypatch.setattr(cli.config, "register_env_var", boom)

    cli._register_env_var(Console(), "MY_KEY")

    out = capsys.readouterr().out
    assert "环境变量登记失败" in out
    assert "MY_KEY" in out
