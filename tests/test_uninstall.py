"""cra uninstall（§7.1/§10 M8）：两道确认分支、数据删除、分离任务命令拼装与 PATH 判定。

subprocess 一律 mock——绝不真实执行 uv/卸载/PATH 修改；用户目录经 home_dir 隔离。
"""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cra import cli


def _feed(inputs: list[str]) -> Any:
    """input 打桩：按序返回；清单耗尽后抛 EOFError（与 test_cli._feed 同型）。"""
    iterator = iter(inputs)

    def fake_input(*_args: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return fake_input


def _patch_uv_list(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    """mock uv tool list；返回 run 调用记录。"""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def _patch_popen(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """mock Popen（分离任务绝不真实启动）；返回调用记录。"""
    calls: list[dict[str, Any]] = []

    def fake_popen(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": list(args), "kwargs": kwargs})
        return None

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    return calls


def _seed_user_data(home_dir: Path) -> Path:
    cra_dir = home_dir / ".cra"
    (cra_dir / "sessions").mkdir(parents=True)
    (cra_dir / "models.json").write_text("{}", encoding="utf-8")
    (cra_dir / "sessions" / "_last.json").write_text("[]", encoding="utf-8")
    return cra_dir


class TestUninstallConfirmation:
    """两道交互确认：清单确认默认 N 防误触发；用户数据默认 Y（删干净）。"""

    def test_first_confirm_decline_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        cra_dir = _seed_user_data(home_dir)
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["n"]))

        cli._run_uninstall()  # 正常返回（无 sys.exit，退出码 0 语义）

        out = capsys.readouterr().out
        assert "已取消，未删除任何内容" in out
        assert cra_dir.exists()  # 数据未动
        assert popen_calls == []  # 后台任务未生成

    def test_confirm_with_default_deletes_user_data_and_spawns_task(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """确认卸载 + 数据询问回车（默认 Y）：~/.cra 删除、分离任务生成。"""
        cra_dir = _seed_user_data(home_dir)
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\n")
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["y", ""]))

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert "用户数据已删除" in out
        assert not cra_dir.exists()
        assert "程序将在退出后完成卸载" in out
        assert len(popen_calls) == 1

    def test_confirm_data_interrupt_keeps_data(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """数据询问处 EOF/Ctrl+C（None）不是确认：保留 ~/.cra/，工具卸载照常。"""
        cra_dir = _seed_user_data(home_dir)
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\n")
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["y"]))  # 数据询问处清单耗尽 → EOF

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert "输入已中断" in out
        assert cra_dir.exists()
        assert len(popen_calls) == 1

    def test_confirm_keep_data_preserves_cra_dir(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """数据询问答 n：~/.cra 保留，工具卸载照常。"""
        cra_dir = _seed_user_data(home_dir)
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\n")
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["y", "n"]))

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert "已保留" in out
        assert cra_dir.exists()
        assert len(popen_calls) == 1
        assert "程序将在退出后完成卸载" in out

    def test_missing_uv_tool_skips_uninstall(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """uv tool list 无本工具（如 uv run 方式使用）：跳过工具卸载并说明。"""
        _seed_user_data(home_dir)
        _patch_uv_list(monkeypatch, "other-tool v2.0\n- x\n")
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["y", "n"]))

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert "跳过工具卸载" in out
        assert popen_calls == []

    def test_missing_uv_list_failure_treated_as_not_installed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """uv 不可用（run 抛 OSError）视同未安装：不生成任务、不崩溃。"""

        def broken_run(_cmd: list[str], **_kwargs: Any) -> Any:
            raise OSError("uv not found")

        monkeypatch.setattr(cli.subprocess, "run", broken_run)
        _seed_user_data(home_dir)
        popen_calls = _patch_popen(monkeypatch)
        monkeypatch.setattr("builtins.input", _feed(["y", "n"]))

        cli._run_uninstall()

        assert "跳过工具卸载" in capsys.readouterr().out
        assert popen_calls == []


class TestUninstallBackgroundTask:
    """分离任务命令拼装与 PATH 移除判定（可注入小函数）。"""

    def test_path_entry_decision(self) -> None:
        """无其他 uv 工具才移除 PATH 条目；共享条目保留。"""
        assert cli._should_remove_uv_path_entry([]) is True
        assert cli._should_remove_uv_path_entry(["other-tool"]) is False

    def test_windows_script_contains_wait_uninstall_and_path_cleanup(self) -> None:
        script = cli._windows_cleanup_script(4242, "code-review-agent", remove_path=True)

        assert "Get-Process -Id 4242" in script  # 等待父进程退出
        assert "uv tool uninstall code-review-agent" in script
        assert "SetEnvironmentVariable" in script  # PATH 移除经 PowerShell 用户级写回
        assert ".local\\bin" in script
        assert "$LASTEXITCODE -eq 0" in script  # uv 失败（如分离进程无 uv）不得误判为空
        assert "$names.Count -eq 0" in script  # 卸载后复查：无其他工具才动手

    def test_windows_script_without_path_block_when_shared(self) -> None:
        """PATH 条目被其他 uv 工具共享（others 非空 → remove_path=False）：不含 PATH 块。"""
        script = cli._windows_cleanup_script(1, "code-review-agent", remove_path=False)

        assert "uv tool uninstall code-review-agent" in script
        assert "SetEnvironmentVariable" not in script

    def test_posix_script_waits_then_uninstalls(self) -> None:
        script = cli._posix_cleanup_script(77, "code-review-agent")

        assert "kill -0 77" in script  # 等待父进程退出
        assert "uv tool uninstall code-review-agent" in script

    def test_spawn_cleanup_task_windows(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """Windows：detached PowerShell + CREATE_* 旗标；脚本含本进程 PID。"""
        monkeypatch.setattr(cli.sys, "platform", "win32")
        popen_calls = _patch_popen(monkeypatch)

        cli._spawn_cleanup_task("code-review-agent", remove_path=True)

        assert len(popen_calls) == 1
        args = popen_calls[0]["args"]
        kwargs = popen_calls[0]["kwargs"]
        assert args[0][0] == "powershell"
        script = args[0][-1]
        assert str(os.getpid()) in script  # 等待的就是当前进程
        assert kwargs["creationflags"] != 0
        assert kwargs["stdout"] == subprocess.DEVNULL

    def test_spawn_cleanup_task_posix(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """POSIX：nohup sh -c 语义（start_new_session 分离），不触碰 PATH。"""
        monkeypatch.setattr(cli.sys, "platform", "linux")
        popen_calls = _patch_popen(monkeypatch)

        cli._spawn_cleanup_task("code-review-agent", remove_path=True)

        assert len(popen_calls) == 1
        args = popen_calls[0]["args"]
        kwargs = popen_calls[0]["kwargs"]
        assert args[0] == ["sh", "-c", cli._posix_cleanup_script(os.getpid(), "code-review-agent")]
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs

    def test_posix_prints_rc_hint_when_path_removal_due(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """POSIX 无其他 uv 工具：打印待删的 shell 配置行指令（不自动改 rc）。"""
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\n")
        _patch_popen(monkeypatch)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        monkeypatch.setattr("builtins.input", _feed(["y", "n"]))

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert 'export PATH="$HOME/.local/bin:$PATH"' in out
        assert "程序将在退出后完成卸载" in out

    def test_shared_path_entry_no_posix_hint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """PATH 条目被其他 uv 工具共享：不打印删除指令（保留条目）。"""
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\nother v2.0\n- x\n")
        _patch_popen(monkeypatch)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        monkeypatch.setattr("builtins.input", _feed(["y", "n"]))

        cli._run_uninstall()

        out = capsys.readouterr().out
        assert "export PATH" not in out
        assert "程序将在退出后完成卸载" in out  # 工具卸载仍执行


class TestUvToolNames:
    """uv tool list 解析：非 "-" 行首词为工具名；失败/不可用返回空表。"""

    def test_parses_tool_names_skipping_alias_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_uv_list(monkeypatch, "code-review-agent v0.1.0\n- cra\nother v2.0\n- x\n- y\n")

        assert cli._uv_tool_names() == ["code-review-agent", "other"]

    def test_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_run(_cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(_cmd, 1, stdout="", stderr="err")

        monkeypatch.setattr(cli.subprocess, "run", failing_run)

        assert cli._uv_tool_names() == []

    def test_uv_unavailable_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_run(_cmd: list[str], **_kwargs: Any) -> Any:
            raise OSError("no uv")

        monkeypatch.setattr(cli.subprocess, "run", broken_run)

        assert cli._uv_tool_names() == []


class TestUninstallWiring:
    """§7.1 入口接线：子命令解析与 main 分发。"""

    def test_uninstall_subcommand_parses(self) -> None:
        args = cli._build_parser().parse_args(["uninstall"])

        assert args.command == "uninstall"

    def test_main_dispatches_to_uninstall(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[bool] = []
        monkeypatch.setattr(cli, "_run_uninstall", lambda: calls.append(True))
        monkeypatch.setattr("sys.argv", ["cra", "uninstall"])

        cli.main()

        assert calls == [True]

    def test_delete_user_data_removes_cra_dir(self, capsys: pytest.CaptureFixture[str], home_dir: Path) -> None:
        cra_dir = _seed_user_data(home_dir)

        cli._delete_user_data(capsys_console(capsys))

        assert not cra_dir.exists()
        assert "用户数据已删除" in capsys.readouterr().out

    def test_delete_user_data_tolerates_missing_dir(
        self, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        cli._delete_user_data(capsys_console(capsys))

        assert "跳过数据清理" in capsys.readouterr().out

    def test_delete_user_data_refuses_unexpected_dir(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """删除目标名非 .cra 时拒绝执行（config 路径演化下的误删护栏，cra 自审复核）。"""
        rogue = tmp_path / "elsewhere"
        rogue.mkdir()
        (rogue / "models.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(cli.config, "config_path", lambda: rogue / "models.json")

        cli._delete_user_data(capsys_console(capsys))

        assert rogue.exists()
        assert "已取消删除" in capsys.readouterr().out


def capsys_console(capsys: pytest.CaptureFixture[str]) -> Any:
    """供 _delete_user_data 使用的极简 console 替身（print 直写，经 capsys 捕获）。"""

    class _MiniConsole:
        def print(self, *args: Any, **_kwargs: Any) -> None:
            print(*args)

    return _MiniConsole()
