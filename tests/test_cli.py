"""cli 单元：参数解析、分级标记着色渲染与围栏状态机、REPL 循环与设置指引
（不产生真实终端输出，无网络）。"""

import json
import re
import sys
from typing import Any

import pytest
from rich.console import Console

from cra import cli
from cra.cli import (
    _build_parser,
    _in_fence,
    _marker_style,
    _print_config_guide,
    _setup_hint,
    _StreamingMarkdown,
)


class TestParser:
    """M2 补齐：CLI 参数解析此前无测试覆盖。"""

    def test_chat_subcommand_parses(self) -> None:
        args = _build_parser().parse_args(["chat"])

        assert args.command == "chat"

    def test_missing_subcommand_exits_with_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

        assert "chat" in capsys.readouterr().err

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["nope"])

    def test_chat_help_declares_non_sandbox_and_timeout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """§8.2 核对项：非沙箱、限时执行的限制声明在 CLI 层面可见。"""
        with pytest.raises(SystemExit) as excinfo:
            _build_parser().parse_args(["chat", "--help"])

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "非沙箱" in out
        assert "限时" in out


class TestSetupGuide:
    """M3 挂账：key 指引跨平台化；输出统一经 rich Console（不再混用 print）。

    M5：无任何配置时的引导改为 _print_config_guide（§5/§7.4，含 cra config 指引）。
    """

    def test_windows_hint_mentions_setx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        assert "setx" in _setup_hint()

    def test_posix_hint_mentions_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert "export" in _setup_hint()

    def test_config_guide_mentions_cra_config_and_env_fallback(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_config_guide(Console())

        out = capsys.readouterr().out
        assert "cra config" in out
        assert "SE_CodeAgent" in out


class _FakeAgent:
    """替换真实 Agent:避免构造模型客户端;记录 /clear、对话轮次与命令族操作。

    run 按真实 Agent 的回调时序模拟:叙述增量 → 工具打点 → 最终回答增量
    (供过程叙述分流的行为断言)。
    """

    last: "_FakeAgent | None" = None
    undo_default = True  # 类属性:undo 测试经 monkeypatch 覆盖(每次构造实例会重建)

    def __init__(self, **_kwargs: Any) -> None:
        self.model = "test-model"
        self.reset_calls = 0
        self.turns: list[str] = []
        self.switched: list[Any] = []
        self.loaded: list[list[dict[str, Any]]] = []
        self.undo_result = _FakeAgent.undo_default
        _FakeAgent.last = self

    def reset(self) -> None:
        self.reset_calls += 1

    def run(self, user_input: str, **kwargs: Any) -> str:
        self.turns.append(user_input)
        on_delta = kwargs.get("on_delta")
        on_tool = kwargs.get("on_tool")
        if on_delta is not None:
            on_delta("先看看文件再作答。")  # 工具轮伴随叙述(无段落空行)
        if on_tool is not None:
            on_tool("list_dir", "{}")
        if on_delta is not None:
            on_delta("最终回答正文。")
        return "最终回答正文。"

    # ---- M5 命令族所需的能力(/model /save /load /undo /context)----

    def switch_model(self, resolved: Any) -> None:
        self.switched.append(resolved)
        self.model = resolved.model

    def undo_turn(self) -> bool:
        return self.undo_result

    def context_status(self) -> dict[str, Any]:
        return {"used": 1234, "limit": 100000, "estimated": False}

    def dump_session(self) -> list[dict[str, Any]]:
        return [{"type": "human", "data": {"content": "审查 x.py"}}]

    def load_session(self, data: list[dict[str, Any]]) -> None:
        self.loaded.append(data)


class TestRunChat:
    """M3 挂账：_run_chat 的 sys.exit 路径与 REPL 内置命令可测（mock input/Agent）。"""

    def test_missing_key_exits_with_guide(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """§5/§7.5：无任何配置打印 cra config 引导后退出 2（M5 起退出码从 1 改 2）。"""
        monkeypatch.delenv("SE_CodeAgent", raising=False)

        with pytest.raises(SystemExit) as excinfo:
            cli._run_chat()

        assert excinfo.value.code == 2
        assert "cra config" in capsys.readouterr().out

    def test_exit_command_ends_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", lambda: "exit")

        cli._run_chat()

        assert "再见" in capsys.readouterr().out

    def test_eof_ends_repl_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)

        def raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)

        cli._run_chat()  # 不抛异常即通过

        assert "再见" in capsys.readouterr().out

    def test_clear_resets_agent_and_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        inputs = iter(["/clear", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))

        cli._run_chat()

        assert _FakeAgent.last is not None
        assert _FakeAgent.last.reset_calls == 1

    def test_empty_input_continues_without_api_call(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        inputs = iter(["", "   ", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))

        cli._run_chat()

        # 空输入/纯空白只重新提示，不发起对话轮次（§6）
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.turns == []
        assert capsys.readouterr().out.count("你>") == 3

    def test_narration_flushed_before_tool_mark_and_answer_separate(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """M4 验收反馈：过程性叙述跟随工具轮实时暗色显示，不与最终回答堆叠。"""
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        inputs = iter(["审查 x.py", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))

        cli._run_chat()

        out = capsys.readouterr().out
        narration_at = out.find("先看看文件再作答。")
        tool_at = out.find("⚙ list_dir")
        answer_at = out.find("最终回答正文。")
        assert narration_at != -1 and tool_at != -1 and answer_at != -1
        assert narration_at < tool_at < answer_at  # 叙述 → 工具打点 → 最终回答 的时序


def test_inline_marker_kept_on_one_line() -> None:
    """回归（验收走查实测）：行内标记不再被 Markdown 片段自带换行拆成多行。"""
    out = _render_segment("这里有个问题【严重】需要修复。")

    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "这里有个问题" in lines[0] and "需要修复。" in lines[0]
    assert "\x1b[1;31m【严重】" in lines[0]  # 标记着色不变


def test_inline_marker_preserves_surrounding_text_and_space() -> None:
    """拼接收敛行尾后，原文的词间空格与行内代码渲染保持不变。"""
    out = _render_segment("检查 `x=1` 这里 【严重】 崩溃风险。")

    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "检查" in lines[0] and "崩溃风险。" in lines[0]
    assert "`x=1`" not in lines[0]  # 文本片仍经 Markdown 渲染（行内代码反引号不字面出现）
    plain = re.sub(r"\x1b\[[0-9;]*m", "", lines[0])  # 剥离样式码后比对原文空格
    assert "这里 【严重】 崩溃风险。" in plain


def test_marker_only_line_still_renders() -> None:
    """整行只有一个标记（行首标记行）时正常渲染一次，不产生多余空行。"""
    out = _render_segment("【建议】")
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines == ["【建议】"] or len(lines) == 1
    assert "\x1b[1;32m【建议】" in out


def test_flush_pending_renders_dim_plain_text_and_clears_buffer() -> None:
    """flush_pending：暗色平文落盘（非 Markdown 渲染），并清空缓冲。"""
    console = Console(record=True, width=120, force_terminal=True, color_system="truecolor")
    streamer = _StreamingMarkdown(console)
    streamer.feed("先看看 `夹具` 再答。")  # 无段落空行，feed 阶段不会渲染
    streamer.flush_pending()
    out = console.export_text(styles=True)

    assert "先看看 `夹具` 再答。" in out  # 平文：反引号字面保留（未走 Markdown 渲染）
    assert "\x1b[2m" in out  # dim 样式（与正文区分）
    streamer.feed("最终回答段落。")
    streamer.finish()
    assert "最终回答段落" in console.export_text(styles=True)  # 缓冲已清空，正文正常渲染


class _SilentAgent:
    """不产生任何增量回调的假 Agent（兼容端点无文本增量的兜底路径验证）。"""

    def __init__(self, **_kwargs: Any) -> None:
        self.model = "test-model"

    def reset(self) -> None:
        pass

    def run(self, user_input: str, **_kwargs: Any) -> str:
        return "无增量完整回答。"


def test_no_delta_answer_falls_back_to_return_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """验收走查建议：兼容端点不返回文本增量时，界面用返回值兜底而非空白。"""
    monkeypatch.setenv("SE_CodeAgent", "sk-test")
    monkeypatch.setattr(cli, "Agent", _SilentAgent)
    inputs = iter(["问", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    cli._run_chat()

    assert "无增量完整回答。" in capsys.readouterr().out


def test_all_markers_bolded_in_one_segment() -> None:
    """回归：三个标记都要替换，不能命中第一个就返回。"""
    text = "【一般】a.py:1 — x。\n【建议】a.py:2 — y。"

    result = _render_segment(text)

    assert "\x1b[1;33m【一般】" in result  # 黄色加粗
    assert "\x1b[1;32m【建议】" in result  # 绿色加粗


def test_markers_inside_code_fence_untouched() -> None:
    """回归：围栏代码块内的标记不着色、不插入任何标记字符。"""
    text = "正文【严重】。\n```\n【严重】不要着色\n```\n尾部【建议】"

    result = _render_segment(text)

    assert "\x1b[1;31m【严重】" in result  # 正文着色
    fence_line = next(line for line in result.splitlines() if "不要着色" in line)
    assert "\x1b[1;31m" not in fence_line  # 围栏内保持原样
    assert "\x1b[1;32m【建议】" in result


def test_paragraph_end_skips_tilde_fence() -> None:
    streamer = _StreamingMarkdown(Console())
    streamer._buffer = "a\n\n~~~py\nx\n\ny\n~~~\n\nb"

    assert streamer._paragraph_end() == 1  # 第一个空行在围栏外
    streamer._buffer = "~~~py\nx\n\ny\n~~~\n\nb"
    assert streamer._paragraph_end() == 14  # 围栏内的空行不切段


def test_paragraph_end_respects_longer_backtick_fence() -> None:
    streamer = _StreamingMarkdown(Console())
    # ```` 围栏内部的 ``` 不闭合围栏
    streamer._buffer = "````md\nx\n\n```\n````\n\nb"

    assert streamer._paragraph_end() == 18


def test_in_fence_state() -> None:
    assert not _in_fence("a\n\nb")
    assert _in_fence("```py\nx")
    assert not _in_fence("```py\nx\n```")
    assert _in_fence("~~~\nx\n```")  # 字符不同的围栏不互相闭合


class TestMarkerStyling:
    """M2 验收反馈：分级标记红/黄/绿、修改建议加粗。"""

    def test_style_mapping(self) -> None:
        assert _marker_style("【严重】") == "bold red"
        assert _marker_style("【一般】") == "bold yellow"
        assert _marker_style("【建议】") == "bold green"
        assert _marker_style("修改建议：") == "bold"


def _render_segment(text: str) -> str:
    """经 _StreamingMarkdown 渲染并导出带 ANSI 样式的文本，供着色断言。"""
    console = Console(record=True, width=120, force_terminal=True, color_system="truecolor")
    streamer = _StreamingMarkdown(console)
    streamer.feed(text)
    streamer.finish()
    return console.export_text(styles=True)


class TestSeverityRendering:
    def test_severity_colored_and_suggestion_directly_below(self) -> None:
        out = _render_segment("【严重】a.py:1 — `x=1` 会崩溃。\n修改建议：改为 `x=2`。")

        assert "\x1b[1;31m【严重】" in out  # 红色加粗
        assert "\x1b[1m修改建议：" in out  # 加粗
        lines = [line.rstrip() for line in out.splitlines()]
        issue_index = next(i for i, line in enumerate(lines) if "【严重】" in line)
        assert "修改建议：" in lines[issue_index + 1]  # 紧跟问题行，无空行
        assert "`x=1`" not in out  # 文本片仍经 Markdown 渲染，行内代码反引号不字面出现

    def test_all_three_severities_colored(self) -> None:
        out = _render_segment("【一般】b.py:2 — y。\n\n【建议】c.py:3 — z。")

        assert "\x1b[1;33m【一般】" in out  # 黄色
        assert "\x1b[1;32m【建议】" in out  # 绿色

    def test_marker_after_cjk_letter_styled_without_literal_asterisks(self) -> None:
        """回归（M2 修复轮）：中文后紧跟的标记也必须加粗——CommonMark flanking
        规则会漏掉 `中文**【标记】**中文`，故渲染不依赖 ** 插入。"""
        out = _render_segment("前文【严重】中缀出现。")

        assert "\x1b[1;31m【严重】" in out
        assert "**" not in out

    def test_model_self_bold_wrapping_consumed(self) -> None:
        out = _render_segment("**【一般】**x.py:1 — y")

        assert "**" not in out  # 模型自带的星号只被消耗，不字面出现
        assert "\x1b[1;33m【一般】" in out

    def test_inline_suggestion_bolded(self) -> None:
        out = _render_segment("问题一。修改建议：改。")

        assert "\x1b[1m修改建议：" in out

    def test_marker_inside_fence_not_colored(self) -> None:
        out = _render_segment("```\n【严重】围栏内不着色\n```")

        assert "\x1b[1;31m" not in out
        assert "【严重】围栏内不着色" in out

    def test_plain_paragraph_still_markdown_rendered(self) -> None:
        out = _render_segment("普通段落 `code` 文本")

        assert "普通段落" in out
        assert "`code`" not in out


# ---- M5:REPL 命令族(§7.3)与全局参数(§7.2)----

# 测试占位值(明显假值,非真实凭据);以变量构造避免密钥扫描对字面量的误报
_PLACEHOLDER_KEY = "placeholder-a"


def _write_models_json(home: Any, data: dict) -> None:
    path = home / ".cra" / "models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _provider_models_json() -> dict:
    """单一供应商的配置样例;凭据字段以变量赋值(占位假值)。"""
    entry: dict[str, Any] = {
        "name": "prov",
        "base_url": "https://p.example.com",
        "models": ["m-one", "m-two"],
    }
    entry["api_key"] = _PLACEHOLDER_KEY
    return {"providers": [entry]}


class TestReplCommands:
    """§7.3 内置命令族:/help /save /load /undo /context /export /confirm /model。"""

    def _repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: list[str],
    ) -> str:
        monkeypatch.setenv("SE_CodeAgent", "sk-test")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        iterator = iter(inputs)
        # 命令族内部有带提示语的 input("...") 调用,lambda 须接受可选位置参数
        monkeypatch.setattr("builtins.input", lambda *a: next(iterator))
        cli._run_chat()
        return capsys.readouterr().out

    def test_help_lists_all_commands(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/help", "exit"])

        for keyword in (
            "/help", "/clear", "/model", "/save", "/load", "/undo", "/context", "/export", "/confirm",
        ):
            assert keyword in out

    def test_save_then_load_roundtrip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        out = self._repl(monkeypatch, capsys, ["/save mysession", "/load mysession", "exit"])

        session_file = tmp_path / ".cra" / "sessions" / "mysession.json"
        assert session_file.exists()
        assert "会话已保存" in out
        assert "会话已加载" in out

    def test_load_without_name_shows_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/load", "exit"])

        assert "用法" in out

    def test_save_invalid_name_rejected(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/save bad name", "exit"])

        assert "非法" in out

    def test_undo_reports_result(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/undo", "exit"])
        assert "已作废" in out

    def test_undo_without_turns_reports_nothing_to_revert(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_FakeAgent, "undo_default", False)
        out = self._repl(monkeypatch, capsys, ["/undo", "exit"])
        assert "没有可作废" in out

    def test_context_shows_usage_and_limit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/context", "exit"])

        assert "1234" in out and "100000" in out

    def test_export_writes_markdown_of_last_answer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Any,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        out = self._repl(monkeypatch, capsys, ["审查", "/export", "exit"])

        report = tmp_path / "cra-export.md"
        assert report.exists()
        assert "最终回答正文。" in report.read_text(encoding="utf-8")
        assert "已导出" in out

    def test_export_without_answer_hinted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)

        out = self._repl(monkeypatch, capsys, ["/export", "exit"])

        assert "尚无可导出" in out

    def test_confirm_toggle_and_query(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/confirm on", "/confirm", "exit"])

        assert "执行确认" in out
        assert "开" in out

    def test_confirm_bad_argument_shows_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/confirm maybe", "exit"])

        assert "用法" in out

    def test_model_without_providers_prompts_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        out = self._repl(monkeypatch, capsys, ["/model", "exit"])

        assert "cra config" in out

    def test_change_model_flow_selects_provider_then_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _write_models_json(tmp_path, _provider_models_json())

        out = self._repl(monkeypatch, capsys, ["/change model", "1", "2", "exit"])

        assert "已切换到 prov/m-two" in out
        assert _FakeAgent.last is not None
        assert len(_FakeAgent.last.switched) == 1

    def test_change_alias_model_word_required(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/change other", "exit"])

        assert "用法" in out

    def test_unknown_slash_sent_as_text(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._repl(monkeypatch, capsys, ["/notacommand 试试", "exit"])

        assert _FakeAgent.last is not None
        assert "/notacommand 试试" in _FakeAgent.last.turns


class TestCliArgs:
    """§7.1/§7.2:子命令与全局参数解析。"""

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _build_parser().parse_args(["--version"])

        assert excinfo.value.code == 0
        assert "cra" in capsys.readouterr().out

    def test_chat_accepts_global_options(self) -> None:
        args = _build_parser().parse_args(
            ["chat", "--model", "m", "--provider", "p", "--no-stream", "--max-rounds", "5", "--ask-exec"]
        )

        assert args.model == "m"
        assert args.provider == "p"
        assert args.no_stream is True
        assert args.max_rounds == 5
        assert args.ask_exec is True

    def test_review_accepts_paths_output_and_json(self) -> None:
        args = _build_parser().parse_args(
            ["review", "a.py", "b.py", "-o", "report.md", "--json"]
        )

        assert args.paths == ["a.py", "b.py"]
        assert args.output == "report.md"
        assert args.json is True

    def test_review_accepts_stdin_marker(self) -> None:
        args = _build_parser().parse_args(["review", "-"])

        assert args.paths == ["-"]

    def test_review_accepts_ask_exec_flag_for_explicit_error(self) -> None:
        """review 接受 --ask-exec 解析,由 main 以中文报错退出 2(§7.2)。"""
        args = _build_parser().parse_args(["review", "a.py", "--ask-exec"])

        assert args.ask_exec is True

    def test_config_subcommand_parses(self) -> None:
        assert _build_parser().parse_args(["config"]).command == "config"
        assert _build_parser().parse_args(["config", "test"]).config_action == "test"
