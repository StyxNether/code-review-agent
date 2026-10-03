"""cli 单元：参数解析、分级标记着色渲染与围栏状态机、REPL 循环与设置指引
（不产生真实终端输出，无网络）。"""

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from cra import cli, config, sessions
from cra.cli import (
    _build_parser,
    _fence_toggle,
    _marker_style,
    _print_config_guide,
    _setup_hint,
    _StreamingMarkdown,
)
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ERROR_PREFIX


class TestParser:
    """M2 补齐：CLI 参数解析此前无测试覆盖。"""

    def test_chat_subcommand_parses(self) -> None:
        args = _build_parser().parse_args(["chat"])

        assert args.command == "chat"

    def test_missing_subcommand_yields_none_for_bare_entry(self) -> None:
        """裸入口：子命令可选化，缺省 command=None 由 main 进入 chat REPL。"""
        args = _build_parser().parse_args([])

        assert args.command is None

    def test_bare_entry_defaults_match_chat_subcommand(self) -> None:
        """chat 子解析器的参数缺省值必须与 main 裸入口的 getattr 缺省一致，
        否则裸 cra 与 cra chat 行为静默分叉（main 注释所指的不变量）。"""
        from cra.agent import MAX_TOOL_ROUNDS

        chat = _build_parser().parse_args(["chat"])

        assert chat.model is None
        assert chat.provider is None
        assert chat.no_stream is False
        assert chat.max_rounds == MAX_TOOL_ROUNDS
        assert chat.ask_exec is False

    def test_missing_subcommand_enters_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """M7 语义反转：缺失子命令原期望报错退出，现期望默认进入 REPL（mock input 验证）。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr(sys, "argv", ["cra"])
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(inputs))

        cli.main()  # 不抛 SystemExit 即通过：裸入口进入 REPL 并正常退出

        out = capsys.readouterr().out
        assert "已就绪" in out
        assert "再见" in out

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["nope"])

    def test_chat_help_declares_non_sandbox_and_timeout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """非沙箱、限时执行的限制声明在 CLI 层面可见。"""
        with pytest.raises(SystemExit) as excinfo:
            _build_parser().parse_args(["chat", "--help"])

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "非沙箱" in out
        assert "限时" in out


class TestSetupGuide:
    """key 指引按平台给出；输出统一经 rich Console（不再混用 print）。

    无任何配置时的引导走 _print_config_guide（含 cra config 指引）。
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
        assert "CRA_DEEPSEEK_API_KEY" in out


class _FakeAgent:
    """替换真实 Agent:避免构造模型客户端;记录 /clear、对话轮次与命令族操作。

    run 按真实 Agent 的回调时序模拟:叙述增量 → 工具打点 → 最终回答增量
    (供过程叙述分流的行为断言)。
    """

    last: "_FakeAgent | None" = None
    undo_default = True  # 类属性:undo 测试经 monkeypatch 覆盖(每次构造实例会重建)

    def __init__(self, **_kwargs: Any) -> None:
        self.model = "m-one"
        self.provider = "prov"  # 当前供应商（/change model 的"当前使用中"标记数据源）
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
        on_tool_result = kwargs.get("on_tool_result")
        if on_delta is not None:
            on_delta("先看看文件再作答。")  # 工具轮伴随叙述(无段落空行)
        if on_tool is not None:
            on_tool("list_dir", "{}")
        if on_tool_result is not None:
            on_tool_result("目录内容")  # 真实 Agent 协议:结果回调必然触发
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
    """_run_chat 的 sys.exit 路径与 REPL 内置命令可测（mock input/Agent）。"""

    def test_missing_key_exits_with_guide(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """无任何配置打印 cra config 引导后退出 2。"""
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SE_CodeAgent", raising=False)

        with pytest.raises(SystemExit) as excinfo:
            cli._run_chat()

        assert excinfo.value.code == 2
        assert "cra config" in capsys.readouterr().out

    def test_exit_command_ends_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", lambda: "exit")

        cli._run_chat()

        assert "再见" in capsys.readouterr().out

    def test_eof_ends_repl_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)

        def raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)

        cli._run_chat()  # 不抛异常即通过

        assert "再见" in capsys.readouterr().out

    def test_clear_resets_agent_and_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        inputs = iter(["/clear", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))

        cli._run_chat()

        assert _FakeAgent.last is not None
        assert _FakeAgent.last.reset_calls == 1

    def test_empty_input_continues_without_api_call(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        inputs = iter(["", "   ", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))

        cli._run_chat()

        # 空输入/纯空白只重新提示，不发起对话轮次
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.turns == []
        assert capsys.readouterr().out.count("你>") == 3

    def test_narration_flushed_before_tool_mark_and_answer_separate(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """过程性叙述跟随工具轮实时暗色显示，不与最终回答堆叠。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
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
    """行内标记不再被 Markdown 片段自带换行拆成多行。"""
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
    """兼容端点不返回文本增量时，界面用返回值兜底而非空白。"""
    monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
    monkeypatch.setattr(cli, "Agent", _SilentAgent)
    inputs = iter(["问", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    cli._run_chat()

    assert "无增量完整回答。" in capsys.readouterr().out


class _PartialDeltaAgent:
    """只发一条不成单元的增量(无换行)就返回的假 Agent。"""

    def __init__(self, **_kwargs: Any) -> None:
        self.model = "test-model"

    def reset(self) -> None:
        pass

    def run(self, user_input: str, **kwargs: Any) -> str:
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta("简短回答。")
        return "简短回答。"


def test_partial_delta_not_duplicated_by_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M6 兜底判据修复:增量已收到(尚在缓冲)时不得再用返回值重复渲染。"""
    monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
    monkeypatch.setattr(cli, "Agent", _PartialDeltaAgent)
    inputs = iter(["问", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    cli._run_chat()

    assert capsys.readouterr().out.count("简短回答。") == 1


class _NarrateOnlyAgent:
    """工具轮有叙述增量、最终消息无增量的假 Agent(旧版端点行为)。"""

    def __init__(self, **_kwargs: Any) -> None:
        self.model = "test-model"

    def reset(self) -> None:
        pass

    def run(self, user_input: str, **kwargs: Any) -> str:
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta("让我看看。")
        if kwargs.get("on_tool") is not None:
            kwargs["on_tool"]("list_dir", "{}")
        if kwargs.get("on_tool_result") is not None:
            kwargs["on_tool_result"]("ok")
        return "最终结论。"


def test_narration_then_silent_final_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """叙述已冲洗、最终消息无增量:返回值兜底渲染最终回答(界面不空白)。"""
    monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
    monkeypatch.setattr(cli, "Agent", _NarrateOnlyAgent)
    inputs = iter(["问", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    cli._run_chat()

    out = capsys.readouterr().out
    assert "让我看看。" in out
    assert "最终结论。" in out


def test_system_prompt_guides_process_narration() -> None:
    """D16:思考过程展示的替代方案——提示词引导工具前意图说明与审查计划。"""
    assert "一句简短的话" in SYSTEM_PROMPT
    assert "审查计划" in SYSTEM_PROMPT


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


def test_fence_interior_held_until_closed() -> None:
    """M6 行级流式：围栏未闭合前内容不落盘（完整性保护），闭合后整组渲染。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("看代码：\n```py\nx = 1\n")
    assert "x = 1" not in console.export_text(clear=False)  # 未闭合：围栏内容扣住
    streamer.feed("print(x)\n```\n之后正文")
    out = console.export_text(clear=False)
    assert "x = 1" in out and "print(x)" in out  # 闭合后整组出现
    assert out.count("x = 1") == 1  # 只渲染一次，不重复


def test_tilde_fence_hold_and_interior_blank_lines() -> None:
    """~~~ 围栏与围栏内部空行同样扣住（原段落切分的围栏保护语义，行级等价）。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("铺垫。\n")  # 先占住分类窗口的一个单元
    streamer.feed("~~~py\nx\n\ny")
    assert "x" not in console.export_text(clear=False)  # 未闭合不落盘
    streamer.feed("\n~~~\n之后\n")
    out = console.export_text(clear=False)
    assert "x" in out and "y" in out and "之后" in out


def test_longer_fence_not_closed_by_inner_short_fence() -> None:
    """```` 围栏内部的 ``` 不闭合围栏。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("````md\nx\n\n```\n````\n\nb\n")
    out = console.export_text(clear=False)
    assert "x" in out and "b" in out
    assert out.count("x") == 1  # 内部 ``` 未提前闭合围栏


def test_fence_after_blank_line_in_single_feed() -> None:
    """空行消费使缓冲起点前移后，围栏开栏偏移 limit 须同步平移，
    否则一次投喂（--no-stream 与返回值兜底路径的形态）时围栏被拆成普通行。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("前言。\n\n```python\nprint('hi')\n```\n")

    out = console.export_text(clear=False)
    assert "前言。" in out
    assert "```" not in out  # 围栏标记不字面出现（整组经 Markdown 代码块渲染）
    assert "print('hi')" in out


def test_fence_toggle_state() -> None:
    assert _fence_toggle("a", None) is None
    state = _fence_toggle("```py", None)
    assert state == ("`", 3)
    assert _fence_toggle("x", state) == state  # 围栏内普通行不改变状态
    assert _fence_toggle("```", state) is None  # 同字符闭围栏
    state2 = _fence_toggle("~~~", None)
    assert _fence_toggle("```", state2) == state2  # 字符不同的围栏不互相闭合


# ---- 行级单元流式、句级早发、暂扣分类与分层样式 ----


def _stream_console() -> Console:
    return Console(record=True, width=120, force_terminal=True, color_system="truecolor")


def test_complete_lines_render_before_finish() -> None:
    """行级粒度:完整行即时落盘,不等 finish。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("第一行。\n第二行。\n")

    out = console.export_text(clear=False)
    assert "第一行。" in out and "第二行。" in out  # 未调 finish 已落盘

    streamer.finish()
    assert console.export_text(clear=False).count("第一行。") == 1  # finish 不重复渲染


def test_first_unit_held_until_second_arrives() -> None:
    """分类窗口(D15):边界后的首个单元暂扣,防止单行过程叙述以正文样式抢先显示。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("叙述单行。\n")
    assert "叙述单行。" not in console.export_text(clear=False)  # 暂扣中

    streamer.feed("第二行。\n")
    out = console.export_text(clear=False)
    assert "叙述单行。" in out and "第二行。" in out  # 第二单元到达即解除暂扣


def test_sentence_flush_on_long_prose_line() -> None:
    """句级早发:无结构长行超过阈值后在句界落盘,不等换行。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("先起个头。\n再铺垫一句。\n")  # 两个单元解除暂扣
    streamer.feed("这一段没有换行但是" + "很长" * 40 + "的一句话。后半句还没写完")

    assert "的一句话。" in console.export_text(clear=False)  # 句界已提前落盘
    streamer.finish()
    assert "后半句还没写完" in console.export_text(clear=False)


def test_sentence_flush_skips_inline_code_interior() -> None:
    """句界守卫:句读全在行内代码内部时不早发(`a.b` 的点不是句界)。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("先起个头。\n再铺垫一句。\n")
    streamer.feed("检查 `" + "x." * 40 + "` 结束")

    assert "x." not in console.export_text(clear=False)  # 句界全在代码内:不早发
    streamer.finish()
    assert "x." in console.export_text(clear=False)  # finish 整行渲染


def test_list_group_renders_without_item_gaps() -> None:
    """列表整组渲染:项与项之间无空行(逐项打印会被 rich 列表边距拆开,实证)。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("开头。\n- 甲\n- 乙\n\n后续。")

    lines = [line for line in console.export_text(clear=False).splitlines() if line.strip()]
    jia = next(i for i, line in enumerate(lines) if "甲" in line)
    yi = next(i for i, line in enumerate(lines) if "乙" in line)
    assert yi == jia + 1  # 两个列表项相邻


def test_table_group_renders_as_table() -> None:
    """表格整组渲染:出现 Markdown 表格制表线,而非逐行字面竖线。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("前文。\n| a | b |\n|---|---|\n| 1 | 2 |\n\n后文。")

    assert "─" in console.export_text(clear=False)  # rich 表格框线


def test_paragraph_blank_line_spacing_preserved() -> None:
    """段落间空行在行级流式下保持。"""
    console = _stream_console()
    streamer = _StreamingMarkdown(console)
    streamer.feed("段一。\n\n段二。\n")

    out = console.export_text()
    assert "\n\n" in out[out.index("段一。") : out.index("段二。")]


def test_tool_mark_and_error_result_styling() -> None:
    """三类内容分层样式:工具名 cyan 加粗、错误结果行红色、最终回答照常渲染。"""
    console = _stream_console()

    class _ToolAgent:
        def run(self, user_input: str, **kwargs: Any) -> str:
            if kwargs.get("on_tool") is not None:
                kwargs["on_tool"]("run_python", '{"code": "1/0"}')
            if kwargs.get("on_tool_result") is not None:
                kwargs["on_tool_result"](ERROR_PREFIX + "boom")
            if kwargs.get("on_delta") is not None:
                kwargs["on_delta"]("结论。")
            return "结论。"

    cli._run_turn(_ToolAgent(), console, "审查", no_stream=False, process_events=[])

    out = console.export_text(styles=True)
    assert "\x1b[1;36mrun_python" in out  # 工具名 bold cyan(与暗色叙述区分)
    assert "\x1b[31m" in out  # 错误结果行红色
    assert "结论。" in out


def test_process_events_recorded_for_replay() -> None:
    """_run_turn 就地记录过程事件(叙述/工具/结果),供 /process 重放。"""
    console = _stream_console()
    events: list[tuple[str, str]] = []

    class _ToolAgent:
        def run(self, user_input: str, **kwargs: Any) -> str:
            if kwargs.get("on_delta") is not None:
                kwargs["on_delta"]("让我看看。")
            if kwargs.get("on_tool") is not None:
                kwargs["on_tool"]("read_file", '{"path": "a.py"}')
            if kwargs.get("on_tool_result") is not None:
                kwargs["on_tool_result"]("内容")
            if kwargs.get("on_delta") is not None:
                kwargs["on_delta"]("结论。")
            return "结论。"

    cli._run_turn(_ToolAgent(), console, "审查", no_stream=False, process_events=events)

    kinds = [kind for kind, _ in events]
    assert kinds == ["narration", "tool", "result"]
    assert "让我看看。" in events[0][1]
    assert "read_file" in events[1][1]
    assert events[2][1] == "内容"  # 结果事件存内容预览(重放价值高于字数摘要)


class TestMarkerStyling:
    """分级标记红/黄/绿、修改建议加粗。"""

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
        """中文后紧跟的标记也必须加粗——CommonMark flanking
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


# ---- REPL 命令族与全局参数 ----

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
    """内置命令族:/help /save /load /undo /context /export /confirm /model。"""

    def _repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: list[str],
    ) -> str:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
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
            "/help", "/clear", "/model", "/save", "/load", "/undo", "/context", "/export",
            "/process", "/confirm",
        ):
            assert keyword in out

    def test_process_replays_last_turn_events(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """/process:最近一轮的叙述/工具/结果按序重放（紧凑打点的按需展开）。"""
        out = self._repl(monkeypatch, capsys, ["审查", "/process", "exit"])

        assert "过程明细" in out
        assert "先看看文件再作答。" in out  # 叙述全文
        assert "list_dir" in out  # 工具调用
        assert "↳" in out  # 结果行

    def test_process_without_history_hinted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/process", "exit"])

        assert "没有过程记录" in out

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

    def test_load_without_name_lists_sessions_and_loads_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """/load 省略名称列出会话数字选择；加载成功自动打印全部会话历史。"""
        seeded = [
            {"type": "human", "data": {"content": "早前的问题"}},
            {"type": "ai", "data": {"content": "早前的回答"}},
        ]
        sessions_dir = home_dir / ".cra" / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "alpha.json").write_text("[]", encoding="utf-8")
        (sessions_dir / "beta.json").write_text(json.dumps(seeded), encoding="utf-8")

        out = self._repl(monkeypatch, capsys, ["/load", "2", "exit"])

        assert "1. alpha" in out and "2. beta" in out
        assert "会话已加载" in out
        assert "你> 早前的问题" in out  # 加载后打印历史（你>/cra> 前缀逐条）
        assert "cra> 早前的回答" in out
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.loaded == [seeded]

    def test_load_without_name_cancel_returns_to_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """列表选择处回车取消，会话不变。"""
        sessions_dir = home_dir / ".cra" / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "alpha.json").write_text("[]", encoding="utf-8")

        out = self._repl(monkeypatch, capsys, ["/load", "", "exit"])

        assert "已取消" in out
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.loaded == []

    def test_load_without_any_sessions_hinted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """M8 语义变化：省略名称不再打印用法，而是提示没有已保存会话。"""
        out = self._repl(monkeypatch, capsys, ["/load", "exit"])

        assert "没有已保存的会话" in out

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

    def test_change_model_marks_current_provider_and_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Any,
    ) -> None:
        """清单标注"当前使用中"（_FakeAgent 当前为 prov/m-one）。"""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _write_models_json(tmp_path, _provider_models_json())

        out = self._repl(monkeypatch, capsys, ["/model", "1", "", "exit"])

        assert "（当前使用中）" in out
        assert "1. m-one（当前使用中）" in out
        assert out.count("（当前使用中）") == 2  # 供应商清单 1 处 + 模型清单 1 处

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
    """子命令与全局参数解析。"""

    def test_parse_index_rejects_zero_negative_and_non_ascii_digits(self) -> None:
        """0/负数/上标数字(isdigit 为 True 但 int 接受集不一致)全拒绝。"""
        assert cli._parse_index("0", 3) is None
        assert cli._parse_index("-1", 3) is None
        assert cli._parse_index("²", 3) is None  # isdigit()==True 但 int() 抛错
        assert cli._parse_index("9", 3) is None  # 越界
        assert cli._parse_index(None, 3) is None
        assert cli._parse_index("2", 3) == 1

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
        """review 接受 --ask-exec 解析,由 main 以中文报错退出 2。"""
        args = _build_parser().parse_args(["review", "a.py", "--ask-exec"])

        assert args.ask_exec is True

    def test_config_subcommand_parses(self) -> None:
        assert _build_parser().parse_args(["config"]).command == "config"
        assert _build_parser().parse_args(["config", "test"]).config_action == "test"


# ---- 裸入口、/setting 菜单、/exit、会话自动保留与 /resume ----


class _EmptyDumpAgent(_FakeAgent):
    """dump_session 为空的假 Agent:新开即退的会话(自动保留跳过空存档)。"""

    def dump_session(self) -> list[dict[str, Any]]:
        return []


def _feed(inputs: list[str]) -> Any:
    """input 打桩:按序返回;清单耗尽后抛 EOFError(模拟管道结束/ Ctrl+D)。"""
    iterator = iter(inputs)

    def fake_input(*_args: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return fake_input


def _seed_last(home: Path, data: Any) -> None:
    """预置 _last.json 存档(合法消息数组或任意坏内容)。"""
    path = home / ".cra" / "sessions" / "_last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestExitAndSetting:
    """/exit 等价退出与 /setting 菜单分发：菜单复用既有命令实现（mock 验证映射）。"""

    def _repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: list[str],
    ) -> str:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", _feed(inputs))
        cli._run_chat()
        return capsys.readouterr().out

    def test_slash_exit_ends_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """/exit 与 exit 等价退出。"""
        out = self._repl(monkeypatch, capsys, ["/exit"])

        assert "再见" in out

    def test_setting_menu_dispatches_to_existing_implementations(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """菜单 1–6 逐项分发到既有命令函数（mock 记录映射），操作后回到菜单，0 返回对话。"""
        calls: list[Any] = []
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr(cli, "_cmd_change_model", lambda state: calls.append("model"))
        monkeypatch.setattr(
            cli, "_config_interactive_session", lambda console, data: calls.append("config")
        )
        monkeypatch.setattr(cli, "_cmd_save", lambda state, name: calls.append(("save", name)))
        monkeypatch.setattr(cli, "_cmd_load", lambda state, name: calls.append(("load", name)))
        monkeypatch.setattr(cli, "_cmd_confirm", lambda state, arg: calls.append(("confirm", arg)))
        monkeypatch.setattr(cli, "_cmd_delete", lambda state: calls.append("delete"))
        monkeypatch.setattr(
            "builtins.input",
            _feed(["/setting", "1", "2", "3", "s1", "4", "s2", "5", "on", "6", "0", "exit"]),
        )

        cli._run_chat()
        out = capsys.readouterr().out

        assert calls == [
            "model", "config", ("save", "s1"), ("load", "s2"), ("confirm", "on"), "delete"
        ]
        # 初始菜单 + 每个操作后各回一次菜单（6 项操作 → 共 7 次显示）
        assert out.count("设置菜单") == 7

    def test_setting_empty_choice_returns_to_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """回车（空选择）返回对话，REPL 主循环继续。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "", "exit"])

        assert out.count("设置菜单") == 1
        assert "再见" in out

    def test_setting_invalid_choice_stays_in_menu(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """无效选择提示后回到菜单（嵌套循环：任何失败/取消回到菜单）。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "9", "0", "exit"])

        assert "无效选择" in out
        assert out.count("设置菜单") == 2

    def test_setting_config_corruption_returns_to_menu(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """会话进行中配置损坏（ConfigError）：菜单内降级为回菜单（可读提示），会话存活。

        patch _config_interactive_session 抛 ConfigError——启动阶段的 load_raw
        不受影响，模拟"REPL 打开后配置才损坏"的真实场景。
        """
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)

        def broken(_console: Any, _data: dict[str, Any]) -> None:
            raise config.ConfigError("models.json 不是 JSON 对象")

        monkeypatch.setattr(cli, "_config_interactive_session", broken)
        monkeypatch.setattr("builtins.input", _feed(["/setting", "2", "0", "exit"]))

        cli._run_chat()
        out = capsys.readouterr().out

        assert "配置文件损坏" in out
        assert "回到设置菜单" in out
        assert "再见" in out  # 菜单退出后 REPL 存活至正常退出

    def test_setting_save_uses_real_save(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """菜单 3) 保存会话走真实 /save 路径：输入名称后写盘。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "3", "menu-save", "0", "exit"])

        assert (home_dir / ".cra" / "sessions" / "menu-save.json").exists()
        assert "会话已保存" in out

    def test_setting_save_empty_name_uses_timestamp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """菜单 3) 回车自动以时间戳命名保存（%Y%m%d-%H%M%S）。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "3", "", "0", "exit"])

        names = [p.stem for p in (home_dir / ".cra" / "sessions").glob("*.json")]
        assert any(re.fullmatch(r"\d{8}-\d{6}", name) for name in names)
        assert "会话已保存" in out

    def test_setting_load_empty_name_lists_sessions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """菜单 4) 回车列出会话数字选择（_cmd_load 空名分支），取消回菜单。"""
        sessions_dir = home_dir / ".cra" / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "alpha.json").write_text("[]", encoding="utf-8")

        out = self._repl(monkeypatch, capsys, ["/setting", "4", "", "", "0", "exit"])

        assert "1. alpha" in out
        assert "已取消" in out
        assert out.count("设置菜单") == 2  # 取消后回到菜单，0 退出菜单

    def test_setting_load_missing_file_returns_to_menu(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """菜单 4) 加载不存在的会话：报错后回到菜单（不终止菜单循环）。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "4", "ghost", "0", "exit"])

        assert "会话文件不存在" in out
        assert out.count("设置菜单") == 2

    def test_setting_confirm_toggle(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """菜单 5) 执行确认开关复用 /confirm：状态行显示开启（断言精确文案，菜单回显不含状态）。"""
        out = self._repl(monkeypatch, capsys, ["/setting", "5", "on", "0", "exit"])

        assert "run_python 执行确认：开" in out

    def test_setting_eof_returns_to_repl_then_exits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """菜单内 EOF 经 _ask 归一为返回对话，REPL 主循环随后同样 EOF 优雅退出。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", _feed(["/setting"]))

        cli._run_chat()  # 不抛异常即通过

        out = capsys.readouterr().out
        assert "设置菜单" in out
        assert "再见" in out


class TestSessionAutosave:
    """会话自动保留与 /resume：退出写 _last、启动提示、/resume 恢复。"""

    @staticmethod
    def _last_path(home_dir: Path) -> Path:
        return home_dir / ".cra" / "sessions" / "_last.json"

    def _repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: list[str],
        agent_cls: Any = _FakeAgent,
    ) -> str:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", agent_cls)
        monkeypatch.setattr("builtins.input", _feed(inputs))
        cli._run_chat()
        return capsys.readouterr().out

    def _run_and_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
        inputs: list[str],
        agent_cls: Any = _FakeAgent,
    ) -> str:
        """跑一轮 REPL 并断言自动保留提示，返回 _last.json 内容。"""
        out = self._repl(monkeypatch, capsys, inputs, agent_cls)
        assert "会话已自动保留" in out
        return self._last_path(home_dir).read_text(encoding="utf-8")

    def test_exit_autosaves_last_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        data = json.loads(
            self._run_and_read(monkeypatch, capsys, home_dir, ["审查 x", "exit"])
        )
        assert len(data) == 1  # _FakeAgent.dump_session 的单条对话消息

    def test_eof_autosaves_last_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """EOF 退出同样触发自动保留（任何方式退出）。"""
        data = json.loads(
            self._run_and_read(monkeypatch, capsys, home_dir, ["审查 x"])
        )
        assert len(data) == 1

    def test_slash_exit_autosaves_last_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        data = json.loads(
            self._run_and_read(monkeypatch, capsys, home_dir, ["审查 x", "/exit"])
        )
        assert len(data) == 1

    def test_empty_session_skips_autosave_keeps_previous(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """空会话不覆盖既有存档：新开即退不该丢掉上一次的自动保留。"""
        previous = [{"type": "human", "data": {"content": "上一会话"}}]
        _seed_last(home_dir, previous)

        out = self._repl(monkeypatch, capsys, ["exit"], agent_cls=_EmptyDumpAgent)

        assert "会话已自动保留" not in out
        assert json.loads(self._last_path(home_dir).read_text(encoding="utf-8")) == previous

    def test_autosave_failure_does_not_block_exit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """写盘失败只提示、不阻断退出（失败仅提示不阻断退出）。"""

        def boom(_path: Any, _data: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(cli.sessions, "save_session", boom)

        out = self._repl(monkeypatch, capsys, ["审查 x", "exit"])

        assert "再见" in out
        assert "自动保留失败" in out

    def test_startup_hint_on_nonempty_archive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        _seed_last(home_dir, [{"type": "human", "data": {"content": "上次"}}])

        out = self._repl(monkeypatch, capsys, ["exit"])

        assert "检测到上次会话存档" in out
        assert "/resume" in out

    @pytest.mark.parametrize("seeded", [None, "[]", "{broken", "{}", "[1,2]"])
    def test_startup_no_hint_without_usable_archive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
        seeded: str | None,
    ) -> None:
        """无存档 / 空存档 / 损坏与结构非法存档都不提示（视同无存档，/resume 时明确报错）。"""
        if seeded is not None:
            _seed_last(home_dir, seeded)

        out = self._repl(monkeypatch, capsys, ["exit"])

        assert "检测到上次会话存档" not in out

    def test_resume_loads_last_archive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        seeded = [{"type": "human", "data": {"content": "上次对话"}}]
        _seed_last(home_dir, seeded)

        out = self._repl(monkeypatch, capsys, ["/resume", "exit"])

        assert "会话已加载" in out
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.loaded == [seeded]

    def test_resume_corrupt_archive_reports_and_keeps_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """结构自检失败提示存档不可用，正本不变（agent.load_session 未被调用）。"""
        _seed_last(home_dir, "{broken")

        out = self._repl(monkeypatch, capsys, ["/resume", "exit"])

        assert "会话加载失败" in out
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.loaded == []

    def test_session_path_accepts_leading_underscore(self, home_dir: Path) -> None:
        """_last 存档名（下划线开头）在 sessions 层合法。"""
        assert sessions.session_path("_last").name == "_last.json"


# ---- 首次配置引导、/delete 会话删除 ----


class TestFirstRunSetup:
    """首次配置引导：确认 → 引导式配置 → 进入 REPL；拒绝 → 指引 + 退出 2。"""

    def test_decline_prints_guide_and_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """明确拒绝：维持既有指引 + 退出码 2（拒绝后仍无配置）。"""
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SE_CodeAgent", raising=False)
        monkeypatch.setattr("builtins.input", _feed(["n"]))

        with pytest.raises(SystemExit) as excinfo:
            cli._run_chat()

        assert excinfo.value.code == 2
        out = capsys.readouterr().out
        assert "已跳过配置" in out  # 询问被拒绝（mock input 下提示语不进 stdout）
        assert "cra config" in out  # 既有指引维持

    def test_confirm_wizard_configures_and_enters_repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        home_dir: Path,
    ) -> None:
        """DoD 演示路径：Y → DeepSeek 预设 → 回车 base_url → 回车预填 → key →
        环境变量存储 → 自动设默认 → 进入 REPL。"""
        # 真实环境若已有上述变量会触发覆盖确认或使 resolve 提前成功，导致输入序列错位
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SE_CodeAgent", raising=False)
        run_calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
            run_calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr(cli.sys, "platform", "win32")  # 断言 setx 分支，与平台解耦
        monkeypatch.setattr(cli.os, "environ", dict(os.environ))
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        key_iter = iter(["placeholder-wizard-key"])
        monkeypatch.setattr(cli.getpass, "getpass", lambda _p: next(key_iter))
        # 确认(回车=Y) → 预设 2(DeepSeek) → base_url 回车 → 预填回车 → 存储 2(环境变量)
        # → 设默认：供应商 1 → 模型 1 → REPL exit
        monkeypatch.setattr(
            "builtins.input", _feed(["", "2", "", "", "2", "1", "1", "exit"])
        )

        cli._run_chat()

        out = capsys.readouterr().out
        assert "已就绪" in out  # 配置完成后进入 REPL
        data = config.load_raw()
        assert data["default"] == "DeepSeek/deepseek-flash"  # 引导式配置已设默认
        entry = data["providers"][0]
        assert entry["name"] == "DeepSeek"
        assert "api_key" not in entry  # 环境变量存储：models.json 不落明文
        assert run_calls[0][:2] == ["setx", "CRA_DEEPSEEK_API_KEY"]
        assert os.environ.get("CRA_DEEPSEEK_API_KEY") == "placeholder-wizard-key"


class TestDeleteSession:
    """/delete：列表数字选择 + y/N 确认；_last 同样可删。"""

    def _repl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: list[str],
    ) -> str:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", _feed(inputs))
        cli._run_chat()
        return capsys.readouterr().out

    def _seed(self, home_dir: Path, *names: str) -> None:
        directory = home_dir / ".cra" / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / f"{name}.json").write_text("[]", encoding="utf-8")

    def test_delete_confirm_removes_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        self._seed(home_dir, "alpha", "beta")

        out = self._repl(monkeypatch, capsys, ["/delete", "1", "y", "exit"])

        assert "会话已删除" in out
        assert not (home_dir / ".cra" / "sessions" / "alpha.json").exists()
        assert (home_dir / ".cra" / "sessions" / "beta.json").exists()

    def test_delete_decline_keeps_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        self._seed(home_dir, "alpha")

        out = self._repl(monkeypatch, capsys, ["/delete", "1", "n", "exit"])

        assert "已取消" in out
        assert (home_dir / ".cra" / "sessions" / "alpha.json").exists()

    def test_delete_last_archive_allowed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_dir: Path
    ) -> None:
        """_last（自动保留存档）同样可删；退出时自动保留以当前会话重建同名存档。"""
        self._seed(home_dir, "_last")

        out = self._repl(monkeypatch, capsys, ["/delete", "1", "y", "exit"])

        assert "会话已删除" in out
        recreated = json.loads(
            (home_dir / ".cra" / "sessions" / "_last.json").read_text(encoding="utf-8")
        )
        assert recreated == [{"type": "human", "data": {"content": "审查 x.py"}}]  # 全新自动保留

    def test_delete_without_sessions_hinted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self._repl(monkeypatch, capsys, ["/delete", "exit"])

        assert "没有可删除的会话" in out


# ---- 交付前终审轮回归：输入健壮性、命令容错、输出编码与状态一致性 ----


class TestReplInputRobustness:
    """stdin 不可用与命令容错：输入路径异常一律干净退出/取消，不 traceback。"""

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], input_stub: Any
    ) -> str:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setattr(cli, "Agent", _FakeAgent)
        monkeypatch.setattr("builtins.input", input_stub)
        cli._run_chat()
        return capsys.readouterr().out

    def test_closed_stdin_runtime_error_ends_repl(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stdin 被整体关闭（fd 0<&-、无 stdin 启动）时 CPython 抛 RuntimeError：与 EOF 同义退出。"""

        def raise_lost_stdin() -> str:
            raise RuntimeError("lost sys.stdin")

        out = self._run(monkeypatch, capsys, raise_lost_stdin)

        assert "再见" in out

    def test_exit_with_trailing_argument_exits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """/exit 带参数按命令意图退出，不作为普通文本发给模型。"""
        out = self._run(monkeypatch, capsys, _feed(["/exit now"]))

        assert "再见" in out
        assert _FakeAgent.last is not None
        assert _FakeAgent.last.turns == []  # 模型未收到任何消息

    def test_change_model_uppercase_argument_routed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """/CHANGE MODEL 参数大小写不敏感：走切换流程而非打印用法提示。"""
        out = self._run(monkeypatch, capsys, _feed(["/CHANGE MODEL", "exit"]))

        assert "用法：" not in out


class TestLoadClearsLastRoundState:
    def test_load_replaces_last_answer_and_process(self, home_dir: Path) -> None:
        """加载会话整体替换后，"最近一轮结论/过程"一并失效（存档不含它们）。"""
        state = cli._ReplState(
            agent=_FakeAgent(),
            console=Console(file=io.StringIO()),
            no_stream=True,
            confirm_flag={"on": False},
            last_answer="加载前的旧结论",
            last_process=[("n", "旧过程")],
        )
        cli.sessions.save_session(cli.sessions.session_path("t1"), [])

        cli._cmd_load(state, "t1")

        assert state.last_answer is None
        assert state.last_process == []


class TestForceUtf8Stdio:
    def test_reconfigures_text_wrapper_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp936")
        err = io.TextIOWrapper(io.BytesIO(), encoding="cp936")
        monkeypatch.setattr(cli.sys, "stdout", out)
        monkeypatch.setattr(cli.sys, "stderr", err)

        cli._force_utf8_stdio()

        assert out.encoding == "utf-8"
        assert err.encoding == "utf-8"

    def test_non_wrapper_streams_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = io.StringIO()
        monkeypatch.setattr(cli.sys, "stdout", fake)
        monkeypatch.setattr(cli.sys, "stderr", fake)

        cli._force_utf8_stdio()  # 不抛异常即通过（测试桩等非常规流直接跳过）
