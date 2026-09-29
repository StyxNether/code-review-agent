"""cli 单元：参数解析、分级标记着色渲染与围栏状态机（不产生终端输出，无网络）。"""

import pytest
from rich.console import Console

from cra.cli import _build_parser, _in_fence, _marker_style, _StreamingMarkdown


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
