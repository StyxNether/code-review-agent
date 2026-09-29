"""cli 单元：参数解析、分级标记加粗与围栏状态机（不产生终端输出，无网络）。"""

import pytest
from rich.console import Console

from cra.cli import _build_parser, _emphasize_grade_markers, _in_fence, _StreamingMarkdown


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

    result = _emphasize_grade_markers(text)

    assert "**【一般】**" in result
    assert "**【建议】**" in result


def test_markers_inside_code_fence_untouched() -> None:
    """回归：围栏代码块内插入 ** 会以字面文本渲染，必须跳过。"""
    text = "正文【严重】。\n```\n【严重】不要加粗\n```\n尾部【建议】"

    result = _emphasize_grade_markers(text)

    assert "**【严重】**" in result  # 正文加粗
    assert "【严重】不要加粗" in result  # 围栏内保持原样
    assert "**【建议】**" in result


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
