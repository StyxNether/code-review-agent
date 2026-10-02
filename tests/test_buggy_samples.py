"""边界样例黄金回归：fixtures/buggy_samples 每个样例经工具处理
均不崩且有可操作反馈——死循环超时被杀、note.txt 得到非 Python 提示、
binary.bin 由 read_file 兜底、empty.py 正常退出/空文件提示。

只走工具层（真实子进程，无 LLM/网络）；超时用例注入 timeout=1 不拖慢套件。
"""

import json
from pathlib import Path

import pytest

from cra.tools import exec_py, fs, get_default_registry

SAMPLES = Path(__file__).parent / "fixtures" / "buggy_samples"


def test_sample_set_complete() -> None:
    """fixtures 即黄金回归清单：样例缺失视为仓库损坏。

    只统计文件：安全钩子等工具可能在该目录留下隐藏状态目录（如 .mimosa）。
    """
    assert {item.name for item in SAMPLES.iterdir() if item.is_file()} == {
        "empty.py",
        "syntax_error.py",
        "infinite_loop.py",
        "note.txt",
        "binary.bin",
    }


class TestEmptyPy:
    def test_run_python_exits_cleanly(self) -> None:
        result = exec_py.run_python(path=str(SAMPLES / "empty.py"))

        assert "退出码：0" in result

    def test_read_file_reports_empty_file(self) -> None:
        result = fs.read_file(str(SAMPLES / "empty.py"))

        assert "空文件" in result
        assert not result.startswith("错误")


class TestSyntaxErrorPy:
    def test_run_python_returns_stderr(self) -> None:
        result = exec_py.run_python(path=str(SAMPLES / "syntax_error.py"))

        assert "退出码：1" in result
        assert "SyntaxError" in result


class TestInfiniteLoopPy:
    def test_timeout_raises_structured_error(self) -> None:
        with pytest.raises(RuntimeError, match="超时"):
            exec_py.run_python(path=str(SAMPLES / "infinite_loop.py"), timeout=1)

    def test_timeout_via_dispatch_returns_error_text(self) -> None:
        result = get_default_registry().dispatch(
            "run_python",
            json.dumps({"path": str(SAMPLES / "infinite_loop.py"), "timeout": 1}),
        )

        assert result.startswith("错误：")
        assert "超时" in result


class TestNoteTxt:
    def test_run_python_gives_friendly_hint(self) -> None:
        result = exec_py.run_python(path=str(SAMPLES / "note.txt"))

        assert result.startswith("错误")
        assert "不是 Python 文件" in result

    def test_read_file_still_readable(self) -> None:
        result = fs.read_file(str(SAMPLES / "note.txt"))

        assert "普通笔记" in result


class TestBinaryBin:
    def test_read_file_binary_fallback(self) -> None:
        result = fs.read_file(str(SAMPLES / "binary.bin"))

        assert result.startswith("错误")
        assert "二进制" in result

    def test_run_python_rejects_non_python(self) -> None:
        result = exec_py.run_python(path=str(SAMPLES / "binary.bin"))

        assert "不是 Python 文件" in result
