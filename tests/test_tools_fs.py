"""fs 工具行为与限制：tmp_path 造临时文件，无网络。"""

from pathlib import Path

import pytest

from cra.tools import fs


class TestListDir:
    def test_two_level_tree_with_ignored_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
        (tmp_path / "sub" / "deep").mkdir()
        (tmp_path / "sub" / "deep" / "c.py").write_text("z = 3\n", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".git").mkdir()
        (tmp_path / ".hidden").mkdir()

        result = fs.list_dir(str(tmp_path))

        assert "a.py" in result
        assert "sub/" in result
        assert "b.py" in result
        assert "deep/" in result  # 第二层目录名可见
        assert "c.py" not in result  # 第三层不展开
        assert "__pycache__" not in result
        assert ".git" not in result
        assert ".hidden" not in result

    def test_relative_path_resolved_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "hello.py").write_text("1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert "hello.py" in fs.list_dir(".")

    def test_missing_path_structured_error(self, tmp_path: Path) -> None:
        result = fs.list_dir(str(tmp_path / "nope"))
        assert result.startswith("错误")
        assert "不存在" in result

    def test_file_path_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "a.py"
        target.write_text("x\n", encoding="utf-8")
        assert fs.list_dir(str(target)).startswith("错误")


class TestReadFile:
    def test_line_numbers_and_content(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("line1\nline2\nline3\n", encoding="utf-8")

        result = fs.read_file(str(target))

        assert "共 3 行" in result
        assert "1 | line1" in result
        assert "3 | line3" in result

    def test_range_is_1based_closed(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), start=3, end=5)

        assert "3 | line3" in result
        assert "5 | line5" in result
        assert "line2" not in result
        assert "line6" not in result

    def test_default_read_truncated_at_400_with_hint(self, tmp_path: Path) -> None:
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target))

        assert "400 | line400" in result
        assert "line401" not in result
        assert "start=401" in result  # 提示模型用 start/end 分段

    def test_explicit_range_exactly_at_limit_ok_with_hint(self, tmp_path: Path) -> None:
        """分段边界：显式区间恰好 400 行合法（上限只拒绝超出）。"""
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), start=1, end=400)

        assert not result.startswith("错误")
        assert "1 | line1" in result and "400 | line400" in result
        assert "start=401" in result

    def test_second_segment_continues_exactly(self, tmp_path: Path) -> None:
        """分段边界：续段 401-800 行首尾精确，再提示下一段起点。"""
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), start=401, end=800)

        assert "401 | line401" in result
        assert "800 | line800" in result
        assert "line400" not in result and "line801" not in result
        assert "start=801" in result

    def test_last_segment_no_truncation_hint(self, tmp_path: Path) -> None:
        """分段边界：末段到 EOF 不再提示继续分段。"""
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), start=801)

        assert "801 | line801" in result
        assert "1000 | line1000" in result
        assert "已按单次" not in result

    def test_end_without_start_reads_prefix(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), end=2)

        assert "1 | line1" in result
        assert "2 | line2" in result
        assert "line3" not in result

    def test_end_zero_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a\nb\n", encoding="utf-8")

        result = fs.read_file(str(target), end=0)

        assert result.startswith("错误")
        assert "1-based 正整数" in result

    def test_explicit_range_over_limit_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n", encoding="utf-8")

        result = fs.read_file(str(target), start=1, end=401)

        assert result.startswith("错误")
        assert "400" in result

    def test_start_greater_than_end_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a\nb\n", encoding="utf-8")

        result = fs.read_file(str(target), start=2, end=1)

        assert result.startswith("错误")
        assert "start" in result

    def test_non_positive_start_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a\nb\n", encoding="utf-8")
        assert fs.read_file(str(target), start=0).startswith("错误")
        assert fs.read_file(str(target), start=-1, end=1).startswith("错误")

    def test_start_beyond_eof_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a\nb\n", encoding="utf-8")

        result = fs.read_file(str(target), start=10)

        assert result.startswith("错误")
        assert "10" in result

    def test_end_clamped_to_eof(self, tmp_path: Path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a\nb\n", encoding="utf-8")

        result = fs.read_file(str(target), start=2, end=100)

        assert "2 | b" in result
        assert "显示 2-2 行" in result

    def test_binary_file_structured_error(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"\x00\x01\x02PNG\xff\xfe")

        result = fs.read_file(str(target))

        assert result.startswith("错误")
        assert "二进制" in result

    def test_empty_file_reports_empty_not_range_error(self, tmp_path: Path) -> None:
        """空文件按区间规则会误报 start>end，应提示空文件。"""
        target = tmp_path / "empty.py"
        target.touch()

        result = fs.read_file(str(target))

        assert "空文件" in result
        assert not result.startswith("错误")

    def test_invalid_utf8_replaced_instead_of_crash(self, tmp_path: Path) -> None:
        target = tmp_path / "gbk.py"
        target.write_bytes("中文注释\n".encode("gbk"))  # 无 null 字节，不算二进制

        result = fs.read_file(str(target))

        assert "1 |" in result  # 乱码经 errors="replace" 呈现，不抛异常

    def test_utf8_chinese_content(self, tmp_path: Path) -> None:
        target = tmp_path / "zh.py"
        target.write_text("def f():\n    return '你好'\n", encoding="utf-8")

        assert "你好" in fs.read_file(str(target))

    def test_missing_file_structured_error(self, tmp_path: Path) -> None:
        assert fs.read_file(str(tmp_path / "nope.py")).startswith("错误")

    def test_directory_path_structured_error(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        assert fs.read_file(str(tmp_path / "pkg")).startswith("错误")

    def test_oversized_file_structured_error(self, tmp_path: Path) -> None:
        """超过单文件读取上限（10 MB）返回结构化错误，不整文件进内存。"""
        target = tmp_path / "big.py"
        target.write_bytes(b"\n" * (10 * 1024 * 1024 + 1))

        out = fs.read_file(str(target))

        assert out.startswith("错误：")
        assert "10 MB" in out


class TestSplitLines:
    def test_only_real_newlines_split(self) -> None:
        assert fs.split_lines("a\nb") == ["a", "b"]
        assert fs.split_lines("a\r\nb") == ["a", "b"]
        assert fs.split_lines("a\rb") == ["a", "b"]
        assert fs.split_lines("") == []
        assert fs.split_lines("a\n") == ["a"]
        assert fs.split_lines("a\n\n") == ["a", ""]

    def test_unicode_line_separators_do_not_split(self) -> None:
        """\x0c（form feed）与 U+2028 不是行界：splitlines 会错误断行。"""
        assert fs.split_lines("a\x0cb") == ["a\x0cb"]
        assert fs.split_lines("ab") == ["ab"]

    def test_read_file_line_numbers_ignore_form_feed(self, tmp_path: Path) -> None:
        """含 form feed 的文件行号与编辑器口径一致（splitlines 会整体漂移）。"""
        target = tmp_path / "ff.py"
        target.write_text("first\n\x0cmid\nlast\n", encoding="utf-8")

        out = fs.read_file(str(target))

        assert "   1 | first" in out
        assert "   2 | \x0cmid" in out
        assert "   3 | last" in out
        assert "共 3 行" in out
