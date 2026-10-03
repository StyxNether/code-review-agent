"""cra review 测试:文件展开、prompt 构造、JSON schema 解析校验与
退出码判定。Agent 一律 mock,无网络。"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from cra import cli
from cra.llm import LLMError


def _review_args(paths: list[str], **kwargs: Any) -> argparse.Namespace:
    """构造 _run_review 的 args namespace(与 parser 默认一致)。"""
    defaults: dict[str, Any] = {
        "model": None,
        "provider": None,
        "no_stream": False,
        "max_rounds": 10,
        "output": None,
        "json": False,
        "ask_exec": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(paths=paths, **defaults)


class _FakeReviewAgent:
    """替换真实 Agent:按类属性 reply 回答;raised 非空时抛出。"""

    reply = "审查结论文本。"
    raised: Exception | None = None
    prompts: ClassVar[list[str]] = []
    resets: ClassVar[int] = 0

    def __init__(self, *, resolved: Any = None, max_tool_rounds: int = 10, **_kwargs: Any) -> None:
        assert resolved is not None, "review 路径必须携带解析结果"

    def reset(self) -> None:
        _FakeReviewAgent.resets += 1

    def run(self, prompt: str, **_kwargs: Any) -> str:
        _FakeReviewAgent.prompts.append(prompt)
        if _FakeReviewAgent.raised is not None:
            raise _FakeReviewAgent.raised
        return _FakeReviewAgent.reply


def _patch_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeReviewAgent.reply = "审查结论文本。"
    _FakeReviewAgent.prompts = []
    _FakeReviewAgent.raised = None
    _FakeReviewAgent.resets = 0
    monkeypatch.setattr(cli, "Agent", _FakeReviewAgent)


class TestCollectReviewFiles:
    def test_single_file(self, tmp_path: Path) -> None:
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        files = cli._collect_review_files([str(target)])

        assert files == [(str(target), "x = 1")]

    def test_directory_expands_first_level_text_files_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "b.py").write_text("b", encoding="utf-8")
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "note.txt").write_text("t", encoding="utf-8")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01binary")
        (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("c", encoding="utf-8")  # 第二层不展开

        files = cli._collect_review_files([str(tmp_path)])

        assert [name for name, _ in files] == [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
            str(tmp_path / "note.txt"),
        ]

    def test_directory_skips_only_hidden_when_all_binary(self, tmp_path: Path) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\x00\xff")
        (tmp_path / ".hidden").write_text("h", encoding="utf-8")

        with pytest.raises(cli._ReviewFileError, match="没有可审查的文本文件"):
            cli._collect_review_files([str(tmp_path)])

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(cli._ReviewFileError, match="没有可审查的文本文件"):
            cli._collect_review_files([str(tmp_path)])

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(cli._ReviewFileError, match="路径不存在"):
            cli._collect_review_files([str(tmp_path / "ghost.py")])

    def test_textual_sniff_io_failure_assumes_text(self, tmp_path: Path) -> None:
        """嗅探读取失败（如路径不可达）按文本处理：真实读取在后续步骤报可读错误，
        不把 IO 故障伪装成"二进制文件"。"""
        assert cli._looks_textual(tmp_path / "ghost.bin") is True

    def test_stdin_marker_reads_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("print('from stdin')"))

        files = cli._collect_review_files(["-"])

        assert files == [("<stdin>", "print('from stdin')")]


class TestReviewPrompt:
    def test_contains_name_numbered_lines_and_plain_spec(self) -> None:
        prompt = cli._review_prompt("a.py", "x = 1\ny = 2", json_mode=False)

        assert "a.py" in prompt
        assert "   1 | x = 1" in prompt
        assert "分级" in prompt
        assert "JSON" not in prompt

    def test_json_mode_embeds_schema(self) -> None:
        prompt = cli._review_prompt("a.py", "x = 1", json_mode=True)

        assert '"files"' in prompt
        assert '"severity"' in prompt
        assert "不要输出任何其他文字" in prompt

    def test_long_file_truncated_with_notice(self) -> None:
        content = "\n".join(f"line{i} = {i}" for i in range(cli._MAX_REVIEW_LINES + 50))

        prompt = cli._review_prompt("big.py", content, json_mode=False)

        assert f"仅含前 {cli._MAX_REVIEW_LINES} 行" in prompt
        assert f"line{cli._MAX_REVIEW_LINES + 49}" not in prompt  # 截断生效


def _finding(severity: str = "一般", **overrides: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "severity": severity,
        "location": "a.py:1",
        "issue": "问题",
        "suggestion": "建议",
    }
    finding.update(overrides)
    return finding


class TestParseJsonReport:
    def test_valid_report_normalizes_uncertain_default(self) -> None:
        data = {"files": [{"file": "a.py", "findings": [_finding()], "summary": "s"}]}

        parsed = cli._parse_json_report(json.dumps(data, ensure_ascii=False))

        assert parsed["files"][0]["findings"][0]["uncertain"] is False

    def test_strips_code_fence_and_surrounding_text(self) -> None:
        raw = "好的,结论如下:\n```json\n" + json.dumps(
            {"files": [{"file": "a.py", "findings": [], "summary": "s"}], "summary": "s"},
            ensure_ascii=False,
        ) + "\n```\n以上。"

        parsed = cli._parse_json_report(raw)

        assert parsed["summary"] == "s"

    def test_missing_top_level_summary_defaults_to_empty(self) -> None:
        """模型省略顶层 summary 时补全空串,输出满足完整 schema(下游可稳定消费)。"""
        raw = json.dumps({"files": [{"file": "a.py", "findings": [], "summary": "s"}]})

        parsed = cli._parse_json_report(raw)

        assert parsed["summary"] == ""

    def test_no_json_object_raises(self) -> None:
        with pytest.raises(ValueError, match="未找到 JSON"):
            cli._parse_json_report("没有对象")

    def test_invalid_severity_raises(self) -> None:
        data = {"files": [{"file": "a.py", "findings": [_finding("critical")], "summary": "s"}]}

        with pytest.raises(ValueError, match="severity"):
            cli._parse_json_report(json.dumps(data))

    def test_missing_file_field_raises(self) -> None:
        data = {"files": [{"findings": [], "summary": "s"}]}

        with pytest.raises(ValueError, match="file"):
            cli._parse_json_report(json.dumps(data))

    def test_non_bool_uncertain_raises(self) -> None:
        data = {"files": [{"file": "a.py", "findings": [_finding(uncertain="yes")], "summary": "s"}]}

        with pytest.raises(ValueError, match="uncertain"):
            cli._parse_json_report(json.dumps(data))


class TestRunReviewExitCodes:
    """退出码契约:0 完成、1 存在【严重】(仅 --json)、2 运行错误。"""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """隔离用户目录与真实 key:review 走 resolve,须保证文件配置可用。"""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        entry: dict[str, Any] = {
            "name": "prov",
            "base_url": "https://p.example.com",
            "models": ["m"],
        }
        entry["api_key"] = "placeholder-a"
        (tmp_path / ".cra").mkdir(exist_ok=True)
        (tmp_path / ".cra" / "models.json").write_text(
            json.dumps({"default": "prov/m", "providers": [entry]}, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_plain_review_prints_and_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)]))

        assert excinfo.value.code == 0
        assert "审查结论文本。" in capsys.readouterr().out

    def test_json_severe_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        _FakeReviewAgent.reply = json.dumps(
            {
                "files": [{"file": "a.py", "findings": [_finding("严重")], "summary": "s"}],
                "summary": "s",
            },
            ensure_ascii=False,
        )
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)], json=True))

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        report = json.loads(out)  # stdout 必须是纯 JSON(可被下游程序化消费)
        assert report["files"][0]["findings"][0]["severity"] == "严重"

    def test_json_non_severe_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        _FakeReviewAgent.reply = json.dumps(
            {"files": [{"file": "a.py", "findings": [_finding("建议")], "summary": "s"}], "summary": "s"},
            ensure_ascii=False,
        )
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)], json=True))

        assert excinfo.value.code == 0

    def test_invalid_json_output_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        _FakeReviewAgent.reply = "这不是 JSON"
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)], json=True))

        assert excinfo.value.code == 2

    def test_agent_failure_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        _FakeReviewAgent.raised = LLMError("网络失败")
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)]))

        assert excinfo.value.code == 2

    def test_ask_exec_rejected_before_any_work(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)], ask_exec=True))

        assert excinfo.value.code == 2
        assert "--ask-exec" in capsys.readouterr().err
        assert _FakeReviewAgent.prompts == []  # 未发起任何审查

    def test_unexpected_error_exits_2_not_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """顶层兜底:未预期异常按运行错误退出 2,禁止裸逃逸伪装成"发现严重问题"。"""
        _patch_agent(monkeypatch)
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")

        def broken_collect(_paths: list[str]) -> list[tuple[str, str]]:
            raise RuntimeError("展开阶段意外缺陷")

        monkeypatch.setattr(cli, "_collect_review_files", broken_collect)

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)]))

        assert excinfo.value.code == 2
        assert "意外错误" in capsys.readouterr().err

    def test_report_written_to_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")
        report = tmp_path / "report.md"

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)], output=str(report)))

        assert excinfo.value.code == 0
        text = report.read_text(encoding="utf-8")
        assert "# cra 审查报告" in text
        assert "审查结论文本。" in text

    def test_report_marks_failed_file_chapter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cra 复审:失败的文件在报告中保留标题并标注缺章,标题与正文一致。"""
        _patch_agent(monkeypatch)
        ok_file = tmp_path / "ok.py"
        ok_file.write_text("ok = 1", encoding="utf-8")
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("bad = 2", encoding="utf-8")
        report = tmp_path / "report.md"
        # 第一个文件成功、第二个文件失败:按输入顺序编排脚本
        replies = iter(["审查结论文本。"])

        class _SeqAgent(_FakeReviewAgent):
            def run(self, prompt: str, **kwargs: Any) -> str:
                if "bad.py" in prompt:
                    raise LLMError("网络失败")
                return next(replies)

        monkeypatch.setattr(cli, "Agent", _SeqAgent)

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(ok_file), str(bad_file)], output=str(report)))

        assert excinfo.value.code == 2
        text = report.read_text(encoding="utf-8")
        assert "ok.py" in text and "审查结论文本。" in text
        assert bad_file.name in text and "审查失败" in text

    def test_json_mode_report_is_markdown_of_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        _FakeReviewAgent.reply = json.dumps(
            {
                "files": [
                    {"file": "a.py", "findings": [_finding("一般", uncertain=True)], "summary": "s"}
                ],
                "summary": "总体",
            },
            ensure_ascii=False,
        )
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")
        report = tmp_path / "report.md"

        with pytest.raises(SystemExit):
            cli._run_review(_review_args([str(target)], json=True, output=str(report)))

        text = report.read_text(encoding="utf-8")
        assert "【一般】" in text
        assert "不确定" in text

    def test_batch_review_resets_context_per_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """批量审查逐文件重置上下文：各文件独立评审，token 不随文件数累积。"""
        _patch_agent(monkeypatch)
        paths = []
        for name in ("a.py", "b.py", "c.py"):
            target = tmp_path / name
            target.write_text("x = 1", encoding="utf-8")
            paths.append(str(target))

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args(paths))

        assert excinfo.value.code == 0
        assert _FakeReviewAgent.resets == 3

    def test_oversized_file_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """超过单文件读取上限（10 MB，与 read_file 同口径）按运行错误退出 2。"""
        _patch_agent(monkeypatch)
        target = tmp_path / "big.py"
        target.write_bytes(b"\n" * (10 * 1024 * 1024 + 1))

        with pytest.raises(SystemExit) as excinfo:
            cli._run_review(_review_args([str(target)]))

        assert excinfo.value.code == 2

    def test_prompt_carries_file_content(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_agent(monkeypatch)
        target = tmp_path / "a.py"
        target.write_text("SECRET_MARKER = 1", encoding="utf-8")

        with pytest.raises(SystemExit):
            cli._run_review(_review_args([str(target)]))

        assert any("SECRET_MARKER" in prompt for prompt in _FakeReviewAgent.prompts)
