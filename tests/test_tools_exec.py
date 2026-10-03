"""run_python 行为与限制：限时杀树、输出截断、参数校验、编码。

超时用例注入 timeout=1 或 CRA_EXEC_TIMEOUT=1（约 1s/例），不用默认 10s 拖慢套件；
边界样例 fixtures 的黄金回归见 test_buggy_samples.py。
"""

import json
from pathlib import Path

import pytest

from cra.tools import exec_py, get_default_registry


class TestBasicExecution:
    def test_stdout_and_exit_code(self) -> None:
        result = exec_py.run_python(code="print('hello')\nprint('world')")

        assert "退出码：0" in result
        assert "hello\nworld" in result

    def test_stderr_and_nonzero_exit(self) -> None:
        code = "import sys; print('oops', file=sys.stderr); sys.exit(3)"

        result = exec_py.run_python(code=code)

        assert "退出码：3" in result
        assert "oops" in result

    def test_empty_output_marked_explicitly(self) -> None:
        result = exec_py.run_python(code="pass")

        assert "（空）" in result

    def test_utf8_output_roundtrip(self) -> None:
        # 子进程输出显式 utf-8 解码 + PYTHONIOENCODING，中文不乱码不崩溃
        result = exec_py.run_python(code="print('中文输出')")

        assert "中文输出" in result

    def test_child_cwd_is_process_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        result = exec_py.run_python(code="import os; print(os.getcwd())")

        printed = result.splitlines()[2]  # stdout 段第一行（0=退出码，1=stdout 标头）
        assert Path(printed).resolve() == tmp_path.resolve()

    def test_path_resolved_against_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "script.py").write_text("print('rel ok')\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = exec_py.run_python(path="script.py")

        assert "rel ok" in result


class TestArgValidation:
    def test_both_code_and_path_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "a.py"
        target.write_text("pass\n", encoding="utf-8")

        result = exec_py.run_python(code="pass", path=str(target))

        assert result.startswith("错误")
        assert "二选一" in result

    def test_neither_code_nor_path_rejected(self) -> None:
        assert exec_py.run_python().startswith("错误")

    def test_timeout_out_of_range_rejected(self) -> None:
        for bad in (0, -1, 61):
            result = exec_py.run_python(code="pass", timeout=bad)
            assert result.startswith("错误")
            assert "60" in result

    def test_config_timeout_validated_same_as_param(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """配置来源的超时与模型传参同标准校验，非法值不伪装成执行超时。"""
        for bad in ("0", "-5", "61"):
            monkeypatch.setenv("CRA_EXEC_TIMEOUT", bad)

            result = exec_py.run_python(code="pass")

            assert result.startswith("错误")
            assert "CRA_EXEC_TIMEOUT" in result

    def test_missing_file_structured_error(self, tmp_path: Path) -> None:
        result = exec_py.run_python(path=str(tmp_path / "nope.py"))

        assert result.startswith("错误")
        assert "不存在" in result

    def test_directory_path_structured_error(self, tmp_path: Path) -> None:
        (tmp_path / "pkg.py").mkdir()

        assert exec_py.run_python(path=str(tmp_path / "pkg.py")).startswith("错误")

    def test_non_python_file_hint(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hi\n", encoding="utf-8")

        result = exec_py.run_python(path=str(target))

        assert result.startswith("错误")
        assert "不是 Python 文件" in result
        assert "read_file" in result  # 给模型可操作的下一步

    def test_uppercase_py_suffix_accepted(self, tmp_path: Path) -> None:
        """Windows 文件系统大小写不敏感：UPPER.PY 也是 Python 文件。"""
        target = tmp_path / "SCRIPT.PY"
        target.write_text("print('ok-upper')", encoding="utf-8")

        result = exec_py.run_python(path=str(target), timeout=10)

        assert "ok-upper" in result


class TestTimeout:
    def test_timeout_param_kills_and_raises(self) -> None:
        with pytest.raises(RuntimeError, match="超时"):
            exec_py.run_python(code="while True: pass", timeout=1)

    def test_env_default_timeout_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "loop.py"
        target.write_text("while True: pass\n", encoding="utf-8")
        monkeypatch.setenv("CRA_EXEC_TIMEOUT", "1")

        with pytest.raises(RuntimeError, match="超时"):
            exec_py.run_python(path=str(target))

    def test_timeout_structured_via_dispatch(self, tmp_path: Path) -> None:
        # exec 路径的异常同样经 dispatch 兜底为结构化错误回传模型
        target = tmp_path / "loop.py"
        target.write_text("while True: pass\n", encoding="utf-8")

        result = get_default_registry().dispatch(
            "run_python", json.dumps({"path": str(target), "timeout": 1})
        )

        assert result.startswith("错误：工具 'run_python' 执行失败")
        assert "超时" in result


class TestEnvSanitization:
    """子进程环境剔除凭证变量，key 不得经输出回传模型上下文。"""

    def test_agent_key_not_in_child_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SE_CodeAgent", "sk-secret-for-test")

        result = exec_py.run_python(code="import os; print(os.environ.get('SE_CodeAgent'))")

        assert "None" in result
        assert "sk-secret-for-test" not in result

    def test_agent_key_case_variant_filtered(self) -> None:
        """Windows 环境名不区分大小写，全大写变体同样须剔除。"""
        assert exec_py._is_credential_var("SE_CodeAgent")
        assert exec_py._is_credential_var("SE_CODEAGENT")
        assert exec_py._is_credential_var("OPENAI_API_KEY")
        assert exec_py._is_credential_var("my_api_key")  # 大小写不敏感的超集过滤
        assert exec_py._is_credential_var("MY-API-KEY")  # 连字符归一为下划线后命中
        assert exec_py._is_credential_var("apikey")  # 无分隔命名同样命中
        assert not exec_py._is_credential_var("PATH")
        assert not exec_py._is_credential_var("CRA_EXEC_TIMEOUT")
        assert not exec_py._is_credential_var("MONKEY_BUSINESS")  # 仅含 KEY 不误伤

        env = {**exec_py._sanitized_env()}
        assert "SE_CODEAGENT" not in {name.upper() for name in env}

    def test_generic_api_key_vars_filtered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k-openai")
        monkeypatch.setenv("MY_API_KEY", "k-mine")

        result = exec_py.run_python(
            code="import os; print(bool(os.environ.get('OPENAI_API_KEY')), bool(os.environ.get('MY_API_KEY')))"
        )

        assert "False False" in result

    def test_non_credential_vars_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_TEST_MARKER", "kept")

        result = exec_py.run_python(code="import os; print(os.environ.get('CRA_TEST_MARKER'))")

        assert "kept" in result


class TestInterruptCleanup:
    """工具执行期间的中断必须回收子进程，不留残留进程与管道。"""

    def test_keyboard_interrupt_kills_child_before_propagating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        class _StubPopen:
            pid = 424242
            killed = False

            def communicate(self, timeout: object = None) -> tuple[str, str]:
                if not self.killed:
                    events.append("interrupted")
                    raise KeyboardInterrupt
                events.append("reaped")
                return "", ""

            def poll(self) -> None:
                return None  # 模拟进程仍在运行

        stub = _StubPopen()

        def fake_kill(process: object) -> None:
            events.append("killed")
            stub.killed = True

        monkeypatch.setattr(exec_py.subprocess, "Popen", lambda *_, **__: stub)
        monkeypatch.setattr(exec_py, "_kill_process_tree", fake_kill)

        with pytest.raises(KeyboardInterrupt):
            exec_py._run_subprocess([exec_py.sys.executable, "-c", "pass"], 10)

        # 中断路径：先杀树、再有限回收管道，然后原样上抛
        assert events == ["interrupted", "killed", "reaped"]


class TestTruncation:
    def test_stdout_truncated_at_2000(self) -> None:
        result = exec_py.run_python(code="print('A' * 5000)")

        assert "A" * 2000 in result
        assert "A" * 2001 not in result
        assert "已截断" in result

    def test_stderr_truncated_at_2000(self) -> None:
        code = "import sys; sys.stderr.write('B' * 5000)"

        result = exec_py.run_python(code=code)

        assert "B" * 2000 in result
        assert "B" * 2001 not in result
