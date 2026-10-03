"""工具注册表：schema 暴露、分发与结构化错误。"""

import json
from pathlib import Path

from cra.tools import ToolRegistry, get_default_registry


def _boom(**_: str) -> str:
    raise RuntimeError("内部炸了")


def test_default_registry_has_builtin_tools() -> None:
    names = {item["function"]["name"] for item in get_default_registry().schemas()}
    assert names == {"list_dir", "read_file", "run_python"}


def test_run_python_declares_non_sandbox() -> None:
    """非沙箱限制必须在工具 description 中明示（模型与用户可见）。"""
    schema = next(
        item
        for item in get_default_registry().schemas()
        if item["function"]["name"] == "run_python"
    )

    assert "非沙箱" in schema["function"]["description"]


def test_schemas_are_openai_function_format() -> None:
    for item in get_default_registry().schemas():
        assert item["type"] == "function"
        function = item["function"]
        assert isinstance(function["description"], str) and function["description"]
        assert function["parameters"]["type"] == "object"


def test_dispatch_reads_real_file(tmp_path: Path) -> None:
    target = tmp_path / "x.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = get_default_registry().dispatch(
        "read_file", json.dumps({"path": str(target)})
    )

    assert "1 | value = 1" in result


def test_unknown_tool_structured_error() -> None:
    result = get_default_registry().dispatch("no_such_tool", "{}")

    assert result.startswith("错误")
    assert "未注册" in result
    assert "no_such_tool" in result


def test_run_python_missing_both_args_structured_error() -> None:
    """code/path 都缺 → 结构化错误（而非 TypeError 文本），提示二选一。"""
    result = get_default_registry().dispatch("run_python", "{}")

    assert result.startswith("错误")
    assert "二选一" in result


def test_invalid_json_args_structured_error() -> None:
    result = get_default_registry().dispatch("list_dir", "{not json")

    assert result.startswith("错误")
    assert "JSON" in result


def test_non_object_args_structured_error() -> None:
    assert get_default_registry().dispatch("list_dir", "[1, 2]").startswith("错误")


def test_schema_mismatch_structured_error() -> None:
    # read_file 缺必填参数 path → TypeError 被兜底为结构化错误，不向上抛出
    result = get_default_registry().dispatch("read_file", "{}")

    assert result.startswith("错误")


def test_handler_exception_structured_error() -> None:
    registry = ToolRegistry()
    registry.register(
        name="boom",
        description="会抛异常的假工具",
        parameters={"type": "object", "properties": {}},
        handler=_boom,
    )

    result = registry.dispatch("boom", "{}")

    assert result.startswith("错误")
    assert "内部炸了" in result


def test_empty_arguments_treated_as_empty_object() -> None:
    # 部分模型对无参工具会传空字符串
    result = get_default_registry().dispatch("list_dir", "")

    assert not result.startswith("错误")
