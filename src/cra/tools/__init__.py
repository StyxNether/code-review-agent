"""工具注册表：名称 → (JSON schema, handler)。

注册表屏蔽 handler 细节（§3.4 数据封装）：对 agent 只暴露 schema 列表与
dispatch(name, arguments_json) -> str；工具异常一律结构化为错误文本回传模型（§5），
不向上抛出，REPL 不因工具失败崩溃。新增工具 = 调用一次 register。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cra.tools import fs

Handler = Callable[..., str]


@dataclass(frozen=True)
class Tool:
    """单个工具的定义：LLM 可见的 schema 三元组 + 本地执行入口。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


class ToolRegistry:
    """工具集合：负责 schema 暴露与参数解析、分发、异常兜底。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, parameters: dict[str, Any], handler: Handler
    ) -> None:
        """注册一个工具；同名重复注册以后者为准（便于测试替换）。"""
        self._tools[name] = Tool(
            name=name, description=description, parameters=parameters, handler=handler
        )

    def schemas(self) -> list[dict[str, Any]]:
        """返回 OpenAI function calling 的 tools 参数格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, name: str, arguments_json: str) -> str:
        """分发工具调用并返回字符串结果；任何失败都以"错误：…"结构化文本回传（§5）。

        未注册工具名、非法 JSON、参数不符 schema（如缺必填参数）都不抛出，
        而是作为错误文本交给模型自行调整策略，不触发 API 重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "（无）"
            return "错误：未注册的工具 '" + name + "'，可用工具：" + available
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            return "错误：工具 '" + name + "' 的参数不是合法 JSON：" + str(exc)
        if not isinstance(arguments, dict):
            return (
                "错误：工具 '" + name + "' 的参数必须是 JSON 对象，收到："
                + type(arguments).__name__
            )
        try:
            return str(tool.handler(**arguments))
        except Exception as exc:  # noqa: BLE001 — 设计要求：工具异常必须全部在此兜底回传模型（§5）
            return "错误：工具 '" + name + "' 执行失败：" + str(exc)


def _build_default_registry() -> ToolRegistry:
    """内置文件工具（§3.2）；M2 的 run_python 在此追加注册。"""
    registry = ToolRegistry()
    registry.register(
        name="list_dir",
        description=(
            "列出目录的两层树（本层与每个子目录的一层），忽略 .venv、__pycache__、.git、"
            ".pytest_cache、.ruff_cache 及点开头的隐藏目录。相对路径基于当前工作目录。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，默认当前目录"}
            },
            "required": [],
        },
        handler=fs.list_dir,
    )
    registry.register(
        name="read_file",
        description=(
            "带行号读取文本文件（1-based 闭区间 start/end），单次最多 400 行，"
            "超出请用 start/end 分段读取。二进制文件会返回结构化错误。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "start": {"type": "integer", "description": "起始行（1-based，含），默认 1"},
                "end": {"type": "integer", "description": "结束行（1-based，含），默认到文件尾"},
            },
            "required": ["path"],
        },
        handler=fs.read_file,
    )
    return registry


DEFAULT_REGISTRY = _build_default_registry()


def get_default_registry() -> ToolRegistry:
    """返回内置工具注册表（含 list_dir / read_file）。"""
    return DEFAULT_REGISTRY
