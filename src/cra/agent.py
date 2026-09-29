"""ReAct Agent 循环（Design.md §3.1）：messages 维护、function calling 分发、
最大工具轮次、上下文按完整轮次原子截断。

agent 只管循环与状态（§3.4）：LLM 交互经 llm.py，工具执行经 tools 注册表；
messages 的结构与截断规则只存在于本模块。

关于 §3.1 "工具轮次非流式"：轮次类型（工具轮/最终回答）只能在响应返回后判定，
无法预判，故统一采用流式请求——tool_calls 由分片组装，完整性与非流式等价，
usage 经 stream_options 取得；工具轮不渲染增量、仅最终回答渲染，用户可见行为
与设计一致。与 §3.1 字面的差异已在 PROGRESS.md 验收记录中声明，待验收裁定。
"""

from collections.abc import Callable
from typing import Any

from cra import config, llm
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ToolRegistry, get_default_registry

MAX_TOOL_ROUNDS = 10  # §3.1：最大工具轮次，超过则强制要求模型给出结论

ToolEventCallback = Callable[[str, str], None]  # (工具名, 参数 JSON 原文)
ToolResultCallback = Callable[[str], None]  # 工具结果文本
DeltaCallback = Callable[[str], None]  # 最终回答的文本增量


def _user_message_indices(messages: list[dict[str, Any]]) -> list[int]:
    """轮次边界：每条 user 消息开启一个新轮次（messages[0] 是系统提示词）。"""
    return [i for i, message in enumerate(messages) if i > 0 and message["role"] == "user"]


def _drop_oldest_round(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """丢弃最旧的一个完整轮次；只剩一轮（或零轮）时原样返回，绝不拆散最后一轮。"""
    boundaries = _user_message_indices(messages)
    if len(boundaries) < 2:
        return messages
    return [messages[0], *messages[boundaries[1] :]]


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """字符数 // 2 的粗估（中英混合折中）。仅决定额外丢弃量；触发以真实 usage 为准。"""
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            total_chars += len(call.get("function", {}).get("name") or "")
            total_chars += len(call.get("function", {}).get("arguments") or "")
    return total_chars // 2


def truncate_rounds(
    messages: list[dict[str, Any]],
    *,
    usage_total_tokens: int | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """按完整轮次原子截断（§3.1）：真实 usage 超阈值时从最旧丢弃完整轮次。

    以 user 消息为轮次边界整体丢弃，保证 tool_calls 与配对的 tool 消息不被拆散
    （拆散会导致 API 400）；系统提示词（messages[0]）永远保留；至少保留最近一个
    完整轮次——单轮自身超限时宁可超限也不产生非法消息结构。返回新列表（纯函数）。
    """
    if usage_total_tokens is None or usage_total_tokens <= max_tokens:
        return messages
    kept = _drop_oldest_round(messages)
    while _estimate_tokens(kept) > max_tokens and len(_user_message_indices(kept)) >= 2:
        kept = _drop_oldest_round(kept)
    return kept


def _assistant_message_dict(message: llm.AssistantMessage) -> dict[str, Any]:
    """归一化助手消息 → API 消息字典（content 可为 None，与 tool_calls 并存合法）。"""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ],
    }


class Agent:
    """ReAct 循环持有者：一个 REPL 会话对应一个实例，messages 跨轮持续存在。"""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        registry: ToolRegistry | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.client = client if client is not None else llm.create_client()
        self.model = model if model is not None else config.get_model()
        self.registry = registry if registry is not None else get_default_registry()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else config.get_max_context_tokens()
        )
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        """/clear：重置为仅含系统提示词的初始状态，角色与格式契约保留（§6）。"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(
        self,
        user_input: str,
        *,
        on_tool: ToolEventCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        """处理一轮用户输入：自主调用工具直至给出最终回答，返回最终文本。

        on_tool / on_tool_result / on_delta 为渲染回调（cli 注入，agent 不依赖 rich）。
        模型在工具轮若伴随少量文本增量，也会经 on_delta 回调（罕见且无害）。
        """
        snapshot = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop(on_tool, on_tool_result, on_delta)
        except KeyboardInterrupt:
            # §5：中断把 messages 回滚到本轮开始前的合法前缀，该轮作废（防半截消息 400）
            del self.messages[snapshot:]
            raise

    def _loop(
        self,
        on_tool: ToolEventCallback | None,
        on_tool_result: ToolResultCallback | None,
        on_delta: DeltaCallback | None,
    ) -> str:
        tool_rounds = 0
        while True:
            response = llm.complete(
                self.client,
                self.model,
                self.messages,
                tools=self.registry.schemas(),
                stream=True,
                on_delta=on_delta,
            )
            self._absorb_usage(response.total_tokens)
            message = response.message
            if not message.tool_calls:
                self.messages.append({"role": "assistant", "content": message.content or ""})
                return message.content or ""
            if tool_rounds >= MAX_TOOL_ROUNDS:
                return self._force_conclusion(on_delta)
            self.messages.append(_assistant_message_dict(message))
            for call in message.tool_calls:
                if on_tool is not None:
                    on_tool(call.name, call.arguments)
                result = self.registry.dispatch(call.name, call.arguments)
                if on_tool_result is not None:
                    on_tool_result(result)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            tool_rounds += 1

    def _force_conclusion(self, on_delta: DeltaCallback | None) -> str:
        """达到最大工具轮次：丢弃刚收到的 tool_calls 响应（不落库），注入指令后
        以不带 tools 的调用强制收尾，保证循环必然终止。"""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"工具调用已达上限（{MAX_TOOL_ROUNDS} 轮）。请立即基于已获取的信息"
                    "给出最终结论，不要再尝试调用工具。"
                ),
            }
        )
        response = llm.complete(
            self.client, self.model, self.messages, tools=None, stream=True, on_delta=on_delta
        )
        self._absorb_usage(response.total_tokens)
        self.messages.append({"role": "assistant", "content": response.message.content or ""})
        return response.message.content or ""

    def _absorb_usage(self, total_tokens: int | None) -> None:
        """记录真实 usage 并按需触发轮次截断（每轮响应后检查，含工具轮）。"""
        if total_tokens is None:
            return
        self.messages = truncate_rounds(
            self.messages,
            usage_total_tokens=total_tokens,
            max_tokens=self.max_context_tokens,
        )
