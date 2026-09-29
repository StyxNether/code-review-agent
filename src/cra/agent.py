"""Agent 循环：LangChain 1.x create_agent（底层 LangGraph）封装（Design.md §3.1）。

agent 只管循环与状态（§3.4）：会话消息正本由本模块持有（D14——/undo、/save /load
与中断回滚都作用于 self.messages），每轮以无状态方式把消息列表注入图、取回本轮
新增消息回写正本。LangChain/LangGraph 类型（消息、流事件、图对象）不出本模块
向上暴露，cli 只见到字符串回调与返回文本（Agent.run(input, callbacks) -> str，
§2.2 作业二接缝）。

框架映射（D8/D13/D14）：
- 模型客户端为 llm.create_model() 的 ChatOpenAI（OpenAI 兼容）；SDK/框架异常经
  llm.translate_failure 翻译为 LLMError，本模块不 import SDK 类型；
- 工具经 StructuredTool 桥接注册表：模型可见 schema 与注册表逐字段一致（保真），
  执行统一委托 registry.dispatch（"错误："结构化回传协议不变）；
- 最大工具轮次由自定义 middleware 实现（达限对下一次模型调用去工具并注入收尾
  指令；框架内建 Limit 类 middleware 的 exit 语义不等价，不采用），并显式传
  recursion_limit 作崩溃护栏；
- 上下文截断发生在请求侧（模型调用前裁剪 request.messages），正本不动；系统
  提示词经 system 通道每次注入、永不进入裁剪范围；usage 缺失时安全降级不触发。
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool, tool

from cra import config, llm
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ERROR_PREFIX, Tool, ToolRegistry, get_default_registry

MAX_TOOL_ROUNDS = 10  # §3.1：最大工具轮次，超过则强制要求模型给出结论

# ToolNode 对参数校验失败等分支的兜底文案前缀（langgraph prebuilt）；统一改写为
# "错误："前缀后回传模型（§3.1(3)：协议是行为锁定点，框架文案不得直通模型）
_FRAMEWORK_ERROR_PREFIX = "Error:"
# 工厂对"畸形/截断 JSON 参数"在模型请求侧合成的提示标记（不经工具层，绕过
# wrap_tool_call）：同样归一为"错误："前缀
_FRAMEWORK_MALFORMED_MARK = "could not be executed"

# finish_reason="length" 时附加到最终回答（M1 挂账：长度截断的回答此前被静默当
# 最终结论，用户无法察觉不完整）；同段文本一并入库，保证"继续"追问时模型可见
_LENGTH_TRUNCATION_NOTICE = "\n\n⚠ 回答因达到输出长度上限被截断，可能不完整；可让我继续作答。"

ToolEventCallback = Callable[[str, str], None]  # (工具名, 参数 JSON)
ToolResultCallback = Callable[[str], None]  # 工具结果文本
DeltaCallback = Callable[[str], None]  # 最终回答的文本增量


def _forced_conclusion_directive(max_rounds: int) -> str:
    """达到最大工具轮次时注入的收尾指令（请求侧注入，不落入正本）。"""
    return (
        f"工具调用已达上限（{max_rounds} 轮）。请立即基于已获取的信息"
        "给出最终结论，不要再尝试调用工具。"
    )


def _round_starts(messages: Sequence[BaseMessage]) -> list[int]:
    """轮次边界：每条 user 消息开启一个新轮次（请求视图内无系统消息）。"""
    return [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]


def _drop_oldest_round(messages: list[BaseMessage]) -> list[BaseMessage]:
    """丢弃最旧的一个完整轮次；只剩一轮（或零轮）时返回副本，绝不拆散最后一轮。"""
    starts = _round_starts(messages)
    if len(starts) < 2:
        return list(messages)
    return messages[starts[1] :]


def _estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """字符数 // 2 的粗估（中英混合折中）。仅决定额外丢弃量；触发以真实 usage 为准。"""
    total_chars = 0
    for message in messages:
        total_chars += len(message.text)
        for call in getattr(message, "tool_calls", None) or []:
            total_chars += len(call["name"]) + len(str(call.get("args") or {}))
    return total_chars // 2


def truncate_rounds(
    messages: list[BaseMessage],
    *,
    usage_total_tokens: int | None,
    max_tokens: int,
) -> list[BaseMessage]:
    """按完整轮次原子截断（§3.1）：真实 usage 超阈值时从最旧丢弃完整轮次。

    以 user 消息为轮次边界整体丢弃，保证 tool_calls 与配对的 tool 消息不被拆散
    （拆散会导致 API 400）；至少保留最近一个完整轮次——单轮自身超限时宁可超限
    也不产生非法消息结构。返回新列表（纯函数）。本函数只作用于请求视图：
    正本（Agent.messages）不动，系统提示词经 system 通道注入、不在视图内。
    """
    if usage_total_tokens is None or usage_total_tokens <= max_tokens:
        return list(messages)
    kept = _drop_oldest_round(messages)
    while _estimate_tokens(kept) > max_tokens and len(_round_starts(kept)) >= 2:
        kept = _drop_oldest_round(kept)
    return kept


def _rounds_this_run(messages: Sequence[BaseMessage]) -> int:
    """本轮（最后一条 user 消息之后）的工具轮数：带 tool_calls 的 AI 消息记一轮。

    只统计最后一条 user 消息之后——每轮 run 以无状态方式注入完整历史，历史中的
    旧工具轮（此前各轮产生）不得累积到本次限额。
    """
    last_human = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    return sum(
        1
        for message in messages[last_human + 1 :]
        if isinstance(message, AIMessage) and message.tool_calls
    )


def _last_response_usage(messages: Sequence[BaseMessage]) -> int | None:
    """最近一次模型响应的真实 usage；缺失（兼容端点未回）时安全降级为不触发截断。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.usage_metadata:
            return message.usage_metadata.get("total_tokens")
    return None


def _error_tool_message(call: dict[str, Any], content: str) -> ToolMessage:
    """按 tool_call 构造错误结果消息（id/name 与调用一一对应）。"""
    return ToolMessage(
        content=content, name=str(call.get("name", "")), tool_call_id=str(call.get("id", ""))
    )


class _ToolCallbacksMiddleware(AgentMiddleware):
    """工具调用回调与"错误："结构化错误协议（§3.1(3)/§5）。

    框架三类错误分支统一归入"错误："前缀：未知工具名（tool is None，复用注册表
    文案，含可用工具清单）、框架参数校验失败（ToolNode 以"Error: …"兜底，改写
    前缀）、执行异常（兜底捕获）。on_tool/on_tool_result 渲染回调在此统一触发，
    cli 的工具打点与失败展示依赖该协议。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        # 由 Agent.run 按轮设置/清除（REPL 串行执行，一轮最多一个活动回调组）
        self.callbacks: tuple[ToolEventCallback | None, ToolResultCallback | None] | None = None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        """模型通道错误前缀归一（框架对畸形参数的合成提示只在请求侧出现）。"""
        if not any(
            isinstance(message, ToolMessage)
            and isinstance(message.content, str)
            and _FRAMEWORK_MALFORMED_MARK in message.content
            for message in request.messages
        ):
            return handler(request)
        normalized = [
            message.model_copy(update={"content": ERROR_PREFIX + message.content})
            if isinstance(message, ToolMessage)
            and isinstance(message.content, str)
            and _FRAMEWORK_MALFORMED_MARK in message.content
            else message
            for message in request.messages
        ]
        return handler(request.override(messages=normalized))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        on_tool, on_tool_result = self.callbacks or (None, None)
        call = request.tool_call
        name = str(call.get("name", ""))
        arguments = json.dumps(call.get("args") or {}, ensure_ascii=False)
        if on_tool is not None:
            on_tool(name, arguments)
        if request.tool is None:
            # 未知工具名：复用注册表的结构化错误文案，模型可自行纠正策略
            result = self._registry.dispatch(name, arguments)
            if on_tool_result is not None:
                on_tool_result(result)
            return _error_tool_message(call, result)
        try:
            response = handler(request)
        except Exception as exc:  # noqa: BLE001 — 设计要求：工具异常必须全部结构化回传模型（§5）
            result = ERROR_PREFIX + f"工具 '{name}' 执行失败：{exc}"
            if on_tool_result is not None:
                on_tool_result(result)
            return _error_tool_message(call, result)
        result = response.content if isinstance(response, ToolMessage) else ""
        if isinstance(result, str) and result.startswith(_FRAMEWORK_ERROR_PREFIX):
            result = ERROR_PREFIX + result[len(_FRAMEWORK_ERROR_PREFIX) :].lstrip()
            if isinstance(response, ToolMessage):
                response = response.model_copy(update={"content": result})
        if on_tool_result is not None and isinstance(result, str) and result:
            on_tool_result(result)
        return response


class _MaxToolRoundsMiddleware(AgentMiddleware):
    """最大工具轮次（§3.1(4)）：达限后对下一次模型调用去工具并注入收尾指令。

    官方 ModelCallLimit/ToolCallLimit 的 exit 语义不等价（不注入、不去工具），不采用。
    计数只覆盖本轮（见 _rounds_this_run），历史轮次不累积；注入的指令只进本次
    请求，不落入正本（D14：会话消息正本只含真实对话）。
    """

    def __init__(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        if _rounds_this_run(request.state["messages"]) < self.max_rounds:
            return handler(request)
        request = request.override(
            tools=[],
            tool_choice=None,
            messages=[
                *request.messages,
                HumanMessage(content=_forced_conclusion_directive(self.max_rounds)),
            ],
        )
        return handler(request)


class _ContextTrimMiddleware(AgentMiddleware):
    """上下文截断（§3.1(6)）：模型调用前按完整轮次裁剪请求视图，正本不动。"""

    def __init__(self, max_context_tokens: int) -> None:
        self.max_context_tokens = max_context_tokens

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        trimmed = truncate_rounds(
            list(request.messages),
            usage_total_tokens=_last_response_usage(request.state["messages"]),
            max_tokens=self.max_context_tokens,
        )
        return handler(request.override(messages=trimmed))


def _bridge_tool(registry: ToolRegistry, tool_def: Tool) -> StructuredTool:
    """经 @tool 声明单个注册表工具的 LangChain 桥接：执行统一委托 registry.dispatch。

    args_schema 直接采用注册表的 JSON schema（模型可见 schema 逐字段保真，@tool
    的类型标注推断在此不适用）；dict schema 下参数不经框架 pydantic 校验直达
    dispatch——参数缺失/类型不符等仍由注册表以"错误：…"结构化文本兜底，协议
    路径与 M3 一致。
    """
    name = tool_def.name

    @tool(name, description=tool_def.description, args_schema=tool_def.parameters)
    def _dispatch(**kwargs: Any) -> str:
        return registry.dispatch(name, json.dumps(kwargs, ensure_ascii=False))

    return _dispatch


class Agent:
    """create_agent 封装持有者：一个 REPL 会话对应一个实例，messages 跨轮持续存在。"""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        registry: ToolRegistry | None = None,
        max_context_tokens: int | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._chat_model = model if model is not None else llm.create_model()
        # 展示用模型名（cli 欢迎语）；框架模型对象不向上暴露
        self.model = getattr(self._chat_model, "model_name", "") or config.get_model()
        self.registry = registry if registry is not None else get_default_registry()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else config.get_max_context_tokens()
        )
        self.max_tool_rounds = max_tool_rounds
        # 每轮至多 2 个图节点（模型/工具）× 轮次 + 收尾调用 + 余量；护栏不依赖框架默认值
        self.recursion_limit = 2 * max_tool_rounds + 5
        self.messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        self._tool_callbacks = _ToolCallbacksMiddleware(self.registry)
        self._graph = create_agent(
            model=self._chat_model,
            tools=[_bridge_tool(self.registry, tool_def) for tool_def in self.registry.definitions()],
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                self._tool_callbacks,
                _MaxToolRoundsMiddleware(max_tool_rounds),
                _ContextTrimMiddleware(self.max_context_tokens),
            ],
        )

    def reset(self) -> None:
        """/clear：重置为仅含系统提示词的初始状态，角色与格式契约保留（§6）。"""
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    def run(
        self,
        user_input: str,
        *,
        on_tool: ToolEventCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        """处理一轮用户输入：自主调用工具直至给出最终回答，返回最终文本。

        on_tool / on_tool_result / on_delta 为渲染回调（cli 注入，agent 不依赖
        rich）；框架流事件在本层适配为纯字符串增量。任何异常（Ctrl+C 或请求
        失败，含空回答）都会把正本回滚到本轮开始前的状态、本轮作废（§5），然后
        原样上抛。
        """
        self._tool_callbacks.callbacks = (on_tool, on_tool_result)
        self.messages.append(HumanMessage(content=user_input))
        fed = self.messages[1:]  # 系统提示词经 system 通道注入，不入图状态
        fed_length = len(fed)
        try:
            final_state, final_message = self._invoke(fed, on_delta)
            text = final_message.text
            if not text.strip():
                raise llm.LLMError("模型返回了空回答，请重试或换个问法。")
            truncated = _finish_reason(final_message) == "length"
            if truncated:
                text += _LENGTH_TRUNCATION_NOTICE
                if on_delta is not None:
                    on_delta(_LENGTH_TRUNCATION_NOTICE)
            self.messages.extend(final_state["messages"][fed_length:])
            if text != final_message.text:
                # 截断提示同文入库（M3 行为等价）："继续"追问时模型可见截断状态
                self.messages[-1] = final_message.model_copy(update={"content": text})
            return text
        except BaseException:
            # §5：任何异常（中断/请求失败）都回滚本轮再上抛。图状态是一次性请求
            # 视图，运行期间正本只有"本轮起始 user 消息"一处变更，删除即可。
            del self.messages[-1]
            raise
        finally:
            self._tool_callbacks.callbacks = None

    def _invoke(
        self, fed: list[BaseMessage], on_delta: DeltaCallback | None
    ) -> tuple[dict[str, Any], AIMessage]:
        """无状态图调用（D14）：注入消息列表，流式取回增量与最终状态。

        stream_mode="messages" 提供 token 增量（工具轮伴随的少量文本同样渲染，
        §3.1 已知取舍）；"values" 提供每步后的状态快照，末次即最终状态——本轮
        新增消息 = 最终状态超出 fed 的尾部，据此回写正本。
        """
        final_state: dict[str, Any] | None = None
        try:
            for mode, payload in self._graph.stream(
                {"messages": list(fed)},
                {"recursion_limit": self.recursion_limit},
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    chunk = payload[0]
                    if isinstance(chunk, AIMessageChunk):
                        text = chunk.text
                        if text and on_delta is not None:
                            on_delta(text)
                else:
                    final_state = payload
        except Exception as exc:
            translated = llm.translate_failure(exc)
            if translated is exc:
                raise  # 意外缺陷原样上抛（cli 以"意外错误"展示），不伪装成请求失败
            raise translated from exc
        if final_state is None:
            raise llm.LLMError("模型调用未返回最终状态。")
        messages = final_state["messages"]
        final_message = messages[-1] if messages else None
        if not isinstance(final_message, AIMessage) or final_message.tool_calls:
            raise llm.LLMError("模型调用的收尾响应缺少最终回答。")
        return final_state, final_message


def _finish_reason(message: AIMessage) -> str | None:
    """服务端 finish_reason（§3.1：length 截断需向用户明示）。"""
    return message.response_metadata.get("finish_reason")
