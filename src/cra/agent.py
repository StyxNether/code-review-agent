"""Agent 循环：LangChain 1.x create_agent（底层 LangGraph）封装。

agent 只管循环与状态：会话消息正本由本模块持有（/undo、/save /load 与中断回滚
都作用于 self.messages），每轮以无状态方式把消息列表注入图、取回本轮新增消息回写
正本。LangChain/LangGraph 类型（消息、流事件、图对象）不出本模块向上暴露，cli
只见字符串回调与返回文本（Agent.run(input, callbacks) -> str 可编程接口，作业二
工作流复用的接缝）。

框架映射：
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
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import StructuredTool, tool

from cra import config, llm
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ERROR_PREFIX, EXEC_TOOL_NAME, Tool, ToolRegistry, get_default_registry

MAX_TOOL_ROUNDS = 10  # 最大工具轮次，超过则强制要求模型给出结论
_NODES_PER_TOOL_ROUND = 2  # 每个工具轮的图节点数（模型 + 工具）；middleware 引入新节点时须同步

ToolEventCallback = Callable[[str, str], None]  # (工具名, 参数 JSON)
ToolResultCallback = Callable[[str], None]  # 工具结果文本
DeltaCallback = Callable[[str], None]  # 最终回答的文本增量
ExecConfirmCallback = Callable[[dict[str, Any]], bool]  # run_python 执行确认

# ToolNode 对参数校验失败等分支的兜底文案前缀（langgraph prebuilt）；仅用于改写时
# 剥离——分支判别以 ToolMessage.status=="error" 语义字段为准（见
# _normalize_framework_error），框架升版换文案不影响协议
_FRAMEWORK_ERROR_PREFIX = "Error:"

# finish_reason="length" 时附加到最终回答：长度截断的回答若静默当最终结论，
# 用户无法察觉不完整；同段文本一并入库，保证"继续"追问时模型可见
_LENGTH_TRUNCATION_NOTICE = "\n\n⚠ 回答因达到输出长度上限被截断，可能不完整；可让我继续作答。"


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
    """字符数粗估（中英混合折中 × 1.2 安全余量）。

    中英混合按 字符数//2 折中：英文占比高时估算偏高（多丢——安全方向），中文
    占比高时估算偏低（少丢、可能连续超限）。取 0.6 系数（=//2 的 1.2 倍）向
    "宁多丢"倾斜。仅决定额外丢弃量；触发以真实 usage 为准。
    """
    total_chars = 0
    for message in messages:
        total_chars += len(message.text)
        for call in getattr(message, "tool_calls", None) or []:
            total_chars += len(call["name"]) + len(str(call.get("args") or {}))
    return int(total_chars * 0.6)


def truncate_rounds(
    messages: list[BaseMessage],
    *,
    usage_total_tokens: int | None,
    max_tokens: int,
) -> list[BaseMessage]:
    """按完整轮次原子截断：真实 usage 超阈值时从最旧丢弃完整轮次。

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
        content=content,
        name=str(call.get("name") or ""),
        tool_call_id=str(call.get("id") or ""),
    )


def _normalize_framework_error(message: BaseMessage) -> BaseMessage:
    """框架生成的错误 ToolMessage（status="error"）统一为"错误："前缀回传模型。

    以 status 语义字段判别而非文案前缀：工具的正常输出（status="success"）即使
    恰好以 "Error:" 开头也不被改写；框架升版更换文案
    时该门禁依旧成立。"错误："前缀已存在的消息（历史轮次经 wrap_tool_call 改写
    过）原样返回，避免重复叠加。
    """
    if (
        not isinstance(message, ToolMessage)
        or message.status != "error"
        or not isinstance(message.content, str)
        or message.content.startswith(ERROR_PREFIX)
    ):
        return message
    content = message.content
    if content.startswith(_FRAMEWORK_ERROR_PREFIX):
        content = content[len(_FRAMEWORK_ERROR_PREFIX) :].lstrip()
    return message.model_copy(update={"content": ERROR_PREFIX + content})


class _ToolCallbacksMiddleware(AgentMiddleware):
    """工具调用回调、执行确认与"错误："结构化错误协议。

    框架错误分支统一归入"错误："前缀：未知工具名（tool is None，复用注册表
    文案，含可用工具清单）、框架参数校验失败/畸形参数（按 ToolMessage.status
    =="error" 判别后改写前缀）、执行异常（兜底捕获）。on_tool/on_tool_result
    渲染回调在三类分支统一触发，cli 的工具打点与失败展示依赖该协议。

    confirm_exec 为 run_python 的执行前确认回调（--ask-exec / /confirm）：
    返回 False 时以结构化文本拒绝执行并回传模型，REPL 开关由 cli 经属性切换。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        confirm_exec: ExecConfirmCallback | None = None,
    ) -> None:
        self._registry = registry
        self.confirm_exec = confirm_exec
        # 由 Agent.run 按轮设置/清除（REPL 串行执行，一轮最多一个活动回调组；
        # reset() 一并清理，防异常路径挂留引用）
        self.callbacks: tuple[ToolEventCallback | None, ToolResultCallback | None] | None = None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        """模型通道错误前缀归一：框架合成的错误提示（如畸形 JSON 参数）不经
        工具层、绕过 wrap_tool_call，在请求侧统一改写。"""
        normalized = [_normalize_framework_error(message) for message in request.messages]
        if all(same is original for same, original in zip(normalized, request.messages)):
            return handler(request)  # 无框架错误消息：原请求直通，不产生额外列表
        return handler(request.override(messages=normalized))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        on_tool, on_tool_result = self.callbacks or (None, None)
        call = request.tool_call
        name = str(call.get("name") or "")
        arguments = json.dumps(call.get("args") or {}, ensure_ascii=False)
        if on_tool is not None:
            on_tool(name, arguments)
        if (
            request.tool is not None
            and name == EXEC_TOOL_NAME
            and self.confirm_exec is not None
            and not self.confirm_exec(call.get("args") or {})
        ):
            # 用户拒绝执行：结构化回传模型（协议与工具失败一致），不进入 handler
            response: Any = _error_tool_message(
                call,
                ERROR_PREFIX + "用户拒绝了本次代码执行请求；请基于静态分析继续给出结论。",
            )
            if on_tool_result is not None:
                on_tool_result(response.content)
            return response
        if request.tool is None:
            # 未知工具名：复用注册表的结构化错误文案，模型可自行纠正策略
            response: Any = _error_tool_message(call, self._registry.dispatch(name, arguments))
        else:
            try:
                response = handler(request)
            except Exception as exc:  # noqa: BLE001 — 设计要求：工具异常必须全部结构化回传模型
                response = _error_tool_message(call, ERROR_PREFIX + f"工具 '{name}' 执行失败：{exc}")
            else:
                if isinstance(response, ToolMessage):
                    response = _normalize_framework_error(response)
        result = (
            response.content
            if isinstance(response, ToolMessage) and isinstance(response.content, str)
            else ""
        )
        if on_tool_result is not None:
            on_tool_result(result)  # 三分支统一触发（空串也回调），协议对调用方一致
        return response


class _MaxToolRoundsMiddleware(AgentMiddleware):
    """最大工具轮次：达限后对下一次模型调用去工具并注入收尾指令。

    官方 ModelCallLimit/ToolCallLimit 的 exit 语义不等价（不注入、不去工具），不采用。
    计数只覆盖本轮（见 _rounds_this_run），历史轮次不累积；收尾指令经 system
    通道注入（不产生伪造 user 消息）——请求视图的轮次边界只认真实 user 消息，
    截断不会把孤立指令当作"最后一个完整轮次"而丢弃全部上下文；指令不落正本。
    """

    def __init__(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        if _rounds_this_run(request.state.get("messages", [])) < self.max_rounds:
            return handler(request)
        directive = _forced_conclusion_directive(self.max_rounds)
        # 图构建恒传 system_prompt（_build_graph），system_message 必然存在；
        # 显式失效而非伪造 user 消息兜底——伪造 user 会构成截断的轮次边界，
        # 极端场景下截断会保留孤立指令、丢弃全部真实上下文
        if request.system_message is None:
            raise RuntimeError("模型请求缺少系统提示词：达限收尾指令无法经 system 通道注入")
        request = request.override(
            tools=[],
            tool_choice=None,
            system_message=SystemMessage(
                content=f"{request.system_message.content}\n\n{directive}"
            ),
        )
        return handler(request)


class _ContextTrimMiddleware(AgentMiddleware):
    """上下文截断：模型调用前按完整轮次裁剪请求视图，正本不动。"""

    def __init__(self, max_context_tokens: int) -> None:
        self.max_context_tokens = max_context_tokens

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        trimmed = truncate_rounds(
            list(request.messages),
            usage_total_tokens=_last_response_usage(request.state.get("messages", [])),
            max_tokens=self.max_context_tokens,
        )
        return handler(request.override(messages=trimmed))


def _bridge_tool(registry: ToolRegistry, tool_def: Tool) -> StructuredTool:
    """经 @tool 声明单个注册表工具的 LangChain 桥接：执行统一委托 registry.dispatch。

    args_schema 直接采用注册表的 JSON schema（模型可见 schema 逐字段保真，@tool
    的类型标注推断在此不适用）；dict schema 下参数不经框架 pydantic 校验直达
    dispatch——参数缺失/类型不符等仍由注册表以"错误：…"结构化文本兜底。
    """
    name = tool_def.name

    @tool(name, description=tool_def.description, args_schema=tool_def.parameters)
    def _dispatch(**kwargs: Any) -> str:
        return registry.dispatch(name, json.dumps(kwargs, ensure_ascii=False))

    return _dispatch


_SESSION_MESSAGE_TYPES = ("human", "ai", "tool", "system")


def _validate_session_messages(messages: list[BaseMessage]) -> None:
    """会话恢复的结构自检：消息类型合法且 tool 消息不悬挂（/load 兜底）。

    手工编辑或旧版存档可能让正本进入 API 拒绝的非法序列（孤儿 tool 消息），
    在恢复时拒绝并给出可读错误，而非等下一次请求 400。对话段不接受 system
    消息——dump_session 不导出系统提示词，出现即属篡改注入。
    """
    pending: set[str] = set()
    for message in messages:
        if message.type == "system":
            raise ValueError("会话数据不应包含系统消息（系统提示词由当前版本提供）。")
        if message.type not in _SESSION_MESSAGE_TYPES:
            raise ValueError(f"会话数据包含不支持的消息类型：{message.type}")
        for call in getattr(message, "tool_calls", None) or []:
            call_id = str(call.get("id") or "")
            if call_id:
                pending.add(call_id)
        if isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id or "")
            if call_id not in pending:
                raise ValueError("会话数据中 tool 消息缺少配对的 tool_call（序列不合法）。")
            pending.discard(call_id)
    if pending:
        # 反向配对：assistant 的 tool_calls 缺少 tool 响应——放行会在下一次
        # 请求被 API 400 拒绝，取代本应在此处给出的可读错误
        raise ValueError("会话数据中存在没有收到结果的 tool_call（序列不合法）。")


class Agent:
    """create_agent 封装持有者：一个 REPL 会话对应一个实例，messages 跨轮持续存在。"""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        resolved: config.ResolvedModel | None = None,
        registry: ToolRegistry | None = None,
        max_context_tokens: int | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        confirm_exec: ExecConfirmCallback | None = None,
    ) -> None:
        """model 注入仅用于测试（假模型）；真实路径经 resolved（config 层解析结果）
        构建 ChatOpenAI，两者皆缺时走环境变量回退。"""
        if model is not None:
            self._chat_model = model
            self.model = getattr(model, "model_name", "") or config.get_model()
            self.provider: str | None = None  # 注入路径无供应商语义（测试用）
        elif resolved is not None:
            self._chat_model = llm.create_model(
                resolved.model, api_key=resolved.api_key, base_url=resolved.base_url
            )
            self.model = resolved.model
            self.provider = resolved.provider
        else:
            self._chat_model = llm.create_model()
            self.model = getattr(self._chat_model, "model_name", "") or config.get_model()
            self.provider = None
        self.registry = registry if registry is not None else get_default_registry()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else config.get_max_context_tokens()
        )
        self.max_tool_rounds = max_tool_rounds
        # 每轮 _NODES_PER_TOOL_ROUND 个图节点（模型/工具）× 轮次 + 收尾调用 + 余量；
        # 护栏不依赖框架默认值
        self.recursion_limit = _NODES_PER_TOOL_ROUND * max_tool_rounds + 5
        self.messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        self._tool_callbacks = _ToolCallbacksMiddleware(self.registry, confirm_exec)
        self._graph = self._build_graph()

    def _build_graph(self, model: BaseChatModel | None = None) -> Any:
        """按给定模型（缺省当前模型）与参数组装 agent 图。

        switch_model 重建时复用回调 middleware 实例（确认开关与回调引用跨切换
        保留）；轮次/截断 middleware 无跨轮状态，随图重建。
        """
        return create_agent(
            model=model if model is not None else self._chat_model,
            tools=[_bridge_tool(self.registry, tool_def) for tool_def in self.registry.definitions()],
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                self._tool_callbacks,
                _MaxToolRoundsMiddleware(self.max_tool_rounds),
                _ContextTrimMiddleware(self.max_context_tokens),
            ],
        )

    def switch_model(self, resolved: config.ResolvedModel) -> None:
        """切换模型并重建 agent 图；会话消息正本原样保留（/change model）。

        新客户端与新图全部构建成功后才赋值（整体成功或整体失败）：构建失败
        （如 key 缺失）时旧模型与图保持可用。
        """
        new_model = llm.create_model(
            resolved.model, api_key=resolved.api_key, base_url=resolved.base_url
        )
        new_graph = self._build_graph(model=new_model)
        self._chat_model = new_model
        self.model = resolved.model
        self.provider = resolved.provider
        self._graph = new_graph

    def reset(self) -> None:
        """/clear：重置为仅含系统提示词的初始状态，角色与格式契约保留。"""
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self._tool_callbacks.callbacks = None  # 防御性清理：异常路径可能挂留回调引用

    def undo_turn(self) -> bool:
        """作废最后一轮对话：删除最后一条 user 消息及其后的全部消息（/undo）。

        无对话轮次（仅系统提示词）时返回 False，正本不动。
        """
        for index in range(len(self.messages) - 1, -1, -1):
            if isinstance(self.messages[index], HumanMessage):
                del self.messages[index:]
                return True
        return False

    def context_status(self) -> dict[str, Any]:
        """当前上下文占用与截断阈值（/context 的数据源）。

        used 取最近一次响应的真实 usage（含系统提示词，服务端口径）；缺失时
        以字符估算降级并标注 estimated，供 cli 区分显示。
        """
        used = _last_response_usage(self.messages)
        estimated = used is None
        if estimated:
            used = _estimate_tokens(self.messages)
        return {"used": used, "limit": self.max_context_tokens, "estimated": estimated}

    def dump_session(self) -> list[dict[str, Any]]:
        """会话消息 → 可 JSON 化的数组（/save）；框架序列化留在 agent 层。

        正本开头的系统提示词不入档：/load 恢复时拼当前版本 SYSTEM_PROMPT，
        避免把旧提示词固化进存档后随会话漂移。
        """
        dialogue = self.messages[1:] if self.messages else []
        return messages_to_dict(dialogue)

    def load_session(self, data: list[dict[str, Any]]) -> None:
        """从 dump_session 的数据整体恢复会话（/load），系统提示词取当前版本。

        恢复后先做结构自检（_validate_session_messages），失败抛可读 ValueError
        且正本不变；赋值在校验之后，正本不会进入半恢复状态。
        """
        restored = messages_from_dict(data)
        _validate_session_messages(restored)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT), *restored]

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
        失败，含空回答）都会把正本回滚到本轮开始前的状态、本轮作废，然后
        原样上抛。
        """
        self._tool_callbacks.callbacks = (on_tool, on_tool_result)
        prefix_length = len(self.messages)  # 本轮开始前的正本长度（含系统提示词）
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
            new_messages = list(final_state["messages"][fed_length:])
            if truncated and new_messages:
                # 截断提示同文入库："继续"追问时模型可见截断状态。
                # 只改写本轮新增消息的末条——正本末条在新增为空时是本轮 user 消息，
                # 直接按 -1 定位会把它覆盖成回答
                new_messages[-1] = final_message.model_copy(update={"content": text})
            self.messages.extend(new_messages)
            return text
        except BaseException:
            # 任何异常（中断/请求失败）都回滚本轮再上抛。图状态是一次性请求
            # 视图，正本在本轮只有"追加 user 消息"与"成功后追加新消息"两处变更；
            # 按前缀长度截断，覆盖成功收尾之后仍抛异常的窄窗口（契约不依赖窗口概率）
            del self.messages[prefix_length:]
            raise
        finally:
            self._tool_callbacks.callbacks = None

    def _invoke(
        self, fed: list[BaseMessage], on_delta: DeltaCallback | None
    ) -> tuple[dict[str, Any], AIMessage]:
        """无状态图调用：注入消息列表，流式取回增量与最终状态。

        stream_mode="messages" 提供 token 增量（工具轮伴随的少量文本同样渲染，
        已知取舍）；"values" 提供每步后的状态快照，末次即最终状态——本轮
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
    """服务端 finish_reason（length 截断需向用户明示）。"""
    return message.response_metadata.get("finish_reason")
