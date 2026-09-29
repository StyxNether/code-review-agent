"""DeepSeek 客户端封装（openai SDK，OpenAI 兼容接口）。

llm 只管 API 与重试（§3.4）：把 SDK 的流式/非流式响应归一为本模块的数据类型，
tool_calls 分片组装与 usage 读取都在此层完成，agent 不接触 SDK 类型。
重试（§5）：超时/限流/连接错误按 1/2/4s 指数退避，最多重试 3 次；认证错误
（401/403）快速失败并给出 key 排查指引。重试只覆盖建连阶段（请求发出前的失败）；
流式响应中途断开不重试——增量已回调渲染，重发会让用户看到重复文本。
"""

import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from cra import config

REQUEST_TIMEOUT_SECONDS = 60  # 请求级超时（§5）：不用 SDK 默认 600s，避免失败请求长时间挂起
TEMPERATURE = 0.2  # §6：输出稳定，黄金用例可复现
RETRY_DELAYS_SECONDS = (1, 2, 4)  # §5：指数退避间隔；即最多重试 3 次（首次 + 3 次尝试）

_RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)


def _key_hint() -> str:
    """key 设置命令按当前平台给出（M3 挂账：指引跨平台化）。"""
    if sys.platform == "win32":
        return 'Windows：setx SE_CodeAgent "sk-你的key"，然后新开终端生效'
    return 'macOS/Linux：export SE_CodeAgent="sk-你的key"（写入 ~/.bashrc 或 ~/.zshrc 持久化）'


def _auth_guidance() -> str:
    """认证失败的排查指引（§5：401/403 快速失败时随错误给出）。"""
    return (
        "认证失败（HTTP 401/403）：API key 无效、未激活或无权访问该模型，已停止重试。"
        "请检查环境变量 SE_CodeAgent 是否为有效的 DeepSeek API key；"
        f"修改方法（{_key_hint()}）。"
    )
_RETRY_EXHAUSTED_GUIDANCE = (
    "请求多次失败（已按 1s/2s/4s 重试 3 次）。请检查网络连接是否可用、"
    "CRA_BASE_URL 指向的服务是否可达，稍后重试。"
)

DeltaCallback = Callable[[str], None]


class LLMError(RuntimeError):
    """LLM 请求失败（网络/超时/认证等），对上层屏蔽 openai SDK 异常类型。"""


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用：arguments 是模型输出的 JSON 字符串，由注册表负责解析。"""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantMessage:
    """归一化的助手消息：文本与（可能为空的）工具调用列表。"""

    content: str | None
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class LLMResponse:
    """补全结果 + 上下文截断所需的 usage（服务端未返回时为 None）。

    finish_reason 透传服务端取值（M1 挂账：长度截断的回答此前被静默当最终
    结论）；"length" 时由 agent 在最终回答后明示截断提示。服务端未给出时为 None。
    """

    message: AssistantMessage
    total_tokens: int | None
    finish_reason: str | None = None


def create_client() -> OpenAI:
    """按 SE_CodeAgent / CRA_BASE_URL 构建 OpenAI 兼容客户端。"""
    api_key = config.get_api_key()
    if not api_key:
        # 显式校验：SDK 在 api_key 为 None 时会回退读取 OPENAI_API_KEY，可能把
        # OpenAI 的密钥发往 CRA_BASE_URL（§5：key 只从 SE_CodeAgent 读取）
        raise LLMError("未设置环境变量 SE_CodeAgent（DeepSeek API key）。")
    return OpenAI(
        api_key=api_key,
        base_url=config.get_base_url(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        # 显式关闭 SDK 隐式重试（默认 2 次）：重试策略由 M2 在应用层统一实现，
        # 避免两层重试叠加、行为不可观测
        max_retries=0,
    )


def _call_with_retries(send: Callable[[], Any]) -> Any:
    """执行一次 API 请求，仅对超时/限流/连接错误按指数退避重试（§5）。

    认证错误（401/403）快速失败并给出 key 排查指引，绝不重试（重试只会
    重复失败还拖慢报错）；其余 OpenAIError（如 4xx 参数错误、5xx）不属于
    设计约定的重试范围，原样上抛由 complete 包装为 LLMError。
    """
    last_error: OpenAIError | None = None
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            return send()
        except OpenAIError as exc:
            if isinstance(exc, APIStatusError) and exc.status_code in (401, 403):
                raise LLMError(f"{_auth_guidance()}（{exc}）") from exc
            if not isinstance(exc, _RETRYABLE_ERRORS):
                raise
            last_error = exc
            if attempt < len(RETRY_DELAYS_SECONDS):
                time.sleep(RETRY_DELAYS_SECONDS[attempt])
    raise LLMError(f"{_RETRY_EXHAUSTED_GUIDANCE}（最后错误：{last_error}）") from last_error


def complete(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
    on_delta: DeltaCallback | None = None,
) -> LLMResponse:
    """对话补全入口：stream=False 一次取回；stream=True 增量组装并回调文本增量。"""
    try:
        if stream:
            return _complete_streaming(client, model, messages, tools, on_delta)
        return _complete_once(client, model, messages, tools)
    except OpenAIError as exc:
        # OpenAIError 是 APIError 的父类：Length/ContentFilterFinishReason 等非 API
        # 异常同样包装，保证上层只需处理 LLMError（§3.4：屏蔽 SDK 异常类型）
        raise LLMError(str(exc)) from exc


def _request_kwargs(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, stream: bool
) -> dict[str, Any]:
    """构造请求参数；tools 为空时省略该字段，不依赖 None 的序列化语义。"""
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": TEMPERATURE}
    if tools:
        kwargs["tools"] = tools
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


def _complete_once(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> LLMResponse:
    """非流式补全：tool_calls 与 usage 直接可取。"""
    response = _call_with_retries(
        lambda: client.chat.completions.create(
            **_request_kwargs(model, messages, tools, stream=False)
        )
    )
    if not response.choices:
        raise LLMError("服务商返回了空 choices，无法取得回答。")
    message = response.choices[0].message
    tool_calls = [
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "")
        for call in (message.tool_calls or [])
    ]
    usage = response.usage
    return LLMResponse(
        message=AssistantMessage(content=message.content, tool_calls=tool_calls),
        total_tokens=usage.total_tokens if usage else None,
        finish_reason=response.choices[0].finish_reason,
    )


def _complete_streaming(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    on_delta: DeltaCallback | None,
) -> LLMResponse:
    """流式补全：文本增量经 on_delta 回调；tool_calls 分片按 index 组装为完整调用。

    usage 通过 stream_options 在末块取得（§3.1），服务端不支持时为 None（安全降级：
    只是不触发截断，不影响正确性）。仅建连（create 调用）参与重试；迭代中途的
    连接中断不重试（增量已渲染，重发会重复输出），经 complete 包装为 LLMError 上抛。
    """
    stream = _call_with_retries(
        lambda: client.chat.completions.create(
            **_request_kwargs(model, messages, tools, stream=True)
        )
    )
    content_parts: list[str] = []
    fragments: dict[int, dict[str, Any]] = {}
    total_tokens: int | None = None
    finish_reason: str | None = None
    try:
        for chunk in stream:
            if chunk.usage is not None:
                total_tokens = chunk.usage.total_tokens
            if not chunk.choices:
                continue  # 末块只携带 usage
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason  # 末个非空取值生效
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                if on_delta is not None:
                    on_delta(delta.content)
            for piece in delta.tool_calls or []:
                slot = fragments.setdefault(piece.index, {"id": "", "name": "", "arguments": []})
                if piece.id:
                    slot["id"] = piece.id
                if piece.function:
                    if piece.function.name:
                        slot["name"] = piece.function.name
                    if piece.function.arguments:
                        slot["arguments"].append(piece.function.arguments)
    finally:
        with suppress(Exception):  # 清理路径：关流失败不掩盖正在传播的原始异常
            stream.close()
    if fragments and all(isinstance(key, int) for key in fragments):
        ordered_slots = [fragments[key] for key in sorted(fragments)]
    else:
        ordered_slots = list(fragments.values())  # index 缺失（兼容服务端）按到达顺序兜底
    tool_calls = [
        ToolCall(id=slot["id"], name=slot["name"], arguments="".join(slot["arguments"]))
        for slot in ordered_slots
    ]
    return LLMResponse(
        message=AssistantMessage(content="".join(content_parts) or None, tool_calls=tool_calls),
        total_tokens=total_tokens,
        finish_reason=finish_reason,
    )
