"""LLM 客户端层：ChatOpenAI（OpenAI 兼容多供应商）构建与异常翻译。

llm 只管模型客户端与重试：客户端构建（base_url/api_key/温度/请求级超时/SDK 重试/
认证快速失败指引）与 SDK/框架异常 → LLMError 的翻译都在本层完成，agent/cli 不接触
SDK 类型。

重试语义：瞬时错误由 SDK `max_retries=3` 承担（指数退避间隔与 5xx 语义以 openai
SDK 实现为准，仅覆盖建连阶段）；401/403 认证错误 SDK 本就不重试（快速失败），
失败时附 key 排查指引；流式迭代中途断开不重试——增量已渲染，重发会让用户看到
重复文本。
"""

import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError

from cra import config

REQUEST_TIMEOUT_SECONDS = 60  # 请求级超时：不用 SDK 默认 600s，避免失败请求长时间挂起
TEMPERATURE = 0.2  # 输出稳定，黄金用例可复现
MAX_RETRIES = 3  # 瞬时错误重试映射到 SDK（仅建连期，间隔以 SDK 实现为准）
PROBE_TIMEOUT_SECONDS = 15  # cra config test 的自检超时：快速反馈，不等完整请求超时
PROBE_MAX_TOKENS = 1  # 连通性自检只发一次 max_tokens=1 的最小请求


def _key_hint() -> str:
    """key 设置命令按当前平台给出（默认供应商 DeepSeek 的统一变量名）。"""
    if sys.platform == "win32":
        return 'Windows：setx CRA_DEEPSEEK_API_KEY "你的key"，然后新开终端生效'
    return 'macOS/Linux：export CRA_DEEPSEEK_API_KEY="你的key"（写入 ~/.bashrc 或 ~/.zshrc 持久化）'


def _auth_guidance() -> str:
    """认证失败的排查指引（401/403 快速失败时随错误给出）。"""
    return (
        "认证失败（HTTP 401/403）：API key 无效、未激活或无权访问该模型，已停止重试。"
        "请用 cra config 检查该供应商的 key（或其环境变量，如 CRA_DEEPSEEK_API_KEY）；"
        f"设置方法（{_key_hint()}）。"
    )


class LLMError(RuntimeError):
    """LLM 请求失败（网络/超时/认证等），对上层屏蔽 openai SDK / LangGraph 异常类型。"""


def create_model(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """按显式连接参数或环境变量回退构建 OpenAI 兼容的 ChatOpenAI。

    api_key/base_url 缺省时走环境变量回退路径（config 层）；cra config test
    传入各供应商自己的 key 与接入点。显式校验 key 非空：SDK 在 api_key 为 None
    时会回退读取 OPENAI_API_KEY，可能把 OpenAI 的密钥发往 CRA_BASE_URL。
    """
    key = api_key if api_key is not None else config.get_api_key()
    if not key:
        raise LLMError(
            "未配置 API key：请运行 cra config，或设置环境变量 CRA_DEEPSEEK_API_KEY"
            "（使用环境变量回退方式时默认供应商为 DeepSeek）。"
        )
    return ChatOpenAI(
        model=model_name if model_name is not None else config.get_model(),
        api_key=key,
        base_url=base_url if base_url is not None else config.get_base_url(),
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        # usage 是截断触发与 /context 的数据源，OpenAI 兼容端点常缺省，须显式开启
        stream_usage=True,
    )


def translate_failure(exc: Exception) -> Exception:
    """SDK/框架异常 → LLMError 的翻译（留在 llm 层，agent/cli 不 import SDK 类型）。

    LLMError 原样返回；401/403 附 key 排查指引（快速失败语义由 SDK 保证——SDK
    对 401/403 不重试）；recursion_limit 护栏（GraphRecursionError）给可读提示；
    其余 SDK 异常统一包装为 LLMError；非 SDK/框架异常原样返回——那是意外缺陷，
    应以"意外错误"示人而非伪装成请求失败。
    """
    if isinstance(exc, LLMError):
        return exc
    translated: LLMError | None = None
    if isinstance(exc, APIStatusError) and exc.status_code in (401, 403):
        translated = LLMError(f"{_auth_guidance()}（{exc}）")
    elif isinstance(exc, (APITimeoutError, APIConnectionError, OpenAIError)):
        translated = LLMError(str(exc))
    elif isinstance(exc, GraphRecursionError):
        translated = LLMError(
            "已达到最大工具调用护栏（recursion_limit），本轮作废；"
            "若反复出现请反馈该问题。"
        )
    if translated is None:
        return exc
    translated.__cause__ = exc  # 返回而非 raise：原始异常保留在 __cause__ 供排查
    return translated


def invoke_once(model: BaseChatModel, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
    """非流式单次调用（保留的调用路径：--no-stream 与 cra config test 使用）。"""
    try:
        return model.invoke(messages, **kwargs)
    except Exception as exc:
        translated = translate_failure(exc)
        if translated is exc:
            raise
        raise translated from exc


def probe(base_url: str, api_key: str, model_name: str) -> tuple[str, str]:
    """连通性自检（cra config test）：发一次 max_tokens=1 的最小请求。

    返回 (状态, 详情)：ok / auth（401/403，key 无效）/ timeout（超时或网络不可达）/
    error（其余失败，详情为可读文本）。探测用短超时且显式 max_retries=0——自检
    的意义是快速反馈，不承担业务请求的重试语义。异常分类用 SDK 原始类型在本层
    完成（LLMError 的文本面向用户排查，不适合程序化判定），返回纯字符串状态；
    详情截断至 200 字符，防服务端响应体刷屏。
    """
    if not api_key:
        # 与 create_model 的显式校验一致：空 key 不应发到服务端换回误导性的 401
        return "error", "API key 为空；请用 cra config 补充，或设置环境变量 CRA_DEEPSEEK_API_KEY。"
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=TEMPERATURE,
        timeout=PROBE_TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        model.bind(max_tokens=PROBE_MAX_TOKENS).invoke([HumanMessage(content="ping")])
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            return "auth", str(exc)[:200]
        return "error", str(exc)[:200]
    except (APITimeoutError, APIConnectionError) as exc:
        return "timeout", str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 — 自检兜底：任何失败都归入可报告的状态而非上抛
        return "error", str(exc)[:200]
    return "ok", ""
