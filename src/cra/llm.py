"""LLM 客户端层：ChatOpenAI（OpenAI 兼容多供应商）构建与异常翻译（Design.md §3.1/§3.4）。

llm 只管模型客户端与重试（§3.4）：客户端构建（base_url/api_key/温度/请求级超时/
SDK 重试/认证快速失败指引）与 SDK/框架异常 → LLMError 的翻译都在本层完成，
agent/cli 不接触 SDK 类型。

重试（§5/D13）：瞬时错误经 SDK `max_retries=3` 承担（指数退避间隔与 5xx 语义以
openai SDK 实现为准），仅覆盖建连阶段；401/403 认证错误 SDK 本就不重试（快速
失败语义保留），失败时附 key 排查指引。流式迭代中途断开不重试——增量已渲染，
重发会让用户看到重复文本（与 M2/M3 语义一致，现由 SDK 边界天然保证）。
"""

import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError

from cra import config

REQUEST_TIMEOUT_SECONDS = 60  # 请求级超时（§3.1）：不用 SDK 默认 600s，避免失败请求长时间挂起
TEMPERATURE = 0.2  # §6：输出稳定，黄金用例可复现
MAX_RETRIES = 3  # D13：瞬时错误重试映射到 SDK（仅建连期，间隔以 SDK 实现为准）


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


class LLMError(RuntimeError):
    """LLM 请求失败（网络/超时/认证等），对上层屏蔽 openai SDK / LangGraph 异常类型。"""


def create_model(model_name: str | None = None) -> ChatOpenAI:
    """按 SE_CodeAgent / CRA_BASE_URL / CRA_MODEL 构建 OpenAI 兼容的 ChatOpenAI。

    models.json 属 M5：现阶段配置全部走环境变量回退（config 层唯一出入口）。
    """
    api_key = config.get_api_key()
    if not api_key:
        # 显式校验：SDK 在 api_key 为 None 时会回退读取 OPENAI_API_KEY，可能把
        # OpenAI 的密钥发往 CRA_BASE_URL（§5：key 只从 SE_CodeAgent 读取）
        raise LLMError("未设置环境变量 SE_CodeAgent（DeepSeek API key）。")
    return ChatOpenAI(
        model=model_name if model_name is not None else config.get_model(),
        api_key=api_key,
        base_url=config.get_base_url(),
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        # §3.1：usage 是截断触发与 /context 的数据源，OpenAI 兼容端点常缺省，须显式开启
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
    """非流式单次调用（保留的调用路径：后续 --no-stream 与 cra config test 使用）。"""
    try:
        return model.invoke(messages, **kwargs)
    except Exception as exc:
        translated = translate_failure(exc)
        if translated is exc:
            raise
        raise translated from exc
