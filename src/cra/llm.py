"""DeepSeek 客户端封装（openai SDK，OpenAI 兼容接口）。

M0 仅做最小通路：非流式补全；指数退避重试按 Design.md §5 在 M2 加入。
"""

from openai import APIError, OpenAI

from cra import config

REQUEST_TIMEOUT_SECONDS = 60  # 请求级超时（§5）：不用 SDK 默认 600s，避免失败请求长时间挂起


class LLMError(RuntimeError):
    """LLM 请求失败（网络/超时/认证等），对上层屏蔽 openai SDK 异常类型。"""


def create_client() -> OpenAI:
    """按 SE_CodeAgent / CRA_BASE_URL 构建 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=config.get_api_key(),
        base_url=config.get_base_url(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def chat(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    """非流式对话补全，返回纯文本回复（流式渲染属 M1）。"""
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    except APIError as exc:
        raise LLMError(str(exc)) from exc
    return response.choices[0].message.content or ""
