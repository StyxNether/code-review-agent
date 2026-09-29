"""环境变量配置：config 是全项目环境变量的唯一出入口（Design.md §3.4、§6）。"""

import os

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_EXEC_TIMEOUT = 10
DEFAULT_MAX_CONTEXT_TOKENS = 100_000


def get_api_key() -> str | None:
    """读取 DeepSeek API key（必填项；仅启动时校验非空，不校验格式，无效 key 由 API 认证环节暴露）。"""
    return os.environ.get("SE_CodeAgent")


def get_model() -> str:
    """模型名，默认 deepseek-flash。"""
    return os.environ.get("CRA_MODEL") or DEFAULT_MODEL


def get_base_url() -> str:
    """OpenAI 兼容接入点，默认 https://api.deepseek.com。"""
    return os.environ.get("CRA_BASE_URL") or DEFAULT_BASE_URL


def _get_int_env(name: str, default: int) -> int:
    """读取整数型环境变量，未设置或空白时取默认值，非法值给出可读报错。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from exc


def get_exec_timeout() -> int:
    """run_python 超时秒数，默认 10（M2 实现执行工具时启用）。"""
    return _get_int_env("CRA_EXEC_TIMEOUT", DEFAULT_EXEC_TIMEOUT)


def get_max_context_tokens() -> int:
    """messages 截断阈值（tokens），默认 100000（M1 实现截断时启用）。"""
    return _get_int_env("CRA_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)
