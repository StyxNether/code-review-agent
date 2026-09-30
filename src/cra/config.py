"""配置层：~/.cra/models.json 与环境变量的合并读取（唯一出入口，Design.md §3.4/§7.4）。

解析优先级（§7.4）：--model/--provider 参数 > 配置文件 default > 环境变量回退。
api_key 只允许存在于用户目录配置文件与进程内存——仓库内任何文件、日志、测试样例
不得出现真实密钥（§6 密钥边界）；本模块是 key 的唯一读取点。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_EXEC_TIMEOUT = 10
DEFAULT_MAX_CONTEXT_TOKENS = 100_000


class ConfigError(Exception):
    """配置存在但无法使用（损坏、引用失效、参数无法解析），调用方打印后按运行错误退出 2。"""


@dataclass(frozen=True)
class ResolvedModel:
    """一次 LLM 连接的解析结果（§7.4）；provider 为 None 表示来自环境变量回退。"""

    model: str
    base_url: str
    api_key: str | None
    provider: str | None


def config_path() -> Path:
    """models.json 固定在用户目录（§7.4；Windows 为 %USERPROFILE%\\.cra\\）。"""
    return Path.home() / ".cra" / "models.json"


# ---- 环境变量回退路径（§6 配置表；models.json 不存在或未覆盖相应项时生效）----


def get_api_key() -> str | None:
    """读取 DeepSeek API key（仅校验非空，不校验格式，无效 key 由 API 认证环节暴露）。"""
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
    """run_python 超时秒数，默认 10。"""
    return _get_int_env("CRA_EXEC_TIMEOUT", DEFAULT_EXEC_TIMEOUT)


def get_max_context_tokens() -> int:
    """messages 截断阈值（tokens），默认 100000。"""
    return _get_int_env("CRA_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)


# ---- models.json（§7.4）：加载、校验与保存 ----


def load_raw() -> dict[str, Any]:
    """读取 models.json 的原始结构；文件不存在返回空配置，损坏抛 ConfigError。

    结构校验在读取时执行：损坏的配置（类型不符、default 格式非法、供应商名含
    "/"）在入口处即报错，避免下游各自防御。
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"配置文件 {path} 读取失败：{exc}") from exc
    validate_config(data)
    return data


def save_raw(data: dict[str, Any]) -> None:
    """校验后原子写入 models.json（先写临时文件再替换，防写一半损坏）。

    POSIX 上尽力收紧文件权限为 0600（该文件含明文 key，§9）；失败不阻塞写入，
    文件权限与备份注意事项由 cra config 在写入时向用户提示。
    """
    validate_config(data)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        try:
            tmp.chmod(0o600)
        except OSError:
            pass  # 权限收紧失败不阻塞写入（提示已在交互层给出）
    os.replace(tmp, path)


def validate_config(data: Any) -> None:
    """models.json 结构校验（§7.4 schema 与边界规则）；不合法抛 ConfigError。

    api_key 允许缺失或空串（回退环境变量 SE_CodeAgent）；其余字段类型不符、
    供应商名含 "/"、default 不是 "供应商/模型" 形态均视为配置损坏。
    """
    if not isinstance(data, dict):
        raise ConfigError("配置文件结构无效：顶层必须是 JSON 对象。")
    providers = data.get("providers", [])
    if not isinstance(providers, list):
        raise ConfigError("配置文件结构无效：providers 必须是数组。")
    for index, entry in enumerate(providers):
        where = f"providers[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"配置文件结构无效：{where} 必须是对象。")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"配置文件结构无效：{where} 缺少非空 name。")
        if "/" in name:
            raise ConfigError(f"配置文件结构无效：供应商名 '{name}' 不得含 '/'（§7.4）。")
        base_url = entry.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ConfigError(f"配置文件结构无效：供应商 '{name}' 缺少非空 base_url。")
        models = entry.get("models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(item, str) and item for item in models)
        ):
            raise ConfigError(
                f"配置文件结构无效：供应商 '{name}' 的 models 必须是非空字符串数组。"
            )
        api_key = entry.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ConfigError(f"配置文件结构无效：供应商 '{name}' 的 api_key 必须是字符串。")
    default = data.get("default")
    if default is not None:
        if not isinstance(default, str) or "/" not in default:
            raise ConfigError(
                f"配置文件结构无效：default {default!r} 应为 '供应商/模型'（§7.4）。"
            )
        provider_name, _, model_name = default.partition("/")
        if not provider_name or not model_name:
            raise ConfigError(
                f"配置文件结构无效：default {default!r} 的供应商与模型名均不能为空。"
            )


def _resolve_provider(
    providers: list[dict[str, Any]], provider: str, model: str | None
) -> ResolvedModel:
    """解析到指定供应商：--model 须在其清单内匹配，缺省取 models[0]（§7.4）。"""
    entry = next((item for item in providers if item.get("name") == provider), None)
    if entry is None:
        raise ConfigError(
            f"供应商 '{provider}' 未在 {config_path()} 中配置，请先运行 cra config。"
        )
    models = entry["models"]
    if model is not None and model not in models:
        raise ConfigError(
            f"模型 '{model}' 不在供应商 '{provider}' 的清单内（{', '.join(models)}）；"
            "可用 cra config 添加，或从清单中选择。"
        )
    model_name = model if model is not None else models[0]
    api_key = entry.get("api_key") or os.environ.get("SE_CodeAgent")
    if not api_key:
        raise ConfigError(
            f"供应商 '{provider}' 未配置 api_key，且环境变量 SE_CodeAgent 也未设置；"
            "请运行 cra config 补充 key，或设置环境变量。"
        )
    return ResolvedModel(
        model=model_name, base_url=entry["base_url"], api_key=api_key, provider=provider
    )


def resolve(
    *, model: str | None = None, provider: str | None = None
) -> ResolvedModel | None:
    """按 §7.4 优先级解析本次使用的模型连接。

    返回 None 表示无任何可用配置（调用方打印 cra config 引导后退出 2）；
    配置存在但损坏、default 引用失效或 --model/--provider 无法解析时抛 ConfigError。
    """
    data = load_raw()
    providers = data.get("providers") or []
    if provider is not None:
        return _resolve_provider(providers, provider, model)
    if model is not None and providers:
        owners = sorted({entry["name"] for entry in providers if model in entry["models"]})
        if len(owners) > 1:
            raise ConfigError(
                f"模型 '{model}' 在多个供应商中存在（{', '.join(owners)}），"
                "请同时用 --provider 指定供应商（§7.4）。"
            )
        if owners:
            return _resolve_provider(providers, owners[0], model)
        raise ConfigError(
            f"模型 '{model}' 不在任何已配置供应商的清单内，请检查拼写或用 cra config 添加。"
        )
    default = data.get("default")
    if default:
        provider_name, _, model_name = str(default).partition("/")
        return _resolve_provider(providers, provider_name, model_name)
    # 环境变量回退（§6 配置表）；无供应商清单时 --model 直接覆盖 CRA_MODEL
    api_key = os.environ.get("SE_CodeAgent")
    if not api_key:
        return None
    return ResolvedModel(
        model=model or os.environ.get("CRA_MODEL") or DEFAULT_MODEL,
        base_url=os.environ.get("CRA_BASE_URL") or DEFAULT_BASE_URL,
        api_key=api_key,
        provider=None,
    )
