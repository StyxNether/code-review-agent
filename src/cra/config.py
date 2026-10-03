"""配置层：~/.cra/models.json 与环境变量的合并读取（唯一出入口）。

解析优先级：--model/--provider 参数 > 配置文件 default > 环境变量回退。
api_key 只允许存在于用户目录配置文件与进程内存——仓库内任何文件、日志、测试样例
不得出现真实密钥；本模块是 key 的唯一读取点。
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_EXEC_TIMEOUT = 10
DEFAULT_MAX_CONTEXT_TOKENS = 100_000


class ConfigError(Exception):
    """配置存在但无法使用（损坏、引用失效、参数无法解析），调用方打印后按运行错误退出 2。"""


# ---- 供应商预设（离线内置，不联网拉取清单）----


@dataclass(frozen=True)
class ProviderPreset:
    """供应商预设：官方接入点 + 预填模型清单 + key 环境变量名（CRA_ 统一格式）。

    预填项均可在交互中修改；模型名有时效性，cra config test 可兜底验证。
    """

    name: str  # 写入 models.json 的供应商名（default 引用与 --provider 均用它）
    base_url: str
    models: tuple[str, ...]
    env_var: str


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        "DeepSeek", "https://api.deepseek.com", ("deepseek-flash", "deepseek-v4-pro"), "CRA_DEEPSEEK_API_KEY"
    ),
    ProviderPreset(
        "Kimi (Moonshot)", "https://api.moonshot.cn/v1", ("kimi-k3", "kimi-k2.6"), "CRA_MOONSHOT_API_KEY"
    ),
    ProviderPreset(
        "OpenAI", "https://api.openai.com/v1", ("gpt-6-astra", "gpt-6-luna"), "CRA_OPENAI_API_KEY"
    ),
    ProviderPreset(
        "阿里云百炼",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ("qwen3.8-max", "qwen3.8-flash"),
        "CRA_DASHSCOPE_API_KEY",
    ),
    ProviderPreset(
        "Z.ai Coding Plan",
        "https://api.z.ai/api/coding/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRA_ZAI_API_KEY",
    ),
    ProviderPreset(
        "Z.ai API", "https://api.z.ai/api/paas/v4", ("glm-5.3", "glm-5.3-flash"), "CRA_ZAI_API_KEY"
    ),
    ProviderPreset(
        "BigModel Coding Plan",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRA_ZHIPUAI_API_KEY",
    ),
    ProviderPreset(
        "BigModel API",
        "https://open.bigmodel.cn/api/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRA_ZHIPUAI_API_KEY",
    ),
)


def env_var_for_provider(name: str) -> str:
    """自定义供应商 → key 环境变量名：CRA_供应商名大写_API_KEY（全供应商统一格式）。

    规则：`CRA_` 前缀 + 供应商名大写、空格与非字母数字转 `_` + `_API_KEY` 后缀；
    非 ASCII 字符剥离（前缀已保证变量名不以数字开头）。退化输入的处理：剥离后
    为空（纯非 ASCII/纯符号名）时以名称摘要生成唯一变量名 `CRA_<摘要>_API_KEY`
    ——不同供应商不得共用同一兜底名，否则 key 串用会把一家凭据发给另一家的
    接入点。
    """
    parts: list[str] = []
    for char in name.upper():
        if char.isascii() and char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "_":
            parts.append("_")
    stem = "".join(parts).strip("_")
    if not stem:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8].upper()
        return f"CRA_{digest}_API_KEY"
    return f"CRA_{stem}_API_KEY"


def provider_env_var(name: str) -> str:
    """供应商 key 的回退环境变量名：预设与自定义统一为 CRA_ 前缀格式。"""
    for preset in PROVIDER_PRESETS:
        if preset.name == name:
            return preset.env_var
    return env_var_for_provider(name)


def resolve_provider_api_key(name: str, entry_api_key: str | None) -> str | None:
    """供应商 key 回退链：条目 api_key → CRA_供应商名大写_API_KEY → SE_CodeAgent（历史变量）。

    key 是否有效（与供应商匹配）由 API 认证环节暴露，此处只做存在性回退。
    """
    if entry_api_key:
        return entry_api_key
    return os.environ.get(provider_env_var(name)) or os.environ.get("SE_CodeAgent")


# ---- 环境变量登记（设置时登记、卸载时只清登记在案的变量）----


def env_registry_path() -> Path:
    """本程序写入的环境变量登记文件（~/.cra/env-vars.json）。

    卸载时据此区分"程序自己设置的变量"与"用户已有的同名变量"——只清前者，
    绝不触碰后者（使用者在此之前可能已自行设置 CRA_DEEPSEEK_API_KEY 等变量）。
    """
    return config_path().parent / "env-vars.json"


def registered_env_vars() -> list[str]:
    """读取登记清单；文件缺失或损坏返回空表（卸载侧据此保守跳过清理）。"""
    path = env_registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    names = data.get("env_vars") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []
    return [name for name in names if isinstance(name, str) and name]


def register_env_var(name: str) -> None:
    """登记一个由本程序写入的环境变量名（去重、原子替换）。

    Windows 为 setx 持久写入；POSIX 为当前进程即时设置（持久化取决于用户
    是否执行 export，登记项在卸载时用于打印待删行指引）。写盘失败上抛
    OSError，由调用方提示——登记缺失会导致卸载时无法自动清理该变量，须让
    用户知晓后手工处理。读-改-写未加跨进程锁：单用户 CLI 的并发登记（同时
    开两个 cra config）理论上可能丢一次登记，属可接受边界。
    """
    names = registered_env_vars()
    if name in names:
        return
    names.append(name)
    path = env_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"env_vars": names}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


@dataclass(frozen=True)
class ResolvedModel:
    """一次 LLM 连接的解析结果；provider 为 None 表示来自环境变量回退。

    api_key 恒为非空字符串：resolve 的所有返回路径都保证 key 存在（缺失时抛
    ConfigError 或返回 None），调用方无须再做空值回退。
    """

    model: str
    base_url: str
    api_key: str
    provider: str | None


def config_path() -> Path:
    """models.json 固定在用户目录（Windows 为 %USERPROFILE%\\.cra\\）。"""
    return Path.home() / ".cra" / "models.json"


# ---- 环境变量回退路径（models.json 不存在或未覆盖相应项时生效）----


def get_api_key() -> str | None:
    """默认供应商（DeepSeek）的 key 环境变量回退：CRA_DEEPSEEK_API_KEY → SE_CodeAgent（历史变量）。

    仅校验非空，不校验格式，无效 key 由 API 认证环节暴露。
    """
    return os.environ.get("CRA_DEEPSEEK_API_KEY") or os.environ.get("SE_CodeAgent")


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


# ---- models.json：加载、校验与保存 ----


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

    POSIX 上尽力收紧文件权限为 0600（该文件含明文 key）；失败不阻塞写入，
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
    """models.json 结构校验（schema 与边界规则）；不合法抛 ConfigError。

    api_key 允许缺失或空串（回退环境变量）；其余字段类型不符、供应商名含 "/"、
    default 不是 "供应商/模型" 形态均视为配置损坏。
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
            raise ConfigError(f"配置文件结构无效：供应商名 '{name}' 不得含 '/'。")
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
                f"配置文件结构无效：default {default!r} 应为 '供应商/模型'。"
            )
        provider_name, _, model_name = default.partition("/")
        if not provider_name or not model_name:
            raise ConfigError(
                f"配置文件结构无效：default {default!r} 的供应商与模型名均不能为空。"
            )
        # 引用完整性与供应商重名：问题在 cra config 写入时即暴露（手改文件在
        # 加载时同样报错），不必等到运行期 resolve() 才失败
        entry = next((item for item in providers if item.get("name") == provider_name), None)
        if entry is None:
            raise ConfigError(
                f"配置文件结构无效：default 引用的供应商 '{provider_name}' 不存在。"
            )
        if model_name not in entry["models"]:
            raise ConfigError(
                f"配置文件结构无效：default 引用的模型 '{model_name}' 不在供应商 "
                f"'{provider_name}' 的清单内。"
            )
    names = [item.get("name") for item in providers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"配置文件结构无效：供应商名重复：{', '.join(duplicates)}。")
    collisions = conflicting_keyless_env_vars(providers)
    if collisions:
        detail = "；".join(f"{var} ← {', '.join(names)}" for var, names in sorted(collisions.items()))
        raise ConfigError(
            "配置文件结构无效：以下供应商未设置 api_key，且回退到同一个 key 环境变量"
            f"（接入点主机不同，会互相读到对方的凭据）：{detail}。"
            "请为其中一方设置 api_key，或修改供应商名使变量名可区分。"
        )


def conflicting_keyless_env_vars(providers: list[dict[str, Any]]) -> dict[str, list[str]]:
    """未设 api_key 的供应商中，回退环境变量名相同但接入点主机不同的分组。

    环境变量名归一化是有损映射（"DeepSeek" 与 "deepseek" 同名），两个都依赖
    环境变量回退的供应商撞名时，会把同一凭据发往不同接入点（key 串用）。同
    一主机（同一供应商的多端点，如 Z.ai/BigModel 的 Coding Plan 与 API 双预设
    共用官方变量名）属设计内共享，不算冲突。供应商名 → 变量名的映射经
    provider_env_var（预设名取官方变量名，自定义按归一化规则）。回退链末端
    共享的 SE_CodeAgent（历史变量）是有意保留的最后回退，不属于本函数的冲突
    判定范围——多个无 key 供应商在各自专属变量缺失时共同回退到它是设计行为。
    """
    by_var: dict[str, list[tuple[str, str]]] = {}
    for entry in providers:
        if entry.get("api_key"):
            continue
        host = urlsplit(entry.get("base_url", "")).hostname or ""
        by_var.setdefault(provider_env_var(entry["name"]), []).append((entry["name"], host))
    return {
        var: [name for name, _ in entries]
        for var, entries in by_var.items()
        if len({host for _, host in entries}) > 1
    }


def _resolve_provider(
    providers: list[dict[str, Any]], provider: str, model: str | None
) -> ResolvedModel:
    """解析到指定供应商：--model 须在其清单内匹配，缺省取 models[0]。"""
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
    api_key = resolve_provider_api_key(provider, entry.get("api_key"))
    if not api_key:
        raise ConfigError(
            f"供应商 '{provider}' 未配置 api_key，环境变量 {provider_env_var(provider)} 与 "
            "SE_CodeAgent 也未设置；请运行 cra config 补充 key，或设置环境变量。"
        )
    return ResolvedModel(
        model=model_name, base_url=entry["base_url"], api_key=api_key, provider=provider
    )


def resolve(
    *, model: str | None = None, provider: str | None = None
) -> ResolvedModel | None:
    """按配置优先级解析本次使用的模型连接。

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
                "请同时用 --provider 指定供应商。"
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
    # 环境变量回退；无供应商清单时 --model 直接覆盖 CRA_MODEL
    api_key = get_api_key()
    if not api_key:
        return None
    return ResolvedModel(
        model=model or os.environ.get("CRA_MODEL") or DEFAULT_MODEL,
        base_url=os.environ.get("CRA_BASE_URL") or DEFAULT_BASE_URL,
        api_key=api_key,
        provider=None,
    )
