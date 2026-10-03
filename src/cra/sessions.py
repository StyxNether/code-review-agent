"""会话持久化的文件系统侧：/save /load 落在 ~/.cra/sessions/。

文件内容是消息数组（JSON）。消息 ↔ dict 的序列化在 agent 层完成（框架类型不出
agent 层），本模块只负责名称合法性、路径与 JSON 读写。
"""

import json
import os
import re
from pathlib import Path
from typing import Any

# 名称仅限安全字符：会话名直接拼进文件路径，禁止分隔符使任何合法名称（含
# ".."）都只能是单个路径成分、无法穿越；首字符允许下划线——自动保留存档
# 固定名 _last 落在同一命名空间
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
# Windows 保留设备名（大小写不敏感）：作为文件名主干会导致写入失败或行为异常
_WINDOWS_RESERVED = frozenset(
    ["con", "nul", "prn", "aux", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))]
)


def sessions_dir() -> Path:
    """会话文件目录：~/.cra/sessions/。"""
    return Path.home() / ".cra" / "sessions"


def session_path(name: str) -> Path:
    """会话名称 → 文件路径；非法名称（空串/特殊字符/路径穿越/保留设备名）抛 ValueError。"""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"会话名称 {name!r} 非法：仅限字母、数字、下划线、连字符与点，且以字母、数字或下划线开头。"
        )
    if name.endswith(".") or name.split(".")[0].lower() in _WINDOWS_RESERVED:
        raise ValueError(f"会话名称 {name!r} 不可用：Windows 保留设备名或以点结尾。")
    return sessions_dir() / f"{name}.json"


def save_session(path: Path, data: list[dict[str, Any]]) -> None:
    """把消息数组写入会话文件（目录不存在则创建）；原子替换，防写一半损坏。

    POSIX 上尽力收紧为 0600：会话内容可能内嵌被审查的源码，敏感度与配置文件
    相当；失败不阻塞保存（与 config.save_raw 的权限处理同口径）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.name == "posix":
        try:
            tmp.chmod(0o600)
        except OSError:
            pass  # 权限收紧失败不阻塞保存
    os.replace(tmp, path)


def load_session(path: Path) -> list[dict[str, Any]]:
    """读取会话文件并校验外层结构（消息数组）。

    JSON 损坏或外层结构非法抛 ValueError；文件不存在/不可读抛 OSError，
    调用方按各自场景分别处理。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"会话文件 {path} 格式无效：应为消息数组。")
    return raw
