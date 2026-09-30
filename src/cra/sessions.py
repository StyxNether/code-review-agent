"""会话持久化的文件系统侧：/save /load 落在 ~/.cra/sessions/（§7.3）。

文件内容是消息数组（JSON）。消息 ↔ dict 的序列化在 agent 层完成（框架类型不出
agent 层，§3.4），本模块只负责名称合法性、路径与 JSON 读写。
"""

import json
import re
from pathlib import Path
from typing import Any

# 名称仅限安全字符：会话名直接拼进文件路径，拒绝路径分隔符与 ".." 穿越写点
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def sessions_dir() -> Path:
    """会话文件目录：~/.cra/sessions/。"""
    return Path.home() / ".cra" / "sessions"


def session_path(name: str) -> Path:
    """会话名称 → 文件路径；非法名称（空串/特殊字符/路径穿越）抛 ValueError。"""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"会话名称 {name!r} 非法：仅限字母、数字、下划线、连字符与点，且以字母数字开头。"
        )
    return sessions_dir() / f"{name}.json"


def save_session(path: Path, data: list[dict[str, Any]]) -> None:
    """把消息数组写入会话文件（覆盖写；目录不存在则创建）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(path: Path) -> list[dict[str, Any]]:
    """读取会话文件并校验外层结构（消息数组）；损坏抛 ValueError。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"会话文件 {path} 格式无效：应为消息数组。")
    return raw
