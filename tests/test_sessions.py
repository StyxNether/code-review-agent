"""sessions 测试:名称合法性(防路径穿越)、读写往返与坏文件兜底(home_dir 隔离)。"""

import json
from pathlib import Path

import pytest

from cra import sessions


class TestSessionPath:
    def test_valid_name_resolves_under_sessions_dir(self, home_dir: Path) -> None:
        path = sessions.session_path("my-session_1.v2")

        assert path == home_dir / ".cra" / "sessions" / "my-session_1.v2.json"

    @pytest.mark.parametrize("bad", ["", "../escape", "a/b", "有中文", " leading-space", ".hidden"])
    def test_invalid_name_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="非法"):
            sessions.session_path(bad)

    @pytest.mark.parametrize("reserved", ["CON", "con.json", "NUL", "com1", "lpt1", "name."])
    def test_windows_reserved_and_trailing_dot_rejected(self, reserved: str) -> None:
        """自审修复:Windows 保留设备名与以点结尾的名称在写入时必失败,前置拒绝。"""
        with pytest.raises(ValueError, match="不可用"):
            sessions.session_path(reserved)


class TestSaveLoad:
    def test_roundtrip(self, home_dir: Path) -> None:
        path = sessions.session_path("roundtrip")
        data = [{"type": "human", "data": {"content": "审查 x.py"}}]

        sessions.save_session(path, data)

        assert sessions.load_session(path) == data

    def test_save_creates_missing_dir(self, home_dir: Path) -> None:
        path = home_dir / ".cra" / "sessions" / "nested" / "x.json"

        sessions.save_session(path, [])

        assert path.exists()

    def test_load_rejects_non_array(self, home_dir: Path) -> None:
        path = sessions.session_path("bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")

        with pytest.raises(ValueError, match="消息数组"):
            sessions.load_session(path)

    def test_load_rejects_broken_json(self, home_dir: Path) -> None:
        path = sessions.session_path("worse")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            sessions.load_session(path)
