"""共享 fixture:用户目录隔离(config/sessions 的 ~/.cra 路径随 HOME/USERPROFILE 落到 tmp)。"""

from pathlib import Path

import pytest


@pytest.fixture
def home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把用户目录指到 tmp_path:Path.home() 在 Windows 读 USERPROFILE、POSIX 读 HOME。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path
