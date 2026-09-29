"""代码执行工具：run_python（行为与限制见 Design.md §3.2）。

**本机直接运行，非沙箱**：子进程拥有当前用户全部权限，仅用于审查过程中
验证可信代码片段；风险以"限时 + 进程树强杀 + 输出截断 + 限制明示"控制。

相对路径相对进程启动目录（cwd）解析；子进程工作目录继承本进程（即进程启动目录）。
子进程输出捕获显式 encoding='utf-8', errors='replace'（§3.2 通用规则：
Windows 默认 cp936，否则 UTF-8 源码的中文输出会解码崩溃），并设
PYTHONIOENCODING=utf-8 保持子进程侧一致。

超时等失败以异常上抛，由注册表 dispatch 统一兜底为结构化错误文本回传模型（§5）。
"""

import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from cra import config

MAX_OUTPUT_CHARS = 2000  # §3.2：stdout/stderr 各自截断上限
MAX_TIMEOUT_SECONDS = 60  # 模型可请求的单次超时上限，防止长挂拖死会话
KILL_REAP_SECONDS = 5  # 强杀后回收子进程的有限等待；再超时则放弃回收（杀树已尽力）

_TRUNCATION_NOTICE = "…（输出超过 {limit} 字符，已截断）"
_PYTHON_SUFFIX = ".py"


def run_python(
    code: str | None = None, path: str | None = None, timeout: int | None = None
) -> str:
    """执行 Python 代码并返回退出码与 stdout/stderr（各截断至 2000 字符）。

    code 与 path 二选一；path 仅接受 .py 文件（非 Python 文件给出可操作的提示）。
    默认超时取 CRA_EXEC_TIMEOUT（10s），可用 timeout 参数覆盖（上限 60s）。
    超时/进程启动失败等异常上抛，由 dispatch 结构化回传模型。
    """
    if (code is None) == (path is None):
        return "错误：code 与 path 必须二选一（恰好提供一个）。"
    if timeout is not None and not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        return f"错误：timeout 必须是 1~{MAX_TIMEOUT_SECONDS} 的整数秒，当前值：{timeout}"

    if code is not None:
        command = [sys.executable, "-c", code]
    else:
        assert path is not None  # 二选一校验已保证
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"错误：文件不存在：{target}"
        if not target.is_file():
            return f"错误：路径不是文件：{target}（列目录请用 list_dir）"
        if target.suffix != _PYTHON_SUFFIX:
            return (
                f"错误：{target.name} 不是 Python 文件（仅支持 .py），无法执行。"
                "如需查看其内容请改用 read_file。"
            )
        command = [sys.executable, str(target)]

    effective_timeout = timeout if timeout is not None else config.get_exec_timeout()
    exit_code, stdout, stderr = _run_subprocess(command, effective_timeout)
    return _format_result(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _run_subprocess(command: list[str], timeout_seconds: int) -> tuple[int, str, str]:
    """运行子进程并收集输出；超时强杀整棵进程树后以异常上抛（§5）。

    进程启动/杀树等失败同样上抛，由 dispatch 兜底；环境继承父进程（非沙箱
    已知边界，见 §9），仅额外设 PYTHONIOENCODING 统一子进程编码。
    """
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        kill_detail = _kill_process_tree(process)
        # 杀树成功时 communicate 立即返回；失败（进程仍在跑）只有限等待后放弃，
        # 超时信息本身已回传模型，残余输出不再有价值
        with suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=KILL_REAP_SECONDS)
        detail = f"（进程树终止失败：{kill_detail}）" if kill_detail else ""
        raise RuntimeError(
            f"执行超时（{timeout_seconds}s），已强制终止进程树{detail}"
            "。若代码含死循环或长阻塞，请修复后再试。"
        ) from None
    return process.returncode, stdout or "", stderr or ""


def _kill_process_tree(process: subprocess.Popen[str]) -> str | None:
    """强杀进程树；返回失败说明，成功（或进程已自行退出）返回 None。

    Windows 下用 taskkill /T /F 连子进程一起杀（proc.kill() 只杀直接子进程，
    孙进程会残留）；其他验收平台仅 Windows，POSIX 分支 proc.kill() 兜底。
    """
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # 中文版 taskkill 输出为 GBK，只作错误详情，容错解码
            check=False,  # 失败详情在下方按进程是否仍存活判定，不走异常路径
        )
        if result.returncode != 0 and process.poll() is None:
            detail = (result.stderr or result.stdout or "").strip()
            return detail or f"taskkill 退出码 {result.returncode}"
        return None
    process.kill()
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """单路输出截断（§3.2 防刷屏）；截断处附提示让模型知道内容不完整。"""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_NOTICE.format(limit=limit)


def _format_result(exit_code: int, stdout: str, stderr: str) -> str:
    """归一为模型可读的三段结构；空输出显式标注（空与非空对模型含义不同）。"""
    stdout = _truncate(stdout.rstrip("\n"))
    stderr = _truncate(stderr.rstrip("\n"))
    return (
        f"退出码：{exit_code}\n"
        f"stdout：\n{stdout if stdout else '（空）'}\n"
        f"stderr：\n{stderr if stderr else '（空）'}"
    )
