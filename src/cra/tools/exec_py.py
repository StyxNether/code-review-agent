"""代码执行工具：run_python。

**本机直接运行，非沙箱**：子进程拥有当前用户全部权限，仅用于审查过程中
验证可信代码片段；风险以"限时 + 进程树强杀 + 输出截断 + 环境净化 + 限制明示"控制。

相对路径相对进程启动目录（cwd）解析；子进程工作目录继承本进程（即进程启动目录）。
子进程输出捕获显式 encoding='utf-8', errors='replace'（Windows 默认 cp936，
否则 UTF-8 源码的中文输出会解码崩溃），并设
PYTHONIOENCODING=utf-8 保持子进程侧一致；环境经净化后才继承（剔除凭证变量，
防被审查代码经环境读取 key 后经输出回传模型上下文）。

超时等失败以异常上抛，由注册表 dispatch 统一兜底为结构化错误文本回传模型。
"""

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from cra import config

MAX_OUTPUT_CHARS = 2000  # stdout/stderr 各自截断上限
MAX_TIMEOUT_SECONDS = 60  # 模型可请求的单次超时上限，防止长挂拖死会话
KILL_REAP_SECONDS = 5  # 强杀后回收子进程的有限等待；再超时则放弃回收（杀树已尽力）

_TRUNCATION_NOTICE = "…（输出超过 {limit} 字符，已截断）"
_PYTHON_SUFFIX = ".py"


def _is_credential_var(name: str) -> bool:
    """环境变量名是否属凭证类（独立成函数便于测试与扩展名单）。

    大小写不敏感比对：Windows 环境变量名不区分大小写（同名全大写变体确实
    存在，精确匹配过滤会漏）；连字符先归一为下划线（Windows 变量名合法字符，
    归一后覆盖 MY-API-KEY 型命名）；POSIX 侧为大小写敏感，不敏感过滤只会多
    剔除无关变量，属安全方向的超集。名称黑名单无法覆盖"名字无害、值是凭证"
    的变量——这是本控制的已知边界。
    """
    normalized = name.upper().replace("-", "_")
    return normalized == "SE_CODEAGENT" or "API_KEY" in normalized or "APIKEY" in normalized


def _sanitized_env() -> dict[str, str]:
    """父环境剔除凭证变量后的副本（key 不得进入子进程环境）。

    被审查/被生成的代码可 print(os.environ) 读走环境变量，其输出会流式渲染并
    随会话留存于模型上下文——剔除 SE_CodeAgent 与一切 *API_KEY* 型变量名，
    保证代理自身凭证不进入该通道。非凭证类变量（PATH 等）原样保留。
    """
    return {name: value for name, value in os.environ.items() if not _is_credential_var(name)}


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
        # Windows 文件系统大小写不敏感：UPPER.PY 也是 Python 文件
        if target.suffix.lower() != _PYTHON_SUFFIX:
            return (
                f"错误：{target.name} 不是 Python 文件（仅支持 .py），无法执行。"
                "如需查看其内容请改用 read_file。"
            )
        command = [sys.executable, str(target)]

    effective_timeout = timeout if timeout is not None else config.get_exec_timeout()
    if not 0 < effective_timeout <= MAX_TIMEOUT_SECONDS:
        # 配置来源（CRA_EXEC_TIMEOUT）与模型传参同标准校验：
        # 非法配置不应伪装成"执行超时"，而应给出指向配置的可读错误
        source = "timeout 参数" if timeout is not None else "环境变量 CRA_EXEC_TIMEOUT"
        return (
            f"错误：{source}必须是 1~{MAX_TIMEOUT_SECONDS} 的整数秒，"
            f"当前值：{effective_timeout}"
        )
    exit_code, stdout, stderr = _run_subprocess(command, effective_timeout)
    return _format_result(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _run_subprocess(command: list[str], timeout_seconds: int) -> tuple[int, str, str]:
    """运行子进程并收集输出；任何退出路径都杀树并有限回收。

    超时强杀整棵进程树后以异常上抛，由 dispatch 兜底；Ctrl+C 等中断同样先杀树、
    再有限等待回收管道，然后上抛原始异常——不留残留进程与管道；
    杀树失败时回收等待超时后放弃（有限等待，不阻塞中断响应）。
    """
    environment = {**_sanitized_env(), "PYTHONIOENCODING": "utf-8"}
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        # POSIX：独立进程组启动，支撑 killpg 整树强杀（见 _kill_process_tree）
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        **popen_kwargs,
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
    except BaseException:
        # 中断/请求失败等路径：先杀树并回收管道再上抛，不掩盖中断本身
        with suppress(Exception):
            _kill_process_tree(process)
        with suppress(Exception, subprocess.TimeoutExpired):
            process.communicate(timeout=KILL_REAP_SECONDS)
        raise
    return process.returncode, stdout or "", stderr or ""


def _kill_process_tree(process: subprocess.Popen[str]) -> str | None:
    """强杀进程树；返回失败说明，成功（或进程已自行退出）返回 None。

    Windows 下用 taskkill /T /F 连子进程一起杀（proc.kill() 只杀直接子进程，
    孙进程会残留）；POSIX 下子进程以独立进程组启动（start_new_session），对
    整组 SIGKILL 同样覆盖孙进程（proc.kill() 会残留孙进程；
    子孙若自建进程组则仍不可及，与 taskkill /T 的边界一致）。
    """
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # 中文版 taskkill 输出为 GBK，只作错误详情，容错解码
                check=False,  # 失败详情在下方按进程是否仍存活判定，不走异常路径
            )
        except Exception as killed:  # noqa: BLE001 — 杀树是兜底路径，任何失败都转为详情文本，不掩盖调用方的原始异常
            return f"taskkill 调用失败：{killed}"
        if result.returncode != 0 and process.poll() is None:
            detail = (result.stderr or result.stdout or "").strip()
            return detail or f"taskkill 退出码 {result.returncode}"
        return None
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """单路输出截断（防刷屏）；截断处附提示让模型知道内容不完整。"""
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
