"""文件系统工具：list_dir 与 read_file。

所有 handler 返回字符串（错误也是字符串，数据封装）；
相对路径一律相对进程启动目录（cwd）解析，不做"项目根"推断。
"""

from pathlib import Path

MAX_READ_LINES = 400  # 单次读取上限，超出提示模型用 start/end 分段
MAX_READ_BYTES = 10 * 1024 * 1024  # 单文件读取上限：读取本身须有界（先判大小再读，防大文件进内存）

# list_dir 忽略这些目录与所有点开头的隐藏目录
IGNORED_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache"}


def split_lines(text: str) -> list[str]:
    """按真实换行符（\r\n、\r、\n）切分行；尾部换行不产生空尾行（与 splitlines 一致）。

    不用 str.splitlines()：它把 \\x0c（form feed）、U+2028 等 Unicode 换行类
    字符也当行界，而行号口径（编辑器、审查 location）只认真实换行——含这些
    字符的文件会让 read_file 与审查 prompt 的行号整体漂移。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _resolve(path: str) -> Path:
    """相对路径按 cwd 解析；不支持跨驱动器的绝对路径检查交给 Path。"""
    return Path(path).expanduser().resolve()


def list_dir(path: str = ".") -> str:
    """返回目录的两层树（本层 + 每个子目录的一层），忽略忽略名单与隐藏目录。"""
    root = _resolve(path)
    if not root.exists():
        return f"错误：路径不存在：{root}"
    if not root.is_dir():
        return f"错误：路径不是目录：{root}（读文件请用 read_file）"

    lines = [f"{root}"]
    try:
        entries = sorted(
            root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError as exc:
        return f"错误：无法列出目录 {root}：{exc}"

    for entry in entries:
        if entry.is_dir() and (entry.name in IGNORED_DIRS or entry.name.startswith(".")):
            continue
        if entry.is_dir():
            lines.append(f"├── {entry.name}/")
            lines.extend(_list_child_dir(entry))
        else:
            lines.append(f"├── {entry.name}")
    return "\n".join(lines)


def _list_child_dir(directory: Path) -> list[str]:
    """子目录的一层内容（第二层），同样应用忽略规则；不递归第三层。"""
    try:
        children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return ["│   └── （无法读取该子目录）"]
    lines: list[str] = []
    for child in children:
        if child.is_dir() and (child.name in IGNORED_DIRS or child.name.startswith(".")):
            continue
        suffix = "/" if child.is_dir() else ""
        lines.append(f"│   ├── {child.name}{suffix}")
    if not lines:
        lines.append("│   ├── （空目录）")
    return lines


def read_file(path: str, start: int | None = None, end: int | None = None) -> str:
    """带行号读取文本文件；1-based 闭区间 [start, end]，单次上限 MAX_READ_LINES 行。

    二进制文件返回结构化错误（null 字节嗅探：errors='replace' 会把二进制读成
    乱码刷爆上下文，嗅探才能给模型可操作的提示）。
    """
    target = _resolve(path)
    if not target.exists():
        return f"错误：文件不存在：{target}"
    if not target.is_file():
        return f"错误：路径不是文件：{target}（列目录请用 list_dir）"

    try:
        # 先判大小再读：上限必须在读取之前生效，否则大文件已整文件进内存
        size = target.stat().st_size
    except OSError as exc:
        return f"错误：无法读取文件 {target}：{exc}"
    if size > MAX_READ_BYTES:
        return (
            f"错误：{target.name} 约 {size / 1048576:.1f} MB，超过单次读取上限"
            f"（{MAX_READ_BYTES // 1048576} MB）。请让用户指定要审查的具体分段，"
            "或改用 run_python 等方式提取所需内容。"
        )
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"错误：无法读取文件 {target}：{exc}"
    if b"\x00" in raw:
        return (
            f"错误：{target.name} 是二进制文件（含 null 字节），无法按文本读取。"
            "请勿继续读取该文件；如需了解其结构请改用其他方式（如 file 命令或让用户提供说明）。"
        )

    text = raw.decode("utf-8", errors="replace")
    lines = split_lines(text)
    total = len(lines)
    if total == 0:
        # 空文件没有行区间可言，按区间规则会误报"start 不能大于 end"
        return f"{target}（空文件，共 0 行）"

    start_line, end_line = _normalize_range(start, end, total)
    error = _range_error(start, end, start_line, end_line, total)
    if error is not None:
        return error

    selected = lines[start_line - 1 : end_line]
    numbered = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(selected, start=start_line))
    header = f"{target}（共 {total} 行，显示 {start_line}-{end_line} 行）"
    output = f"{header}\n{numbered}"
    if end_line < total and end_line - start_line + 1 == MAX_READ_LINES:
        output += (
            f"\n（已按单次 {MAX_READ_LINES} 行上限截断：请用 start={end_line + 1} 继续读取）"
        )
    return output


def _normalize_range(start: int | None, end: int | None, total: int) -> tuple[int, int]:
    """把可选的 start/end 归一为 1-based 闭区间；end 超出文件按 EOF 截齐。"""
    start_line = 1 if start is None else start
    end_line = min(start_line + MAX_READ_LINES - 1, total) if end is None else end
    end_line = min(end_line, total)
    return start_line, end_line


def _range_error(
    start: int | None, end: int | None, start_line: int, end_line: int, total: int
) -> str | None:
    """非法区间返回结构化错误文本，合法区间返回 None。"""
    if (start is not None and start < 1) or (end is not None and end < 1):
        return f"错误：start/end 必须是 1-based 正整数（文件共 {total} 行），当前 start={start}, end={end}"
    if start is not None and start_line > total:
        return f"错误：start={start} 超出文件范围（文件共 {total} 行）"
    if start_line > end_line:
        return (
            f"错误：start 不能大于 end（1-based 闭区间，文件共 {total} 行），"
            f"当前 start={start}, end={end}"
        )
    if end_line - start_line + 1 > MAX_READ_LINES:
        return (
            f"错误：请求区间 {start_line}-{end_line} 共 {end_line - start_line + 1} 行，"
            f"超过单次 {MAX_READ_LINES} 行上限，请缩小 start/end 范围分段读取"
        )
    return None
