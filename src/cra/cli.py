"""cra 命令行入口：argparse（chat/review/config）+ REPL + rich 流式彩色渲染。

cli 只管交互与渲染（§3.4）：LLM 循环与工具执行一律经 Agent；从 llm 导入的只有
LLMError 类型与 config test 专用的 probe 探测（§7.4，无 Agent 循环的最小通路）、
从 tools 导入的只有 ERROR_PREFIX 常量，不直接执行工具。
"""

import argparse
import getpass
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from cra import config, llm, sessions
from cra.agent import MAX_TOOL_ROUNDS, Agent, ExecConfirmCallback
from cra.llm import LLMError
from cra.tools import ERROR_PREFIX

_TOOL_ARGS_PREVIEW_LIMIT = 80
# 分级标记的着色（M2 验收反馈：严重红 / 一般黄 / 建议绿）
_SEVERITY_STYLES = {"严重": "bold red", "一般": "bold yellow", "建议": "bold green"}
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# 标记识别（行内任意位置）：兼容模型自行输出的 ** 包裹，星号只被消耗不参与渲染
_MARKER_RE = re.compile(r"\*{0,2}(【(?:严重|一般|建议)】|修改建议[：:])\*{0,2}")
_VALID_SEVERITIES = ("严重", "一般", "建议")
_MAX_REVIEW_LINES = 2000  # 单文件嵌入 prompt 的行数上限：防超大文件 token 爆炸（超出截断并明示）
_DEFAULT_EXPORT_NAME = "cra-export.md"

_REPL_HELP = """\
REPL 内置命令：
  /help                          显示本帮助
  /clear                         清空上下文（系统提示词保留）
  /change model（别名 /model）    列出供应商与模型并切换（上下文保留）
  /save [名称]                   保存会话到 ~/.cra/sessions/<名称>.json
  /load <名称>                   从文件恢复会话（整体替换当前会话）
  /undo                          作废上一轮对话
  /context                       显示 token 用量与距截断阈值的余量
  /export [文件]                 导出最近一轮结论为 Markdown（默认 cra-export.md）
  /confirm [on|off]              run_python 执行前逐次确认（默认 off）
  exit 或 Ctrl+C                 退出
其余 /xxx 按普通文本发送给模型。"""

_CONFIG_FILE_NOTICE = (
    "⚠ 配置已写入 {path}。该文件以明文保存 API key，属本机敏感数据：\n"
    "  - 注意文件访问权限：POSIX 已自动收紧为 0600；Windows 下请勿放在共享目录；\n"
    "  - 备份该文件时同样注意保密：勿上传到仓库、网盘或聊天工具。"
)


def _marker_style(marker: str) -> str:
    """标记 → rich 样式：分级标记按级别着色加粗，修改建议加粗。"""
    if marker.startswith("【"):
        return _SEVERITY_STYLES[marker[1:-1]]
    return "bold"


def _fence_toggle(line: str, state: tuple[str, int] | None) -> tuple[str, int] | None:
    """CommonMark 围栏状态机：state 为 (围栏字符, 长度)，None 表示不在围栏内。"""
    match = _FENCE_RE.match(line)
    if match is None:
        return state
    marker = match.group(1)
    if state is None:
        return (marker[0], len(marker))  # 开围栏（允许 info string）
    if marker[0] == state[0] and len(marker) >= state[1] and not line[match.end() :].strip():
        return None  # 闭围栏：同字符、长度不短于开头、行内无其他内容
    return state


def _in_fence(text: str) -> bool:
    """text（可为消息前缀）的最后一行之后是否仍处于围栏代码块内部。"""
    state: tuple[str, int] | None = None
    for line in text.split("\n"):
        state = _fence_toggle(line, state)
    return state is not None


def _short(text: str, limit: int = _TOOL_ARGS_PREVIEW_LIMIT) -> str:
    """单行预览：压掉换行并截断，避免工具参数刷屏。"""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _setup_hint() -> str:
    """key 设置命令按当前平台给出（M3 挂账：指引跨平台化）。"""
    if sys.platform == "win32":
        return '设置方法（Windows）：setx SE_CodeAgent "sk-你的key"，然后新开终端重试。'
    return (
        '设置方法（macOS/Linux）：export SE_CodeAgent="sk-你的key"'
        "（写入 ~/.bashrc 或 ~/.zshrc 可持久化）。"
    )


def _local_now() -> datetime:
    """本地时区当前时间（报告/导出的时间戳；DTZ 规则要求显式 tz 语义）。"""
    return datetime.now().astimezone()


def _print_config_guide(console: Console) -> None:
    """无任何可用配置时的引导（§5/§7.4；退出码 2 由调用方执行）。"""
    try:
        providers = config.load_raw().get("providers") or []
    except config.ConfigError:
        providers = []
    if providers:
        console.print(
            "已配置供应商但未设置默认模型，且环境变量 SE_CodeAgent 也未设置。", markup=False
        )
        console.print("请运行 cra config 设置默认，或用 --model/--provider 临时指定。", markup=False)
        return
    console.print(
        "未找到任何可用配置：既无 ~/.cra/models.json，也未设置环境变量 SE_CodeAgent。",
        markup=False,
    )
    console.print("请先完成以下任一配置：", markup=False)
    console.print("  1. 运行 cra config 进入交互式配置（推荐，支持多供应商与连通性自检）", markup=False)
    console.print(f"  2. {_setup_hint()}", markup=False)


class _StreamingMarkdown:
    """§3.1 流式渲染：已完成的段落用 rich Markdown 逐段打印，终块在流结束时渲染。

    不使用 rich.Live（Windows 传统控制台下 Live 重绘不可靠）。
    段落切分避开 ``` 围栏内部，防止代码块被空行拆碎。
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._buffer = ""
        self._has_output = False
        self._pending_break = False

    @property
    def has_output(self) -> bool:
        """本轮是否已有任何可见输出（增量或冲洗的叙述）。"""
        return self._has_output

    def mark_break(self) -> None:
        """在下一个段落前插入空行（工具打点与正文之间留出呼吸空间）。"""
        self._pending_break = True

    def feed(self, delta: str) -> None:
        """接收一条文本增量，把其中已完成的段落渲染出去。"""
        self._buffer += delta
        while True:
            split_at = self._paragraph_end()
            if split_at == -1:
                return
            segment, self._buffer = self._buffer[:split_at], self._buffer[split_at + 2 :]
            self._print_segment(segment)

    def finish(self) -> None:
        """流结束后渲染终块（最后一段）；中断/报错时也已收到的部分可见。"""
        if self._buffer.strip():
            self._print_segment(self._buffer)
        self._buffer = ""

    def flush_pending(self) -> None:
        """把缓冲中未成段的文本按暗色平文落盘（工具轮伴随叙述的分流）。

        过程性叙述通常不构成完整段落（无空行），若只等段落完成会跨工具轮累积、
        与最终回答一起在流结束时批量出现（M4 验收反馈）；在工具打点前冲洗，叙述
        即实时跟随其轮次显示。平文暗色：过程文本不作 Markdown 渲染、与正文区分。
        """
        if self._buffer.strip():
            if self._has_output or self._pending_break:
                self._console.print()
            self._console.print(self._buffer.strip(), style="dim", markup=False, highlight=False)
            self._has_output = True
        self._buffer = ""

    def _paragraph_end(self) -> int:
        """下一个不在围栏内的空行（双换行）位置；没有则返回 -1。

        围栏判定用行级状态机（``` / ~~~ / 更长的围栏均识别），
        替换早先 count("```") 奇偶法对非标准围栏的漏判。
        """
        index = self._buffer.find("\n\n")
        while index != -1:
            if not _in_fence(self._buffer[:index]):
                return index
            index = self._buffer.find("\n\n", index + 1)
        return -1

    def _print_segment(self, text: str) -> None:
        if not text.strip():
            return
        if self._has_output or self._pending_break:
            self._console.print()  # 段落之间空一行（M1 验收反馈：避免文字堆叠）
        self._render_lines(text)
        self._has_output = True
        self._pending_break = False

    def _render_lines(self, text: str) -> None:
        """逐行分级渲染（M2 验收反馈：分级标记着色、修改建议加粗）。

        含标记的行按标记切片：标记本体着色/加粗、文本片走单行 Markdown
        （保留行内代码渲染），同行拼接；不含标记的行（含围栏内部）聚合为
        Markdown 块，保持既有渲染不变。不往 Markdown 里插 **——CommonMark
        的 flanking 规则下"中文**【标记】**中文"不会解析为加粗（修复轮实测）。
        """
        state: tuple[str, int] | None = None
        chunk: list[str] = []
        for line in text.split("\n"):
            state = _fence_toggle(line, state)
            if state is not None or not _MARKER_RE.search(line):
                chunk.append(line)
            else:
                self._flush_chunk(chunk)
                self._render_marked_line(line)
        self._flush_chunk(chunk)

    def _flush_chunk(self, chunk: list[str]) -> None:
        if not any(line.strip() for line in chunk):
            return
        self._console.print(Markdown("\n".join(chunk)))

    def _render_marked_line(self, line: str) -> None:
        """单行按标记切片渲染：文本片经 Markdown 渲染为 Text 片段、标记片着色，
        拼为单行 Text 一次打印（验收走查修复：Markdown 片段自带换行与全宽填充，
        逐片 print 会把行内标记拆成多行）。"""
        line_text = Text()
        position = 0
        for match in _MARKER_RE.finditer(line):
            before = line[position : match.start()]
            if before:
                line_text += self._markdown_fragment(before)
                if before[-1].isspace():
                    line_text.append(" ")  # 片段行尾被收敛，恢复原文的词间空格
            marker = match.group(1)
            line_text.append(marker, style=_marker_style(marker))
            position = match.end()
        tail = line[position:]
        if tail.strip():
            if tail[0].isspace() and position > 0:
                line_text.append(" ")  # 行中的尾片同理恢复首空格（行首缩进仍交 Markdown 语义）
            line_text += self._markdown_fragment(tail)
        self._console.print(line_text)

    def _markdown_fragment(self, text: str) -> Text:
        """单行 Markdown 文本片 → 带样式 Text（保留行内代码等渲染），供同行拼接。

        Markdown 块渲染会把行右侧填充到控制台全宽且自带换行，直接 print 会拆行；
        render_lines 取回段后收敛行尾空白，使多片可拼回一行（超长仍由 rich 软换行）。
        """
        rendered = Text()
        for fragment_line in self._console.render_lines(
            Markdown(text), self._console.options, pad=False, new_lines=False
        ):
            for segment in fragment_line:
                if segment.text:
                    rendered.append(segment.text, style=segment.style)
        rendered.rstrip()  # 原地收敛 Markdown 的全宽填充（返回 None，不可链式）
        return rendered


def _run_turn(agent: Agent, console: Console, user_input: str, *, no_stream: bool) -> str | None:
    """执行一轮对话：工具调用打点、最终回答流式渲染、中断与请求错误兜底。

    返回本轮最终文本（供 /export 记录最近一轮结论）；中断或失败返回 None。
    no_stream 时不接收增量，待 run 返回后整体渲染（§7.2 --no-stream）。
    """
    streamer = _StreamingMarkdown(console)

    def on_tool(name: str, args: str) -> None:
        streamer.flush_pending()  # 过程性叙述先于工具打点落盘，不再与最终回答堆叠
        console.print(f"⚙ {name}({_short(args)})", style="dim", markup=False, highlight=False)

    def on_tool_result(result: str) -> None:
        if result.startswith(ERROR_PREFIX):
            console.print("  ↳ 工具返回错误（已回传模型）", style="dim", markup=False)
        else:
            console.print(f"  ↳ 完成（{len(result)} 字符）", style="dim", markup=False)
        streamer.mark_break()

    try:
        result = agent.run(
            user_input,
            on_tool=on_tool,
            on_tool_result=on_tool_result,
            on_delta=None if no_stream else streamer.feed,
        )
        if no_stream or (result and not streamer.has_output):
            # --no-stream：整体渲染；兼容端点可能不产生文本增量：界面不能因此
            # 空白，兜底渲染完整返回文本
            streamer.feed(result)
        return result
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ 本轮已中断作废。[/]")
    except LLMError as exc:
        console.print("[请求失败] " + str(exc), style="red", markup=False)
        console.print("请检查网络与 API key。", style="dim", markup=False)
    except Exception as exc:  # noqa: BLE001 — REPL 边界兜底：意外异常也不终止会话（§3.4）
        console.print(f"[意外错误] {type(exc).__name__}: {exc}", style="red", markup=False)
        console.print("本轮已终止，上下文保留；可重试或输入 exit 退出。", style="dim", markup=False)
    finally:
        streamer.finish()
        console.print()
    return None


@dataclass
class _ReplState:
    """REPL 会话级状态：内置命令族共享（agent 之外的可变面）。"""

    agent: Agent
    console: Console
    no_stream: bool
    confirm_flag: dict[str, bool]  # 执行确认开关（闭包读取的可变共享，/confirm 切换）
    last_answer: str | None = None


def _exec_confirm_callback(console: Console, flag: dict[str, bool]) -> ExecConfirmCallback:
    """run_python 执行确认回调（§7.2）：开关关闭时自动放行，开启时逐次询问。"""

    def confirm(args: dict[str, Any]) -> bool:
        if not flag.get("on"):
            return True
        preview = str(args.get("code") or args.get("path") or "")
        console.print(
            "⚠ Agent 请求执行 Python 代码（本机直接运行，非沙箱）：" + _short(preview),
            style="yellow",
            markup=False,
        )
        try:
            answer = input("允许执行?[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    return confirm


def _model_label(resolved: config.ResolvedModel) -> str:
    """欢迎语的模型标识：文件配置显示 供应商/模型，环境变量回退注明来源。"""
    if resolved.provider is None:
        return f"{resolved.model}（环境变量）"
    return f"{resolved.provider}/{resolved.model}"


def _resolve_or_exit(
    console: Console, model: str | None, provider: str | None
) -> config.ResolvedModel:
    """解析本次使用的模型连接；无配置打印引导、损坏打印错误，均退出 2（§5/§7.5）。"""
    try:
        resolved = config.resolve(model=model, provider=provider)
    except config.ConfigError as exc:
        console.print(f"配置错误：{exc}", style="red", markup=False)
        sys.exit(2)
    if resolved is None:
        _print_config_guide(console)
        sys.exit(2)
    return resolved


def _cmd_change_model(state: _ReplState) -> None:
    """/change model（§7.3）：列出供应商与模型清单、序号选择；切换后上下文保留。"""
    console = state.console
    try:
        data = config.load_raw()
    except config.ConfigError as exc:
        console.print(f"配置错误：{exc}", markup=False)
        return
    providers = data.get("providers") or []
    if not providers:
        console.print("尚未配置任何供应商清单。请先运行 cra config 添加供应商与模型。", markup=False)
        return
    console.print("可用供应商：", markup=False)
    for index, entry in enumerate(providers, start=1):
        console.print(f"  {index}. {entry['name']}　模型：{', '.join(entry['models'])}", markup=False)
    provider_raw = input("选择供应商序号：").strip()
    try:
        entry = providers[int(provider_raw) - 1]
    except (ValueError, IndexError):
        console.print("无效序号，已取消切换。", markup=False)
        return
    models = entry["models"]
    console.print(f"供应商 '{entry['name']}' 的模型：", markup=False)
    for index, name in enumerate(models, start=1):
        console.print(f"  {index}. {name}", markup=False)
    model_raw = input("选择模型序号：").strip()
    try:
        model_name = models[int(model_raw) - 1]
    except (ValueError, IndexError):
        console.print("无效序号，已取消切换。", markup=False)
        return
    try:
        resolved = config.resolve(provider=entry["name"], model=model_name)
        state.agent.switch_model(resolved)
    except (config.ConfigError, LLMError) as exc:
        console.print(f"切换失败：{exc}", markup=False)
        return
    console.print(f"已切换到 {resolved.provider}/{resolved.model}（上下文保留）。", markup=False)


def _cmd_save(state: _ReplState, name: str) -> None:
    """/save（§7.3）：会话消息数组写入 ~/.cra/sessions/<名称>.json；缺省名用时间戳。"""
    if not name:
        name = _local_now().strftime("%Y%m%d-%H%M%S")
    try:
        path = sessions.session_path(name)
    except ValueError as exc:
        state.console.print(str(exc), markup=False)
        return
    data = state.agent.dump_session()
    try:
        sessions.save_session(path, data)
    except OSError as exc:
        state.console.print(f"会话保存失败：{exc}", markup=False)
        return
    state.console.print(f"会话已保存：{path}（共 {len(data)} 条对话消息）", markup=False)


def _cmd_load(state: _ReplState, name: str) -> None:
    """/load（§7.3）：从会话文件整体替换当前会话；系统提示词保持当前版本。"""
    if not name:
        state.console.print("用法：/load <名称>（文件位于 ~/.cra/sessions/<名称>.json）", markup=False)
        return
    try:
        path = sessions.session_path(name)
    except ValueError as exc:
        state.console.print(str(exc), markup=False)
        return
    if not path.exists():
        state.console.print(f"会话文件不存在：{path}", markup=False)
        return
    try:
        data = sessions.load_session(path)
        state.agent.load_session(data)
    except Exception as exc:  # noqa: BLE001 — 文件损坏/格式非法都转为可读提示，REPL 存活
        state.console.print(f"会话加载失败：{exc}", markup=False)
        return
    state.console.print(f"会话已加载：{path}（共 {len(data)} 条消息，当前会话已被整体替换）", markup=False)


def _cmd_undo(state: _ReplState) -> None:
    """/undo（§7.3）：作废上一轮对话，上下文回滚到该轮开始前。"""
    if state.agent.undo_turn():
        state.console.print("[dim]已作废上一轮对话（上下文回滚）。[/]")
    else:
        state.console.print("没有可作废的对话轮次。", markup=False)


def _cmd_context(state: _ReplState) -> None:
    """/context（§7.3）：token 用量与距截断阈值的余量。"""
    status = state.agent.context_status()
    used, limit = int(status["used"]), int(status["limit"])
    source = "字符估算" if status["estimated"] else "真实 usage"
    state.console.print(
        f"上下文约 {used} tokens / 截断阈值 {limit}（{source}），余量 {limit - used} tokens。",
        markup=False,
    )


def _cmd_export(state: _ReplState, filename: str) -> None:
    """/export（§7.3）：最近一轮结论导出 Markdown；缺省 cra-export.md。"""
    if state.last_answer is None:
        state.console.print("尚无可导出的结论：先完成一轮对话再 /export。", markup=False)
        return
    target = Path(filename) if filename else Path(_DEFAULT_EXPORT_NAME)
    markdown = (
        f"# cra 审查结论\n\n> 导出时间：{_local_now():%Y-%m-%d %H:%M:%S}\n\n"
        f"{state.last_answer}\n"
    )
    try:
        target.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        state.console.print(f"导出失败：{exc}", markup=False)
        return
    state.console.print(f"最近一轮结论已导出：{target.resolve()}", markup=False)


def _cmd_confirm(state: _ReplState, argument: str) -> None:
    """/confirm（§7.3）：run_python 执行确认开关；省略参数显示当前状态。"""
    arg = argument.lower()
    if arg == "on":
        state.confirm_flag["on"] = True
    elif arg == "off":
        state.confirm_flag["on"] = False
    elif arg != "":
        state.console.print("用法：/confirm on|off（省略显示当前状态）。", markup=False)
        return
    current = "开" if state.confirm_flag.get("on") else "关"
    state.console.print(f"run_python 执行确认：{current}（默认 off；--ask-exec 可在启动时开启）。", markup=False)


def _handle_command(user_input: str, state: _ReplState) -> bool:
    """内置 REPL 命令分发（§7.3）；返回 False 表示按普通文本交给模型。"""
    if not user_input.startswith("/"):
        return False
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    if command == "/help":
        state.console.print(_REPL_HELP, markup=False, highlight=False)
    elif command == "/clear":
        state.agent.reset()
        state.console.print("[dim]上下文已清空（系统提示词保留）。[/]")
    elif command in ("/change", "/model"):
        if command == "/change" and argument != "model":
            state.console.print("用法：/change model（别名 /model）。", markup=False)
        else:
            _cmd_change_model(state)
    elif command == "/save":
        _cmd_save(state, argument)
    elif command == "/load":
        _cmd_load(state, argument)
    elif command == "/undo":
        _cmd_undo(state)
    elif command == "/context":
        _cmd_context(state)
    elif command == "/export":
        _cmd_export(state, argument)
    elif command == "/confirm":
        _cmd_confirm(state, argument)
    else:
        return False
    return True


def _run_chat(
    *,
    model: str | None = None,
    provider: str | None = None,
    no_stream: bool = False,
    max_rounds: int = MAX_TOOL_ROUNDS,
    ask_exec: bool = False,
) -> None:
    """REPL 主循环：内置命令族（§7.3）分发，其余输入交给 Agent。"""
    console = Console()
    if max_rounds < 1:
        console.print("--max-rounds 必须是不小于 1 的整数。", style="red", markup=False)
        sys.exit(2)
    resolved = _resolve_or_exit(console, model, provider)
    confirm_flag = {"on": ask_exec}
    try:
        agent = Agent(
            resolved=resolved,
            max_tool_rounds=max_rounds,
            confirm_exec=_exec_confirm_callback(console, confirm_flag),
        )
    except Exception as exc:  # noqa: BLE001 — 启动失败按运行错误退出，不裸抛堆栈（§7.5）
        console.print(f"[意外错误] {type(exc).__name__}: {exc}", style="red", markup=False)
        sys.exit(2)
    state = _ReplState(
        agent=agent, console=console, no_stream=no_stream, confirm_flag=confirm_flag
    )
    console.print(
        f"[bold]code-review-agent[/] 已就绪（模型：{_model_label(resolved)}）。"
        "提出审查需求即可；输入 /help 查看命令，exit 退出。"
    )
    while True:
        try:
            # 提示符着色便于在流式输出中定位输入位置（M2 验收反馈）
            console.print("你> ", style="bold cyan", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见。", markup=False, highlight=False)
            return
        if not user_input:
            continue  # 空输入/纯空白直接进入下一轮提示，不调 API（§6）
        if user_input == "exit":
            console.print("再见。")
            return
        if _handle_command(user_input, state):
            continue
        result = _run_turn(state.agent, console, user_input, no_stream=state.no_stream)
        if result is not None:
            state.last_answer = result


# ---- cra config（§7.4）：交互式配置与连通性自检 ----


class _ConfigSession:
    """cra config 交互会话：内存中修改、每个完整操作即写盘（任何时刻退出都一致）。

    首次成功写盘后打印文件权限与备份注意事项（§7.4：key 为明文属已知取舍 D9）。
    """

    def __init__(self, console: Console, data: dict[str, Any]) -> None:
        self.console = console
        self.data = data
        self._notice_shown = False

    def providers(self) -> list[dict[str, Any]]:
        return self.data.setdefault("providers", [])

    def save(self) -> None:
        try:
            config.save_raw(self.data)
        except config.ConfigError as exc:
            self.console.print(f"写入失败：{exc}", style="red", markup=False)
            return
        if not self._notice_shown:
            self.console.print(
                _CONFIG_FILE_NOTICE.format(path=config.config_path()), markup=False
            )
            self._notice_shown = True


def _mask_key(api_key: Any) -> str:
    """key 打码展示：只露前 4 后 4 位，配置界面不整段回显。"""
    if not isinstance(api_key, str) or not api_key:
        return "(未设置，回退环境变量 SE_CodeAgent)"
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"


def _print_providers(console: Console, data: dict[str, Any]) -> None:
    """当前配置概览（key 打码）。"""
    providers = data.get("providers") or []
    if not providers:
        console.print("当前无供应商配置。", markup=False)
        return
    console.print(f"默认模型：{data.get('default') or '（未设置）'}", markup=False)
    for entry in providers:
        console.print(
            f"  - {entry['name']}　base_url={entry['base_url']}　key={_mask_key(entry.get('api_key'))}"
            f"　模型：{', '.join(entry['models'])}",
            markup=False,
        )


def _read_key(prompt: str) -> str:
    """key 输入不回显（getpass），避免明文残留在终端回滚缓冲。"""
    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _config_add_provider(session: _ConfigSession) -> None:
    """添加供应商；同名已存在时询问是否更新（§7.4 同名覆盖确认）。"""
    console = session.console
    name = input("供应商名称（不含 '/'，如 deepseek）：").strip()
    if not name:
        console.print("已取消。", markup=False)
        return
    if "/" in name:
        console.print("供应商名不得含 '/'（§7.4）。", markup=False)
        return
    providers = session.providers()
    existing = next((item for item in providers if item.get("name") == name), None)
    if existing is not None:
        answer = input(f"供应商 '{name}' 已存在，是否更新其配置？（y/N）").strip().lower()
        if answer not in ("y", "yes"):
            console.print("已取消。", markup=False)
            return
        new_base = input(f"API 接入点（回车保留 {existing['base_url']}）：").strip()
        if new_base:
            existing["base_url"] = new_base
        new_key = _read_key("API key（输入不回显；回车保留现有 key）：")
        if new_key:
            existing["api_key"] = new_key
        console.print(f"供应商 '{name}' 已更新。", markup=False)
        session.save()
        return
    base_url = (
        input(f"API 接入点（回车默认 {config.DEFAULT_BASE_URL}）：").strip()
        or config.DEFAULT_BASE_URL
    )
    key = _read_key("API key（输入不回显；回车留空 = 回退环境变量 SE_CodeAgent）：")
    models_input = input("模型清单（逗号分隔，如 deepseek-flash,deepseek-chat）：").strip()
    models = [item.strip() for item in models_input.split(",") if item.strip()]
    if not models:
        console.print("至少需要一个模型，已取消。", markup=False)
        return
    providers.append({"name": name, "base_url": base_url, "api_key": key, "models": models})
    console.print(f"供应商 '{name}' 已添加。", markup=False)
    session.save()


def _config_remove_provider(session: _ConfigSession) -> None:
    """删除供应商；default 引用被删供应商时警告并清除默认（§7.4 边界规则）。"""
    console = session.console
    providers = session.providers()
    entry = _choose_entry(providers, console, "要删除的供应商")
    if entry is None:
        return
    default = session.data.get("default") or ""
    referenced = default.startswith(f"{entry['name']}/")
    if referenced:
        answer = input(f"默认模型 {default} 引用该供应商，删除后将清除默认设置，继续？（y/N）").strip().lower()
        if answer not in ("y", "yes"):
            console.print("已取消。", markup=False)
            return
    providers.remove(entry)
    if referenced:
        session.data.pop("default", None)
    console.print(f"供应商 '{entry['name']}' 已删除。", markup=False)
    session.save()


def _choose_entry(entries: list[Any], console: Console, label: str) -> Any | None:
    """通用序号选择：打印编号清单，回车或非法输入返回 None（取消）。"""
    if not entries:
        console.print("清单为空，已取消。", markup=False)
        return None
    for index, item in enumerate(entries, start=1):
        if isinstance(item, dict):
            console.print(f"  {index}. {item.get('name')}", markup=False)
        else:
            console.print(f"  {index}. {item}", markup=False)
    raw = input(f"{label}序号（回车取消）：").strip()
    if not raw:
        return None
    try:
        return entries[int(raw) - 1]
    except (ValueError, IndexError):
        console.print("无效序号，已取消。", markup=False)
        return None


def _config_manage_models(session: _ConfigSession) -> None:
    """供应商的模型清单管理：增删模型。"""
    entry = _choose_entry(session.providers(), session.console, "要管理的供应商")
    if entry is None:
        return
    while True:
        session.console.print(
            f"供应商 '{entry['name']}' 模型：{', '.join(entry['models'])}", markup=False
        )
        choice = input("1) 添加模型  2) 删除模型  0) 返回：").strip()
        if choice == "1":
            _config_add_model(session, entry)
        elif choice == "2":
            _config_remove_model(session, entry)
        elif choice == "0":
            return
        else:
            session.console.print("无效选择。", markup=False)


def _config_add_model(session: _ConfigSession, entry: dict[str, Any]) -> None:
    """添加模型；同名模型已存在时询问是否覆盖（§7.4）。

    模型清单是纯名称数组，"覆盖"即确认保留该名称（幂等），用于防误操作。
    """
    name = input("模型名称：").strip()
    if not name:
        session.console.print("已取消。", markup=False)
        return
    if name in entry["models"]:
        answer = input(f"模型 '{name}' 已存在于该供应商，是否覆盖？（y/N）").strip().lower()
        if answer not in ("y", "yes"):
            session.console.print("已取消。", markup=False)
            return
        session.console.print(f"模型 '{name}' 保持不变。", markup=False)
        return
    entry["models"].append(name)
    session.console.print(f"模型 '{name}' 已添加到 '{entry['name']}'。", markup=False)
    session.save()


def _config_remove_model(session: _ConfigSession, entry: dict[str, Any]) -> None:
    """删除模型；default 引用被删组合时警告并清除默认（§7.4 边界规则）。"""
    if len(entry["models"]) <= 1:
        session.console.print("供应商至少保留一个模型，无法删除。", markup=False)
        return
    name = _choose_entry(entry["models"], session.console, "要删除的模型")
    if name is None:
        return
    default = session.data.get("default") or ""
    if default == f"{entry['name']}/{name}":
        answer = input(f"默认模型 {default} 引用该模型，删除后将清除默认设置，继续？（y/N）").strip().lower()
        if answer not in ("y", "yes"):
            session.console.print("已取消。", markup=False)
            return
        session.data.pop("default", None)
    entry["models"].remove(name)
    session.console.print(f"模型 '{name}' 已从 '{entry['name']}' 删除。", markup=False)
    session.save()


def _config_set_default(session: _ConfigSession) -> None:
    """设置默认模型（default = '供应商/模型'）。"""
    entry = _choose_entry(session.providers(), session.console, "供应商")
    if entry is None:
        return
    model = _choose_entry(entry["models"], session.console, "模型")
    if model is None:
        return
    session.data["default"] = f"{entry['name']}/{model}"
    session.console.print(f"默认模型已设为 {entry['name']}/{model}。", markup=False)
    session.save()


def _config_update_key(session: _ConfigSession) -> None:
    """修改供应商 key（getpass 不回显）。"""
    entry = _choose_entry(session.providers(), session.console, "供应商")
    if entry is None:
        return
    key = _read_key("新 API key（输入不回显；回车取消）：")
    if not key:
        session.console.print("已取消。", markup=False)
        return
    entry["api_key"] = key
    session.console.print(f"供应商 '{entry['name']}' 的 key 已更新。", markup=False)
    session.save()


def _run_config_interactive() -> None:
    """cra config 交互式配置模式（§7.4）：增删供应商/模型、设默认、改 key。"""
    console = Console()
    try:
        data = config.load_raw()
    except config.ConfigError as exc:
        console.print(f"配置文件损坏：{exc}", style="red", markup=False)
        console.print("请手工修复该文件，或删除后重试（将重建空配置）。", markup=False)
        sys.exit(2)
    session = _ConfigSession(console, data)
    console.print(f"cra 配置（{config.config_path()}）", markup=False)
    _print_providers(console, data)
    while True:
        console.print(
            "\n菜单：1) 添加/更新供应商  2) 删除供应商  3) 管理供应商模型"
            "  4) 设置默认模型  5) 修改供应商 key  0) 退出",
            markup=False,
        )
        try:
            choice = input("选择：").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n退出配置。", markup=False)
            return
        if choice == "1":
            _config_add_provider(session)
        elif choice == "2":
            _config_remove_provider(session)
        elif choice == "3":
            _config_manage_models(session)
        elif choice == "4":
            _config_set_default(session)
        elif choice == "5":
            _config_update_key(session)
        elif choice == "0":
            console.print("退出配置。", markup=False)
            return
        else:
            console.print("无效选择。", markup=False)


def _run_config_test() -> None:
    """cra config test（§7.4）：对每个供应商发一次 max_tokens=1 请求报告连通性。

    退出码：全部 OK 为 0，存在失败为 1（检查结论），配置损坏为 2（运行错误）。
    """
    console = Console()
    try:
        data = config.load_raw()
    except config.ConfigError as exc:
        console.print(f"配置错误：{exc}", style="red", markup=False)
        sys.exit(2)
    providers = data.get("providers") or []
    if not providers:
        console.print("尚无供应商配置。请先运行 cra config 添加供应商。", markup=False)
        sys.exit(2)
    failures = 0
    for entry in providers:
        name = entry["name"]
        model_name = entry["models"][0]
        api_key = entry.get("api_key") or config.get_api_key()
        if not api_key:
            console.print(f"✗ {name}（{model_name}）：未配置 key（供应商条目与环境变量均缺）", style="red", markup=False)
            failures += 1
            continue
        status, detail = llm.probe(base_url=entry["base_url"], api_key=api_key, model_name=model_name)
        label = {"ok": "OK", "auth": "401（key 无效）", "timeout": "超时（网络不可达）"}.get(status, "失败")
        mark = "✓" if status == "ok" else "✗"
        console.print(f"{mark} {name}（{model_name}）：{label}", style=None if status == "ok" else "red", markup=False)
        if status != "ok":
            if detail:
                console.print(f"  {detail}", style="dim", markup=False)
            failures += 1
    sys.exit(1 if failures else 0)


# ---- cra review（§7.5）：非交互一次性审查 ----


class _ReviewFileError(Exception):
    """审查入参的文件级错误（路径不存在/目录无 .py/不可读），退出码 2。"""


def _collect_review_files(paths: list[str]) -> list[tuple[str, str]]:
    """展开审查入参（§7.1/§7.5）：多路径=批量、目录展开第一层 .py、'-' 读 stdin。

    返回 (显示名, 内容)：显示名保留用户输入形态（location 引用与报告标题用它），
    stdin 固定 "<stdin>"。内容显式 utf-8 + errors='replace'（§3.2 通用规则）。
    """
    collected: list[tuple[str, str]] = []
    for raw in paths:
        if raw == "-":
            collected.append(("<stdin>", sys.stdin.read()))
            continue
        path = Path(raw)
        if not path.exists():
            raise _ReviewFileError(f"路径不存在：{path}")
        if path.is_dir():
            py_files = sorted(path.glob("*.py"))
            if not py_files:
                raise _ReviewFileError(f"目录 {path} 第一层没有 .py 文件。")
            collected.extend((str(item), _read_review_text(item)) for item in py_files)
        else:
            collected.append((str(path), _read_review_text(path)))
    return collected


def _read_review_text(path: Path) -> str:
    """读取待审查文件文本；读取失败抛 _ReviewFileError（退出码 2）。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise _ReviewFileError(f"无法读取文件 {path}：{exc}") from exc


def _review_prompt(display_name: str, content: str, *, json_mode: bool) -> str:
    """构造单文件审查 prompt：文件内容直接附上（带行号），避免工具轮的不确定性，
    并覆盖系统提示词"引导用户保存为文件"的对话分支（§3.3）。

    行号格式与 read_file 一致（1-based、右对齐竖线分隔），保证 location 行号可靠；
    超过 _MAX_REVIEW_LINES 的文件截断并在 prompt 中明示。
    """
    lines = content.splitlines()[:_MAX_REVIEW_LINES]
    numbered = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(lines, start=1))
    total = len(content.splitlines())
    truncated = f"注意：文件共 {total} 行，以下仅含前 {len(lines)} 行，结论只基于这部分内容。\n" if total > len(lines) else ""
    head = (
        f"请审查以下文件。文件名：{display_name}。文件内容已直接附在下方（行号在每行前），"
        "无需再调用工具读取；不要要求用户保存文件，直接给出审查结论。"
    )
    if json_mode:
        spec = (
            "请仅输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码围栏，"
            f'结构如下（file 固定为 "{display_name}"，location 格式为 "{display_name}:行号"，'
            'severity 只能取 严重/一般/建议 之一，uncertain 表示该条结论是否不确定）：\n'
            '{"files": [{"file": "...", "findings": [{"severity": "严重|一般|建议", '
            '"location": "...", "issue": "...", "suggestion": "...", "uncertain": false}], '
            '"summary": "该文件整体评价"}], "summary": "整体评价"}'
        )
    else:
        spec = "请按系统提示词约定的分级格式输出审查结论。"
    return f"{head}\n{truncated}\n{display_name}:\n````\n{numbered}\n````\n\n{spec}"


def _report_schema_errors(data: Any) -> list[str]:
    """§7.5 schema 校验；返回错误列表（空=通过）。uncertain 缺省按 False 补全。"""
    if not isinstance(data, dict):
        return ["顶层必须是 JSON 对象"]
    files = data.get("files")
    if not isinstance(files, list) or not files:
        return ["files 必须是非空数组"]
    errors: list[str] = []
    for index, entry in enumerate(files):
        where = f"files[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} 必须是对象")
            continue
        if not isinstance(entry.get("file"), str) or not entry.get("file"):
            errors.append(f"{where}.file 必须是非空字符串")
        if not isinstance(entry.get("summary"), str):
            errors.append(f"{where}.summary 必须是字符串")
        findings = entry.get("findings")
        if not isinstance(findings, list):
            errors.append(f"{where}.findings 必须是数组")
            continue
        for f_index, finding in enumerate(findings):
            f_where = f"{where}.findings[{f_index}]"
            if not isinstance(finding, dict):
                errors.append(f"{f_where} 必须是对象")
                continue
            if finding.get("severity") not in _VALID_SEVERITIES:
                errors.append(f"{f_where}.severity 必须是 严重/一般/建议 之一")
            if not isinstance(finding.get("location"), str) or not finding.get("location"):
                errors.append(f"{f_where}.location 必须是非空字符串")
            if not isinstance(finding.get("issue"), str) or not finding.get("issue"):
                errors.append(f"{f_where}.issue 必须是非空字符串")
            if not isinstance(finding.get("suggestion"), str):
                errors.append(f"{f_where}.suggestion 必须是字符串")
            uncertain = finding.get("uncertain", False)
            if not isinstance(uncertain, bool):
                errors.append(f"{f_where}.uncertain 必须是布尔值")
            else:
                finding["uncertain"] = uncertain  # 补全缺省，输出满足完整 schema
    return errors


def _parse_json_report(answer: str) -> dict[str, Any]:
    """从模型回答提取并校验 §7.5 schema 的 JSON；失败抛 ValueError。"""
    text = answer.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else text[3:]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("回答中未找到 JSON 对象。")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    errors = _report_schema_errors(data)
    if errors:
        raise ValueError("schema 校验失败：" + "；".join(errors))
    if not isinstance(data.get("summary"), str):
        data["summary"] = ""  # 模型省略顶层 summary 时补全,schema 完整性供下游稳定消费
    return data


def _merge_json_reports(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """多文件的 schema 片段合并为 §7.5 统一形态；单文件直接透传其 summary。"""
    if not entries:
        return {"files": [], "summary": "没有可用的审查结果。"}
    if len(entries) == 1:
        return {"files": entries, "summary": entries[0]["summary"]}
    summary = "；".join(f"{entry['file']}：{entry['summary']}" for entry in entries)
    return {"files": entries, "summary": summary}


def _json_report_markdown(merged: dict[str, Any]) -> str:
    """--json 模式的 Markdown 报告（-o 与 --json 可同时使用，§7.5）。"""
    lines = ["# cra 审查报告", "", f"> 生成时间：{_local_now():%Y-%m-%d %H:%M:%S}（--json 模式）", ""]
    for entry in merged["files"]:
        lines += [f"## {entry['file']}", "", f"**整体评价**：{entry['summary']}", ""]
        for finding in entry["findings"]:
            uncertain = "（不确定）" if finding.get("uncertain") else ""
            lines += [
                f"- **【{finding['severity']}】** `{finding['location']}` — {finding['issue']}{uncertain}",
                f"  修改建议：{finding['suggestion']}",
            ]
        lines.append("")
    lines += ["## 总体评价", "", merged["summary"], ""]
    return "\n".join(lines)


def _text_report_markdown(files: list[tuple[str, str]], reports: list[tuple[str, str]]) -> str:
    """非 --json 模式的 Markdown 报告：结论原文嵌入（本身即 Markdown）。"""
    lines = [
        "# cra 审查报告",
        "",
        f"> 生成时间：{_local_now():%Y-%m-%d %H:%M:%S}",
        f"> 审查对象：{', '.join(name for name, _ in files)}",
        "",
    ]
    for display_name, answer in reports:
        lines += [f"## {display_name}", "", answer, ""]
    return "\n".join(lines)


def _run_review(args: argparse.Namespace) -> None:
    """cra review 主流程（§7.5）：批量审查、-o 报告、--json 结构化输出与退出码。

    stdout 只承载结论/--json（过程与错误走 stderr）；退出码 0 完成、1 为 --json
    模式下存在【严重】问题、2 为运行错误（含单文件失败与顶层兜底）。
    """
    console = Console(file=sys.stderr)
    if args.ask_exec:
        console.print("cra review 不支持 --ask-exec（非交互模式无法逐次确认执行）。", style="red", markup=False)
        sys.exit(2)
    if args.max_rounds < 1:
        console.print("--max-rounds 必须是不小于 1 的整数。", style="red", markup=False)
        sys.exit(2)
    try:
        files = _collect_review_files(args.paths)
    except _ReviewFileError as exc:
        console.print(str(exc), style="red", markup=False)
        sys.exit(2)
    try:
        resolved = config.resolve(model=args.model, provider=args.provider)
    except config.ConfigError as exc:
        console.print(f"配置错误：{exc}", style="red", markup=False)
        sys.exit(2)
    if resolved is None:
        _print_config_guide(console)
        sys.exit(2)
    try:
        agent = Agent(resolved=resolved, max_tool_rounds=args.max_rounds)
    except Exception as exc:  # noqa: BLE001 — 启动失败按运行错误退出，不裸抛堆栈（§7.5）
        console.print(f"[意外错误] {type(exc).__name__}: {exc}", style="red", markup=False)
        sys.exit(2)

    json_entries: list[dict[str, Any]] = []
    text_reports: list[tuple[str, str]] = []
    failed = False
    severe = False
    for display_name, content in files:
        console.print(f"正在审查 {display_name} …", markup=False)
        prompt = _review_prompt(display_name, content, json_mode=args.json)
        try:
            answer = agent.run(prompt)
        except Exception as exc:  # noqa: BLE001 — 单文件失败继续批量；最终退出码 2（§7.5）
            console.print(
                f"[审查失败] {display_name}:{type(exc).__name__}: {exc}", style="red", markup=False
            )
            failed = True
            continue
        if args.json:
            try:
                report = _parse_json_report(answer)
            except ValueError as exc:
                console.print(f"[结构化输出无效] {display_name}：{exc}", style="red", markup=False)
                failed = True
                continue
            if any(finding.get("severity") == "严重" for finding in report["files"][0]["findings"]):
                severe = True
            json_entries.append(report["files"][0])
        else:
            text_reports.append((display_name, answer))

    if args.json:
        merged = _merge_json_reports(json_entries)
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        markdown = _json_report_markdown(merged)
    else:
        if len(files) > 1:
            for index, (display_name, answer) in enumerate(text_reports):
                if index:
                    print()
                print(f"===== {display_name} =====")
                print(answer)
        else:
            for _, answer in text_reports:
                print(answer)
        markdown = _text_report_markdown(files, text_reports)
    if args.output:
        try:
            Path(args.output).write_text(markdown, encoding="utf-8")
            console.print(f"报告已写入：{Path(args.output).resolve()}", markup=False)
        except OSError as exc:
            console.print(f"报告写入失败：{exc}", style="red", markup=False)
            failed = True
    if failed:
        sys.exit(2)
    sys.exit(1 if severe else 0)


def _version_text() -> str:
    """cra --version 的版本文本（包元数据缺失时给出占位而非崩溃）。"""
    try:
        return f"cra {_package_version('code-review-agent')}"
    except PackageNotFoundError:
        return "cra 0.1.0（包元数据缺失）"


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    """chat / review 通用参数（§7.2）。"""
    parser.add_argument("--model", metavar="NAME", help="临时覆盖模型名（须在已解析供应商的清单内）")
    parser.add_argument("--provider", metavar="NAME", help="临时覆盖供应商（单独使用时取其第一个模型）")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式渲染，整体输出")
    parser.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS, metavar="N", help=f"最大工具轮次（默认 {MAX_TOOL_ROUNDS}）")


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（§7.1/§7.2）。"""
    parser = argparse.ArgumentParser(
        prog="cra", description="基于 LLM 的命令行代码审查助手"
    )
    parser.add_argument("--version", action="version", version=_version_text())
    subparsers = parser.add_subparsers(dest="command", required=True)
    # §8.2 核对项：限制声明在 CLI 层面可见（工具 description 之外的第二道明示）
    chat = subparsers.add_parser(
        "chat",
        help="进入对话式代码审查 REPL（run_python 非沙箱、限时执行）",
        description=(
            "进入对话式代码审查 REPL。注意：Agent 的 run_python 工具在本机直接执行"
            "代码，非沙箱、限时执行（默认 10 秒，超时强杀进程树），详见 README「限制」。"
        ),
    )
    _add_shared_options(chat)
    chat.add_argument(
        "--ask-exec",
        action="store_true",
        help="run_python 执行前逐次人工确认（默认自动执行；REPL 内 /confirm on|off 同效）",
    )

    review = subparsers.add_parser(
        "review",
        help="非交互一次性审查（多路径=批量；'-' 读 stdin；目录展开第一层 .py）",
        description=(
            "非交互一次性审查并输出结论。注意：Agent 的 run_python 工具在本机直接"
            "执行代码，非沙箱、限时执行（默认 10 秒，超时强杀进程树），详见 README「限制」。"
        ),
    )
    review.add_argument("paths", nargs="+", metavar="path|-", help="待审查路径（文件/目录）；'-' 表示从 stdin 读取")
    _add_shared_options(review)
    review.add_argument("-o", "--output", metavar="FILE", help="将审查结论写入 Markdown 报告")
    review.add_argument("--json", action="store_true", help="按固定 schema 输出结构化 JSON（stdout；退出码按 severity 判定）")
    review.add_argument("--ask-exec", action="store_true", help="不支持：cra review 非交互模式无法逐次确认执行，指定时报错退出 2")

    config_parser = subparsers.add_parser(
        "config",
        help="交互式配置模式（供应商/模型/key/默认项，写入 ~/.cra/models.json）",
        description=(
            "交互式配置多供应商模型。注意：配置文件以明文保存 API key，"
            "写入时会提示文件权限与备份注意事项。"
        ),
    )
    config_parser.add_argument(
        "config_action",
        nargs="?",
        choices=["test"],
        help="test：连通性自检（对每个供应商发一次 max_tokens=1 请求）；省略进入交互式配置",
    )
    return parser


def main() -> None:
    """cra 入口。"""
    args = _build_parser().parse_args()
    if args.command == "chat":
        _run_chat(
            model=args.model,
            provider=args.provider,
            no_stream=args.no_stream,
            max_rounds=args.max_rounds,
            ask_exec=args.ask_exec,
        )
    elif args.command == "review":
        _run_review(args)
    elif args.command == "config":
        if args.config_action == "test":
            _run_config_test()
        else:
            _run_config_interactive()
