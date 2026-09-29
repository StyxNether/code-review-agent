"""cra 命令行入口：argparse + REPL，rich 流式彩色渲染。

cli 只管交互与渲染（§3.4）：LLM 调用与工具执行一律经 Agent（M1 已收敛 M0 的
直连豁免）；从 llm 导入的只有 LLMError 类型、从 tools 导入的只有 ERROR_PREFIX
常量，不直接发起 LLM 请求、不直接执行工具。
"""

import argparse
import re
import sys

from rich.console import Console
from rich.markdown import Markdown

from cra import config
from cra.agent import Agent
from cra.llm import LLMError
from cra.tools import ERROR_PREFIX

_TOOL_ARGS_PREVIEW_LIMIT = 80
_GRADE_MARKERS = ("【严重】", "【一般】", "【建议】")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


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


def _emphasize_grade_markers(text: str) -> str:
    """分级标记加粗（终端无法改变字号，用字重区分层级——M1 验收反馈）。

    三个标记都替换（连续问题行同属一个 Markdown 段落，不能命中第一个就返回）；
    围栏代码块内的行跳过——块内插入 ** 会以字面文本渲染出来。
    """
    lines: list[str] = []
    state: tuple[str, int] | None = None
    for line in text.split("\n"):
        state = _fence_toggle(line, state)
        if state is None:
            for marker in _GRADE_MARKERS:
                line = line.replace(marker, f"**{marker}**")
        lines.append(line)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器；保留子命令扩展位（Should 的 batch / report 未来挂载）。"""
    parser = argparse.ArgumentParser(
        prog="cra", description="基于 LLM 的命令行代码审查助手"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="进入对话式代码审查 REPL")
    return parser


def _print_setup_guide() -> None:
    """SE_CodeAgent 缺失时的设置指引（§5：打印指引后退出）。"""
    print("未设置环境变量 SE_CodeAgent（DeepSeek API key）。")
    print('设置方法（Windows）：setx SE_CodeAgent "sk-你的key"，然后新开终端重试。')


def _short(text: str, limit: int = _TOOL_ARGS_PREVIEW_LIMIT) -> str:
    """单行预览：压掉换行并截断，避免工具参数刷屏。"""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


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
        self._console.print(Markdown(_emphasize_grade_markers(text)))
        self._has_output = True
        self._pending_break = False


def _run_turn(agent: Agent, console: Console, user_input: str) -> None:
    """执行一轮对话：工具调用打点、最终回答流式渲染、中断与请求错误兜底。"""
    streamer = _StreamingMarkdown(console)

    def on_tool(name: str, args: str) -> None:
        console.print(f"⚙ {name}({_short(args)})", style="dim", markup=False, highlight=False)

    def on_tool_result(result: str) -> None:
        if result.startswith(ERROR_PREFIX):
            console.print("  ↳ 工具返回错误（已回传模型）", style="dim", markup=False)
        else:
            console.print(f"  ↳ 完成（{len(result)} 字符）", style="dim", markup=False)
        streamer.mark_break()

    try:
        agent.run(
            user_input,
            on_tool=on_tool,
            on_tool_result=on_tool_result,
            on_delta=streamer.feed,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ 本轮已中断作废。[/]")
    except LLMError as exc:  # 重试与错误细分在 M2 实现
        console.print("[请求失败] " + str(exc), style="red", markup=False)
        console.print("请检查网络与 API key。", style="dim", markup=False)
    except Exception as exc:  # noqa: BLE001 — REPL 边界兜底：意外异常也不终止会话（§3.4）
        console.print(f"[意外错误] {type(exc).__name__}: {exc}", style="red", markup=False)
        console.print("本轮已终止，上下文保留；可重试或输入 exit 退出。", style="dim", markup=False)
    finally:
        streamer.finish()
        console.print()


def _run_chat() -> None:
    """REPL 主循环：exit 退出；/clear 重置为仅含系统提示词的初始状态（§6）。"""
    if not config.get_api_key():
        _print_setup_guide()
        sys.exit(1)

    console = Console()
    agent = Agent()
    console.print(
        f"[bold]code-review-agent[/] 已就绪（模型：{agent.model}）。"
        "提出审查需求即可；exit 退出，/clear 清空上下文。"
    )
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user_input:
            continue  # 空输入/纯空白直接进入下一轮提示，不调 API（§6）
        if user_input == "exit":
            console.print("再见。")
            return
        if user_input == "/clear":
            agent.reset()
            console.print("[dim]上下文已清空（系统提示词保留）。[/]")
            continue
        _run_turn(agent, console, user_input)


def main() -> None:
    """cra 入口。"""
    args = _build_parser().parse_args()
    if args.command == "chat":
        _run_chat()
