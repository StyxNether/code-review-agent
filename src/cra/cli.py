"""cra 命令行入口：argparse + 最小问答 REPL。

M0 范围（Design.md §3.4 M0 豁免）：无工具、无 Agent 循环、无跨轮记忆，
cli 直连 llm；纯文本输出（rich 流式彩色与内置命令 /clear 属 M1）。
"""

import argparse
import sys

from cra import config, llm


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


def _run_chat() -> None:
    """逐条问答 REPL：每轮独立请求，不保留上下文。"""
    if not config.get_api_key():
        _print_setup_guide()
        sys.exit(1)

    client = llm.create_client()
    model = config.get_model()
    print(f"code-review-agent 已就绪（模型：{model}）。输入内容开始对话，输入 exit 退出。")
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user_input:
            continue  # 空输入/纯空白直接进入下一轮提示，不调 API（§6）
        if user_input == "exit":
            print("再见。")
            return
        try:
            reply = llm.chat(client, model, [{"role": "user", "content": user_input}])
        except KeyboardInterrupt:
            print("\n（本轮已中断）")
        except llm.LLMError as exc:  # M0 直接给出可读报错；重试与错误细分在 M2 实现
            print(f"[请求失败] {exc}")
            print("请检查网络与 API key。")
        else:
            print(f"cra> {reply}\n")


def main() -> None:
    """cra 入口。"""
    args = _build_parser().parse_args()
    if args.command == "chat":
        _run_chat()
