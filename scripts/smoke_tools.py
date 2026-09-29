"""function calling 冒烟测试（全仓库唯一允许真实联网的脚本，AGENTS.md 工程规范）。

M4 起走 LangChain 新栈（llm.create_model 的 ChatOpenAI，OpenAI 兼容），用最小哑
工具 echo 验证模型通路，与业务工具 schema 无关（Design.md §3）：
1. 非流式（llm.invoke_once 保留通路）：bind_tools 后请求，确认模型返回 tool_calls；
   把 echo 执行结果作为 ToolMessage 回传，确认模型能给出最终回答。
2. 流式：验证流式响应的 tool_call_chunks 可组装、末块携带 usage
   （stream_usage=True）、回传工具结果后能流式给出最终回答。

结论须记入 docs/PROGRESS.md。运行：uv run python scripts/smoke_tools.py
"""

import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cra import config, llm


@tool
def echo(text: str) -> str:
    """原样返回输入文本。用于验证 function calling 通路。"""
    return text


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        # Windows 控制台默认 cp936，中文输出会 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not config.get_api_key():
        print("未设置环境变量 SE_CodeAgent，无法执行冒烟测试。")
        return 1

    model = llm.create_model().bind_tools([echo])
    print(f"模型：{config.get_model()}　接入点：{config.get_base_url()}")
    messages = [
        HumanMessage(content='请调用 echo 工具，参数 text 传 "cra-smoke-ok"，不要自己回答。')
    ]

    response = llm.invoke_once(model, messages)
    if not response.tool_calls:
        print(f"[FAIL] 模型未返回 tool_calls，纯文本回复：{response.text!r}")
        return 1

    call = response.tool_calls[0]
    echo_result = call["args"].get("text", "")
    print(f"[PASS] 收到 tool_calls：name={call['name']} arguments={json.dumps(call['args'])}")

    messages.append(response)
    messages.append(ToolMessage(content=echo_result, tool_call_id=call["id"]))
    final = llm.invoke_once(model, messages)
    final_text = final.text
    print(f"工具回传后最终回答：{final_text!r}")
    if "cra-smoke-ok" not in final_text:
        print("[WARN] 最终回答未包含回显内容，tool 结果回传通路需人工复核")
        return 1
    print("[PASS] tool 结果回传后模型正常给出最终回答，function calling 通路完整")
    return 0


def smoke_streaming() -> int:
    """流式通路：tool_call_chunks 组装、stream_usage usage、流式最终回答。"""
    model = llm.create_model().bind_tools([echo])
    messages = [
        HumanMessage(content='请调用 echo 工具，参数 text 传 "cra-stream-ok"，不要自己回答。')
    ]

    fragments: dict[int, dict[str, object]] = {}
    total_tokens: int | None = None
    for chunk in model.stream(messages):
        if chunk.usage_metadata:
            total_tokens = chunk.usage_metadata.get("total_tokens")
        for piece in chunk.tool_call_chunks or []:
            index = piece.get("index") or 0
            slot = fragments.setdefault(index, {"id": "", "name": "", "args": []})
            if piece.get("id"):
                slot["id"] = piece["id"]
            if piece.get("name"):
                slot["name"] = piece["name"]
            if piece.get("args"):
                slot["args"].append(piece["args"])

    if not fragments:
        print("[FAIL] 流式响应未返回 tool_call_chunks，流式工具通路不可用")
        return 1
    slot = fragments[0]
    arguments = "".join(slot["args"])  # type: ignore[union-attr]
    print(f"[PASS] 流式收到 tool_calls：name={slot['name']} arguments={arguments}")
    if total_tokens is None:
        print("[WARN] 流式末块未携带 usage（stream_usage 未生效）；截断将安全降级为不触发")
    else:
        print(f"[PASS] 末块 usage.total_tokens={total_tokens}（stream_usage 生效）")

    messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": str(slot["name"]),
                    "args": json.loads(str(arguments)),
                    "id": str(slot["id"]),
                    "type": "tool_call",
                }
            ],
        )
    )
    messages.append(
        ToolMessage(content=json.loads(str(arguments)).get("text", ""), tool_call_id=str(slot["id"]))
    )
    parts: list[str] = []
    for chunk in model.stream(messages):
        if chunk.text:
            parts.append(chunk.text)
    final_text = "".join(parts)
    print(f"流式最终回答：{final_text!r}")
    if "cra-stream-ok" not in final_text:
        print("[WARN] 流式最终回答未包含回显内容，需人工复核")
        return 1
    print("[PASS] 流式通路完整：分片组装 + usage + 最终回答")
    return 0


if __name__ == "__main__":
    exit_code = main()
    print("\n--- 流式通路冒烟 ---")
    exit_code = smoke_streaming() or exit_code
    sys.exit(exit_code)
