"""M0/M1 function calling 冒烟测试（全仓库唯一允许真实联网的脚本，AGENTS.md 工程规范）。

用最小哑工具 echo 验证 function calling 通路，与业务工具 schema 无关（Design.md §3）：
1. 非流式：发送一次带 tools 参数的请求，确认模型（CRA_MODEL，默认 deepseek-flash）返回 tool_calls；
   把 echo 执行结果作为 tool 消息回传，确认模型能给出最终回答。
2. 流式（M1 追加）：验证流式响应的 tool_calls 分片可组装、末块携带 usage
   （stream_options include_usage）、回传工具结果后能流式给出最终回答。

结论须记入 docs/PROGRESS.md。运行：uv run python scripts/smoke_tools.py
"""

import json
import sys
from typing import Any

from cra import config, llm

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "原样返回输入文本。用于验证 function calling 通路。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文本"}
            },
            "required": ["text"],
        },
    },
}


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        # Windows 控制台默认 cp936，中文输出会 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not config.get_api_key():
        print("未设置环境变量 SE_CodeAgent，无法执行冒烟测试。")
        return 1

    client = llm.create_client()
    model = config.get_model()
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": '请调用 echo 工具，参数 text 传 "cra-smoke-ok"，不要自己回答。',
        }
    ]

    print(f"模型：{model}　接入点：{config.get_base_url()}")
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[ECHO_TOOL], temperature=0.2
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print(f"[FAIL] 模型未返回 tool_calls，纯文本回复：{message.content!r}")
        return 1

    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    echo_result = arguments.get("text", "")
    print(
        f"[PASS] 收到 tool_calls：name={tool_call.function.name} "
        f"arguments={tool_call.function.arguments}"
    )

    messages.append(message)  # openai SDK 接受 pydantic 消息对象直接回传
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": echo_result}
    )
    final = client.chat.completions.create(
        model=model, messages=messages, tools=[ECHO_TOOL], temperature=0.2
    )
    final_text = final.choices[0].message.content or ""
    print(f"工具回传后最终回答：{final_text!r}")
    if "cra-smoke-ok" not in final_text:
        print("[WARN] 最终回答未包含回显内容，tool 结果回传通路需人工复核")
        return 1
    print("[PASS] tool 结果回传后模型正常给出最终回答，function calling 通路完整")
    return 0


def smoke_streaming() -> int:
    """流式通路（M1）：tool_calls 分片组装、include_usage usage、流式最终回答。"""
    client = llm.create_client()
    model = config.get_model()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": '请调用 echo 工具，参数 text 传 "cra-stream-ok"，不要自己回答。',
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ECHO_TOOL],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )
    fragments: dict[int, dict[str, Any]] = {}
    total_tokens: int | None = None
    for chunk in stream:
        if chunk.usage is not None:
            total_tokens = chunk.usage.total_tokens
        if not chunk.choices:
            continue  # 末块只携带 usage
        delta = chunk.choices[0].delta
        for piece in (delta.tool_calls if delta else None) or []:
            slot = fragments.setdefault(piece.index, {"id": "", "name": "", "args": []})
            if piece.id:
                slot["id"] = piece.id
            if piece.function:
                if piece.function.name:
                    slot["name"] = piece.function.name
                if piece.function.arguments:
                    slot["args"].append(piece.function.arguments)
    stream.close()

    if not fragments:
        print("[FAIL] 流式响应未返回 tool_calls 分片，统一流式方案不可行")
        return 1
    slot = fragments[0]
    arguments = "".join(slot["args"])
    print(f"[PASS] 流式收到 tool_calls：name={slot['name']} arguments={arguments}")
    if total_tokens is None:
        print("[WARN] 末块未携带 usage（include_usage 未生效）；截断将安全降级为不触发")
    else:
        print(f"[PASS] 末块 usage.total_tokens={total_tokens}（include_usage 生效）")

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": arguments},
                }
            ],
        }
    )
    messages.append(
        {"role": "tool", "tool_call_id": slot["id"], "content": json.loads(arguments).get("text", "")}
    )
    final_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ECHO_TOOL],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    for chunk in final_stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    final_stream.close()
    final_text = "".join(parts)
    print(f"流式最终回答：{final_text!r}")
    if "cra-stream-ok" not in final_text:
        print("[WARN] 流式最终回答未包含回显内容，需人工复核")
        return 1
    print("[PASS] 流式通路完整：分片组装 + usage + 最终回答")
    return 0


if __name__ == "__main__":
    exit_code = main()
    print("\n--- 流式通路冒烟（M1 追加）---")
    exit_code = smoke_streaming() or exit_code
    sys.exit(exit_code)
