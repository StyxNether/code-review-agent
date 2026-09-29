"""M0 function calling 冒烟测试（全仓库唯一允许真实联网的脚本，AGENTS.md 工程规范）。

用最小哑工具 echo 验证 function calling 通路，与业务工具 schema 无关（Design.md §3）：
1. 发送一次带 tools 参数的请求，确认模型（CRA_MODEL，默认 deepseek-flash）返回 tool_calls；
2. 把 echo 执行结果作为 tool 消息回传，确认模型能给出最终回答。

结论须记入 docs/PROGRESS.md 的 M0 验收记录。运行：uv run python scripts/smoke_tools.py
"""

import json
import sys

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


if __name__ == "__main__":
    sys.exit(main())
