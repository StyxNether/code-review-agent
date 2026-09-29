"""Agent 循环：消息组装、工具分发、轮次上限、轮次原子截断、中断回滚（全 mock，无网络）。"""

import json
from collections.abc import Callable
from typing import Any

import pytest

from cra import llm
from cra.agent import MAX_TOOL_ROUNDS, Agent, truncate_rounds
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ToolRegistry


def make_registry() -> ToolRegistry:
    """测试用注册表：一个确定性的 echo 工具。"""
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="回显文本",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: f"ECHO:{text}",
    )
    return registry


def tool_response(
    call_id: str, name: str, arguments: dict[str, Any], total: int | None = None
) -> llm.LLMResponse:
    message = llm.AssistantMessage(
        content=None,
        tool_calls=[llm.ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
    )
    return llm.LLMResponse(message=message, total_tokens=total)


def text_response(text: str, total: int | None = None) -> llm.LLMResponse:
    message = llm.AssistantMessage(content=text, tool_calls=[])
    return llm.LLMResponse(message=message, total_tokens=total)


def make_agent(
    monkeypatch: pytest.MonkeyPatch,
    script: list[llm.LLMResponse],
    *,
    max_context_tokens: int | None = None,
) -> tuple[Agent, list[dict[str, Any]]]:
    """注入 fake llm.complete（按脚本回放），返回 (agent, 每次请求的参数记录)。"""
    calls: list[dict[str, Any]] = []

    def fake_complete(
        client: object,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> llm.LLMResponse:
        calls.append({"tools": tools, "messages": [dict(m) for m in messages]})
        response = script.pop(0)
        # 模拟流式行为：纯文本回答会把增量回调出去（工具轮的增量由真实 SDK 决定，不模拟）
        if on_delta is not None and response.message.content and not response.message.tool_calls:
            on_delta(response.message.content)
        return response

    monkeypatch.setattr(llm, "complete", fake_complete)
    kwargs: dict[str, Any] = {
        "client": object(),
        "model": "test-model",
        "registry": make_registry(),
    }
    if max_context_tokens is not None:
        kwargs["max_context_tokens"] = max_context_tokens
    return Agent(**kwargs), calls


def make_round(user_text: str, *, with_tools: bool = True, answer: str = "done") -> list[dict[str, Any]]:
    """构造一个结构完整的历史轮次：user → assistant(tool_calls) → tool → assistant。"""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    if with_tools:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{user_text[:8]}",
                        "type": "function",
                        "function": {"name": "echo", "arguments": json.dumps({"text": "x"})},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{user_text[:8]}", "content": "ECHO:x"})
    messages.append({"role": "assistant", "content": answer})
    return messages


def assert_rounds_atomic(messages: list[dict[str, Any]]) -> None:
    """消息序列合法性：每条 tool 消息都对应仍在等待结果的 tool_call，且最终无悬挂。"""
    pending: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                pending.add(call["id"])
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending, f"tool 消息悬挂：{message}"
            pending.discard(message["tool_call_id"])
    assert not pending, f"assistant 的 tool_calls 没有对应 tool 结果：{pending}"


class TestRunBasics:
    def test_initial_messages_are_system_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = make_agent(monkeypatch, [])
        assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]

    def test_reset_restores_system_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = make_agent(monkeypatch, [text_response("答")])
        agent.run("x")
        agent.reset()
        assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]

    def test_plain_answer_assembles_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, calls = make_agent(monkeypatch, [text_response("结论")])

        result = agent.run("审查 agent.py")

        assert result == "结论"
        assert calls[0]["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "审查 agent.py"},
        ]
        assert agent.messages[-1] == {"role": "assistant", "content": "结论"}

    def test_tool_round_dispatch_and_pairing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = make_agent(
            monkeypatch, [tool_response("c1", "echo", {"text": "hi"}), text_response("结论")]
        )
        seen: list[tuple[str, str]] = []

        result = agent.run("帮我", on_tool=lambda name, args: seen.append((name, args)))

        assert result == "结论"
        assert seen == [("echo", json.dumps({"text": "hi"}))]
        assert [m["role"] for m in agent.messages] == [
            "system", "user", "assistant", "tool", "assistant",
        ]
        assistant_msg = agent.messages[2]
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "echo"
        tool_msg = agent.messages[3]
        assert tool_msg["tool_call_id"] == "c1"
        assert tool_msg["content"] == "ECHO:hi"
        assert_rounds_atomic(agent.messages)

    def test_unknown_tool_error_returned_to_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = make_agent(
            monkeypatch, [tool_response("c1", "no_such_tool", {}), text_response("好")]
        )

        agent.run("x")

        tool_msg = agent.messages[3]
        assert tool_msg["content"].startswith("错误")
        assert "未注册" in tool_msg["content"]

    def test_delta_callback_fires_for_final_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deltas: list[str] = []
        agent, _ = make_agent(monkeypatch, [text_response("流式回答")])

        agent.run("x", on_delta=deltas.append)

        assert deltas == ["流式回答"]

    def test_final_call_after_tools_carries_full_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, calls = make_agent(
            monkeypatch, [tool_response("c1", "echo", {"text": "hi"}), text_response("结论")]
        )

        agent.run("x")

        # 最终回答请求发生在追加 assistant 回答之前，因此携带的是除末条外的完整历史
        assert calls[1]["messages"] == agent.messages[:-1]


class TestMaxToolRounds:
    def test_forces_conclusion_after_max_rounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            tool_response(f"c{i}", "echo", {"text": str(i)})
            for i in range(MAX_TOOL_ROUNDS + 1)
        ]
        script.append(text_response("最终结论"))
        agent, calls = make_agent(monkeypatch, script)

        result = agent.run("x")

        assert result == "最终结论"
        assert len(calls) == MAX_TOOL_ROUNDS + 2  # 10 轮工具 + 1 次超限响应 + 1 次强制收尾
        assert calls[-1]["tools"] is None  # 强制收尾不带工具，保证终止
        assert all(call["tools"] is not None for call in calls[:-1])
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_messages) == MAX_TOOL_ROUNDS  # 超限那次 tool_calls 未落库
        assert any(m["role"] == "user" and "上限" in m["content"] for m in agent.messages)
        assert_rounds_atomic(agent.messages)


class TestTruncateRounds:
    def test_under_threshold_untouched(self) -> None:
        messages = [{"role": "system", "content": "S"}, *make_round("u1", with_tools=False)]
        result = truncate_rounds(messages, usage_total_tokens=100, max_tokens=200)
        assert result == messages
        assert result is not messages  # 未触发截断也返回副本（纯函数语义）

    def test_none_usage_untouched(self) -> None:
        messages = [{"role": "system", "content": "S"}, *make_round("u1", with_tools=False)]
        assert truncate_rounds(messages, usage_total_tokens=None, max_tokens=10) == messages

    def test_over_threshold_drops_oldest_round_atomically(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "system", "content": "S"}]
        messages += make_round("u1" * 200)  # 长轮次（含 tool_calls/tool 配对）
        messages += make_round("u2", with_tools=False)
        original = list(messages)

        result = truncate_rounds(messages, usage_total_tokens=50_000, max_tokens=100)

        assert result[0] == {"role": "system", "content": "S"}  # 系统提示词保留
        assert result[1] == {"role": "user", "content": "u2"}  # 第一轮整体消失
        assert any(m["content"] == "done" for m in result)  # 最后一轮完整保留
        assert_rounds_atomic(result)  # tool_calls/tool 配对不被拆散
        assert messages == original  # 纯函数：不改入参

    def test_never_drops_last_round(self) -> None:
        messages = [{"role": "system", "content": "S"}, *make_round("u1" * 400)]

        result = truncate_rounds(messages, usage_total_tokens=999_999, max_tokens=10)

        assert result == messages  # 单轮不可拆：宁可超限也不产生非法结构
        assert result is not messages  # 且始终返回副本（纯函数语义）

    def test_directive_user_message_is_a_valid_boundary(self) -> None:
        # max-rounds 指令也是 user 消息：以它为界丢弃旧前缀后结构仍合法
        messages: list[dict[str, Any]] = [{"role": "system", "content": "S"}]
        messages += make_round("u1" * 300)
        messages.append({"role": "user", "content": "工具调用已达上限（10 轮）……"})
        messages.append({"role": "assistant", "content": "结论"})

        result = truncate_rounds(messages, usage_total_tokens=10_000, max_tokens=50)

        assert result[1]["content"].startswith("工具调用已达上限")
        assert_rounds_atomic(result)

    def test_estimate_keeps_dropping_until_under_threshold(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "system", "content": "S"}]
        messages += make_round("u1" * 2000, with_tools=False)
        messages += make_round("u2" * 2000, with_tools=False)
        messages += make_round("u3" * 2000, with_tools=False)

        result = truncate_rounds(messages, usage_total_tokens=100_000, max_tokens=1500)

        # 每轮约 2004 字符 → 估算约 1002 tokens；1500 阈值下应一直丢到只剩最后一轮
        assert [m["role"] for m in result].count("user") == 1
        assert result[-1]["content"] == "done"

    def test_run_truncates_between_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            tool_response("c1", "echo", {"text": "x"}, total=50_000),
            text_response("答1", total=50_000),
            tool_response("c2", "echo", {"text": "y"}, total=50_000),
            text_response("答2", total=50_000),
        ]
        agent, _ = make_agent(monkeypatch, script, max_context_tokens=100)

        agent.run("第一轮")
        agent.run("第二轮")

        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["content"] == "第二轮"  # 第一轮已整体丢弃
        assert agent.messages[-1] == {"role": "assistant", "content": "答2"}
        assert_rounds_atomic(agent.messages)


class TestInterrupt:
    def test_keyboard_interrupt_rolls_back_to_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = Agent(client=object(), model="test-model", registry=make_registry())
        agent.messages += make_round("u1", with_tools=False, answer="已答")  # 预置合法历史
        attempts = {"count": 0}

        def fake_complete(*args: object, **kwargs: object) -> llm.LLMResponse:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return tool_response("c9", "echo", {"text": "x"})
            raise KeyboardInterrupt

        monkeypatch.setattr(llm, "complete", fake_complete)

        with pytest.raises(KeyboardInterrupt):
            agent.run("第二轮")

        assert attempts["count"] == 2  # 中断发生在第二个（工具轮）请求上
        assert agent.messages == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "已答"},
        ]

    def test_keyboard_interrupt_on_first_request_rolls_back_user_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = Agent(client=object(), model="test-model", registry=make_registry())

        def fake_complete(*args: object, **kwargs: object) -> llm.LLMResponse:
            raise KeyboardInterrupt

        monkeypatch.setattr(llm, "complete", fake_complete)

        with pytest.raises(KeyboardInterrupt):
            agent.run("刚输入的话")

        assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]

    def test_interrupt_during_tool_execution_after_truncation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """回归（验收走查发现）：截断替换 messages 后，位置索引回滚会失位。

        场景：本轮第一个请求触发截断（丢弃旧轮次、列表变短）→ 工具执行中 Ctrl+C
        → 回滚必须按"本轮起始 user 消息"定位，否则残留孤儿 tool_calls（下轮 400）。
        """
        registry = ToolRegistry()

        def ki_handler(text: str) -> str:
            raise KeyboardInterrupt

        registry.register(
            name="ki",
            description="执行时立即中断",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=ki_handler,
        )
        agent = Agent(
            client=object(), model="test-model", registry=registry, max_context_tokens=100
        )
        agent.messages += make_round("u1", with_tools=False, answer="已答")
        script = [tool_response("c9", "ki", {"text": "x"}, total=50_000)]
        monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: script.pop(0))

        with pytest.raises(KeyboardInterrupt):
            agent.run("第二轮")

        # 第一轮已被截断丢弃（usage 超阈值）；中断回滚再把"第二轮"整体作废——
        # 旧实现此处会残留 [system, 第二轮, 孤儿 tool_calls]（下轮 API 400）
        assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]
        assert_rounds_atomic(agent.messages)

    def test_llm_error_rolls_back_round(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(client=object(), model="test-model", registry=make_registry())
        agent.messages += make_round("u1", with_tools=False, answer="已答")

        def failing_complete(*args: object, **kwargs: object) -> llm.LLMResponse:
            raise llm.LLMError("网络中断")

        monkeypatch.setattr(llm, "complete", failing_complete)

        with pytest.raises(llm.LLMError, match="网络中断"):
            agent.run("第二轮")

        # 请求失败同样作废本轮：不留半截消息污染后续对话
        assert agent.messages == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "已答"},
        ]

    def test_empty_final_answer_raises_readable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = make_agent(monkeypatch, [text_response("  ")])

        with pytest.raises(llm.LLMError, match="空回答"):
            agent.run("x")

        # 空回答按本轮作废处理：不写入空 assistant 消息，也不静默吞掉（§3.4）
        assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]
