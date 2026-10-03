"""Agent 循环（create_agent 栈）：消息正本维护、工具桥接与结构化错误、轮次上限
middleware、请求侧轮次原子截断、中断回滚（全假模型，无网络）。"""

import json
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from cra import config, llm
from cra.agent import (
    MAX_TOOL_ROUNDS,
    Agent,
    _bridge_tool,
    _ContextTrimMiddleware,
    _MaxToolRoundsMiddleware,
    _ToolCallbacksMiddleware,
    truncate_rounds,
)
from cra.prompts import SYSTEM_PROMPT
from cra.tools import ToolRegistry


class ScriptedChatModel(BaseChatModel):
    """按脚本回放消息的假模型：记录收到的请求与每次 bind_tools 的工具集。

    脚本项为 AIMessage（正常回放）或 Exception 实例（到达时抛出，模拟请求失败/
    中断）。派生类实现 _stream 即获得流式行为。
    """

    responses: list[Any] = Field(default_factory=list)
    cursor: int = 0
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools.append(list(tools))
        return self

    def bind(self, **kwargs: Any) -> "ScriptedChatModel":
        # 框架在请求无工具可绑时的路径（去工具后的模型调用）：记录为空绑定
        self.bound_tools.append([])
        return self

    def _next_response(self, messages: list[BaseMessage]) -> Any:
        """取出下一条脚本项并记录请求（_generate/_stream 共用）。"""
        self.requests.append(list(messages))
        response = self.responses[self.cursor]
        self.cursor += 1
        if isinstance(response, BaseException):
            raise response
        return response

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        response = self._next_response(list(messages))
        return ChatResult(generations=[ChatGeneration(message=response)])


class StreamingScriptedChatModel(ScriptedChatModel):
    """带流式的假模型：文本按 3 段切片（末块带 finish_reason），usage 独立末块。"""

    def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        response = self._next_response(list(messages))
        if response.tool_calls:
            # tool_call_chunks 聚合后须还原出完整 tool_calls（ToolNode 据此执行）
            for call in response.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"],
                                "args": json.dumps(call["args"]),
                                "id": call["id"],
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
        else:
            text = response.text
            step = max(1, len(text) // 3)
            pieces = [text[i : i + step] for i in range(0, len(text), step)] or [""]
            for i, piece in enumerate(pieces):
                extra: dict[str, Any] = {}
                if i == len(pieces) - 1 and response.response_metadata.get("finish_reason"):
                    extra["response_metadata"] = response.response_metadata
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece, **extra))
        if response.usage_metadata:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=response.usage_metadata)
            )


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


def tool_response(call_id: str, name: str, arguments: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


def text_response(
    text: str,
    finish_reason: str | None = "stop",
    total_tokens: int | None = None,
) -> AIMessage:
    metadata: dict[str, Any] = {"finish_reason": finish_reason} if finish_reason else {}
    usage = (
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": total_tokens}
        if total_tokens is not None
        else None
    )
    return AIMessage(content=text, response_metadata=metadata, usage_metadata=usage)


def make_agent(
    script: list[Any],
    *,
    registry: ToolRegistry | None = None,
    max_context_tokens: int | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    streaming: bool = False,
) -> tuple[Agent, ScriptedChatModel]:
    """构建注入假模型的 Agent，返回 (agent, fake)——fake.requests/bound_tools 供断言。"""
    fake: ScriptedChatModel = (
        StreamingScriptedChatModel(responses=script) if streaming else ScriptedChatModel(responses=script)
    )
    kwargs: dict[str, Any] = {"model": fake, "registry": registry or make_registry()}
    if max_context_tokens is not None:
        kwargs["max_context_tokens"] = max_context_tokens
    if max_tool_rounds != MAX_TOOL_ROUNDS:
        kwargs["max_tool_rounds"] = max_tool_rounds
    return Agent(**kwargs), fake


def make_round(user_text: str, *, with_tools: bool = True, answer: str = "done") -> list[BaseMessage]:
    """构造一个结构完整的历史轮次：user → assistant(tool_calls) → tool → assistant。"""
    messages: list[BaseMessage] = [HumanMessage(content=user_text)]
    if with_tools:
        call_id = f"call-{user_text[:8]}"
        messages.append(tool_response(call_id, "echo", {"text": "x"}))
        messages.append(ToolMessage(content="ECHO:x", tool_call_id=call_id))
    messages.append(AIMessage(content=answer))
    return messages


def assert_rounds_atomic(messages: list[BaseMessage]) -> None:
    """消息序列合法性：每条 tool 消息都对应仍在等待结果的 tool_call，且最终无悬挂。"""
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                pending.add(call["id"])
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, f"tool 消息悬挂：{message}"
            pending.discard(message.tool_call_id)
    assert not pending, f"assistant 的 tool_calls 没有对应 tool 结果：{pending}"


def shape(messages: list[BaseMessage]) -> list[tuple[str, Any]]:
    """消息序列的语义形状（类型 + 内容）：框架会给消息原位分配 id，不做对象等值。"""
    return [(type(m).__name__, m.content) for m in messages]


def _ki_handler(text: str) -> str:
    raise KeyboardInterrupt


class TestRunBasics:
    def test_initial_messages_are_system_only(self) -> None:
        agent, _ = make_agent([])

        assert agent.messages == [SystemMessage(content=SYSTEM_PROMPT)]

    def test_reset_restores_system_only(self) -> None:
        agent, _ = make_agent([text_response("答")])
        agent.run("x")

        agent.reset()

        assert agent.messages == [SystemMessage(content=SYSTEM_PROMPT)]

    def test_plain_answer_assembles_master(self) -> None:
        agent, _ = make_agent([text_response("结论")])

        result = agent.run("审查 agent.py")

        assert result == "结论"
        assert shape(agent.messages) == [
            ("SystemMessage", SYSTEM_PROMPT),
            ("HumanMessage", "审查 agent.py"),
            ("AIMessage", "结论"),
        ]

    def test_system_prompt_injected_via_system_channel(self) -> None:
        """系统提示词经框架 system 通道每次注入（请求首条消息），不落入图状态。"""
        agent, fake = make_agent([text_response("结论")])

        agent.run("x")

        assert isinstance(fake.requests[0][0], SystemMessage)
        assert fake.requests[0][0].content == SYSTEM_PROMPT
        assert [m.content for m in fake.requests[0][1:]] == ["x"]

    def test_tool_round_dispatch_and_pairing(self) -> None:
        agent, _ = make_agent([tool_response("c1", "echo", {"text": "hi"}), text_response("结论")])

        result = agent.run("帮我")

        assert result == "结论"
        assert [type(m).__name__ for m in agent.messages] == [
            "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
        ]
        assert agent.messages[2].tool_calls[0]["name"] == "echo"
        assert agent.messages[3].content == "ECHO:hi"
        assert_rounds_atomic(agent.messages)

    def test_tool_callbacks_fire_with_json_args(self) -> None:
        agent, _ = make_agent([tool_response("c1", "echo", {"text": "hi"}), text_response("好")])
        tools: list[tuple[str, str]] = []
        results: list[str] = []

        agent.run(
            "x", on_tool=lambda name, args: tools.append((name, args)), on_tool_result=results.append
        )

        assert tools == [("echo", json.dumps({"text": "hi"}, ensure_ascii=False))]
        assert json.loads(tools[0][1]) == {"text": "hi"}
        assert results == ["ECHO:hi"]

    def test_multiple_tool_calls_dispatched_in_order(self) -> None:
        script = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "一"}, "id": "c1", "type": "tool_call"},
                    {"name": "echo", "args": {"text": "二"}, "id": "c2", "type": "tool_call"},
                ],
            ),
            text_response("结论"),
        ]
        agent, _ = make_agent(script)
        tools: list[tuple[str, str]] = []

        agent.run("x", on_tool=lambda name, args: tools.append((name, args)))

        assert [args for _, args in tools] == [
            json.dumps({"text": "一"}, ensure_ascii=False),
            json.dumps({"text": "二"}, ensure_ascii=False),
        ]
        assert agent.messages[4].content == "ECHO:二"
        assert_rounds_atomic(agent.messages)

    def test_unknown_tool_error_prefix_reaches_model(self) -> None:
        """未知工具名：模型必须收到"错误："前缀的结构化文案（协议锁定点）。"""
        agent, fake = make_agent([tool_response("c1", "no_such_tool", {}), text_response("好")])

        agent.run("x")

        tool_messages = [m for m in fake.requests[1] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content.startswith("错误")
        assert "未注册" in tool_messages[0].content
        assert "no_such_tool" in tool_messages[0].content

    def test_unknown_tool_error_reported_via_callback(self) -> None:
        agent, _ = make_agent([tool_response("c1", "no_such_tool", {}), text_response("好")])
        results: list[str] = []

        agent.run("x", on_tool_result=results.append)

        assert results and results[0].startswith("错误")

    def test_tool_output_starting_with_error_not_rewritten(self) -> None:
        """工具的正常输出恰以 'Error:' 开头时不被误改写。

        判别依据是 ToolMessage.status 语义字段（正常输出 success），不是文案前缀。
        """
        registry = make_registry()
        registry.register(
            name="errish",
            description="返回以 Error: 开头的正常输出",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "Error: 这只是工具的正常输出",
        )
        agent, fake = make_agent(
            [tool_response("c1", "errish", {}), text_response("好")], registry=registry
        )
        results: list[str] = []

        agent.run("x", on_tool_result=results.append)

        tool_messages = [m for m in fake.requests[1] if isinstance(m, ToolMessage)]
        assert tool_messages[0].content == "Error: 这只是工具的正常输出"  # 原样回传模型
        assert results == ["Error: 这只是工具的正常输出"]  # cli 亦按正常结果展示

    def test_empty_tool_result_still_fires_callback(self) -> None:
        """三分支回调协议一致——空结果也触发 on_tool_result。"""
        registry = make_registry()
        registry.register(
            name="empty",
            description="返回空串",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "",
        )
        agent, _ = make_agent([tool_response("c1", "empty", {}), text_response("好")], registry=registry)
        results: list[str] = []

        agent.run("x", on_tool_result=results.append)

        assert results == [""]

    def test_reset_clears_pending_callbacks(self) -> None:
        agent, _ = make_agent([text_response("答")])
        agent.run("x", on_tool=lambda name, args: None)
        assert agent._tool_callbacks.callbacks is None  # 正常返回即清
        agent._tool_callbacks.callbacks = (lambda *a: None, lambda *a: None)  # 模拟异常挂留

        agent.reset()

        assert agent._tool_callbacks.callbacks is None

    def test_delta_callback_fires_for_final_answer(self) -> None:
        deltas: list[str] = []
        agent, _ = make_agent([text_response("流式回答")], streaming=True)

        agent.run("x", on_delta=deltas.append)

        assert "".join(deltas) == "流式回答"
        assert len(deltas) >= 2  # 分片到达，而非一次性全文

    def test_display_model_name_falls_back_to_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setenv("CRA_MODEL", "env-model")

        agent, _ = make_agent([])

        assert agent.model == "env-model"


class TestFinalAnswerHandling:
    def test_length_truncated_answer_gets_visible_notice(self) -> None:
        """finish_reason=length 的回答不能被静默当完整结论。

        提示须同时到达三处：渲染增量（cli 只认 on_delta）、返回值、正本
        （保证"继续"追问时模型可见截断状态）。
        """
        deltas: list[str] = []
        agent, _ = make_agent(
            [text_response("被截断的回答", finish_reason="length")], streaming=True
        )

        result = agent.run("x", on_delta=deltas.append)

        assert result.startswith("被截断的回答")
        assert "截断" in result
        assert deltas[-1].strip().startswith("⚠")
        assert agent.messages[-1].content == result

    def test_normal_stop_reason_adds_no_notice(self) -> None:
        agent, _ = make_agent([text_response("正常回答")])

        assert agent.run("x") == "正常回答"

    def test_empty_final_answer_raises_and_rolls_back(self) -> None:
        agent, _ = make_agent([text_response("  ")])

        with pytest.raises(llm.LLMError, match="空回答"):
            agent.run("x")

        assert agent.messages == [SystemMessage(content=SYSTEM_PROMPT)]

    def test_final_call_carries_full_history(self) -> None:
        agent, fake = make_agent(
            [tool_response("c1", "echo", {"text": "hi"}), text_response("结论")]
        )

        agent.run("x")

        # 最终回答请求发生在追加 assistant 回答之前：正本去掉系统消息与末条回答
        assert fake.requests[-1] == [SystemMessage(content=SYSTEM_PROMPT), *agent.messages[1:-1]]

    def test_no_new_messages_does_not_overwrite_user_input(self) -> None:
        """新增消息为空时不得把本轮 user 消息覆盖成回答。"""
        agent, _ = make_agent([text_response("答", finish_reason="length")])
        truncated_answer = text_response("答", finish_reason="length")

        def fake_invoke(fed: list[BaseMessage], on_delta: object) -> tuple[dict[str, Any], AIMessage]:
            return {"messages": list(fed)}, truncated_answer  # 仿佛本轮没有任何新增消息

        agent._invoke = fake_invoke  # type: ignore[method-assign]

        result = agent.run("x")

        assert result.startswith("答")
        # 正本仍是 [system, user]：截断提示无处写入，user 消息原样保留
        assert len(agent.messages) == 2
        assert agent.messages[-1].content == "x"


class TestMaxToolRounds:
    def test_forces_conclusion_after_max_rounds(self) -> None:
        script = [tool_response(f"c{i}", "echo", {"text": str(i)}) for i in range(MAX_TOOL_ROUNDS)]
        script.append(text_response("最终结论"))
        agent, _ = make_agent(script)

        result = agent.run("x")

        assert result == "最终结论"
        tool_messages = [m for m in agent.messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == MAX_TOOL_ROUNDS
        assert agent.messages[-1].content == "最终结论"
        assert_rounds_atomic(agent.messages)

    def test_forced_call_has_no_tools_and_injected_directive(self) -> None:
        """达限机制锁定：下一次模型调用去工具 + 收尾指令经 system 通道注入。"""
        script = [tool_response(f"c{i}", "echo", {"text": str(i)}) for i in range(MAX_TOOL_ROUNDS)]
        script.append(text_response("最终结论"))
        agent, fake = make_agent(script)

        agent.run("x")

        assert fake.bound_tools[-1] == []  # 去工具
        assert fake.bound_tools[:-1] and all(fake.bound_tools[:-1])  # 其余调用都带工具
        system_message = fake.requests[-1][0]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content.startswith(SYSTEM_PROMPT)  # 原系统提示词保留在前
        assert "上限" in system_message.content  # 收尾指令追加于 system 通道
        # 请求中除本轮真实输入外无伪造 user 消息——截断的轮次边界不被指令污染
        humans = [m for m in fake.requests[-1] if isinstance(m, HumanMessage)]
        assert [m.content for m in humans] == ["x"]

    def test_directive_not_persisted_in_master(self) -> None:
        """收尾指令只在请求侧（D14）：正本不出现注入的 user 消息。"""
        script = [tool_response(f"c{i}", "echo", {"text": str(i)}) for i in range(MAX_TOOL_ROUNDS)]
        script.append(text_response("最终结论"))
        agent, _ = make_agent(script)

        agent.run("x")

        assert not any(isinstance(m, HumanMessage) and "上限" in m.content for m in agent.messages)

    def test_round_counting_resets_per_run(self) -> None:
        """轮次计数只覆盖本轮：上一轮用过的工具轮不累积进本次限额。"""
        agent, _ = make_agent(
            [
                tool_response("c1", "echo", {"text": "1"}),
                text_response("答1"),
                tool_response("c2", "echo", {"text": "2"}),
                text_response("答2"),
            ]
        )

        assert agent.run("第一轮") == "答1"
        assert agent.run("第二轮") == "答2"  # 若计数跨轮累积，第二轮会被立即强制收尾

    def test_recursion_limit_scales_with_rounds(self) -> None:
        agent, _ = make_agent([], max_tool_rounds=3)

        assert agent.recursion_limit == 2 * 3 + 5


class TestMalformedToolCallArgs:
    def test_malformed_args_normalized_to_error_prefix_in_request(self) -> None:
        """畸形 JSON 参数：框架在模型请求侧合成的提示须归一为"错误："前缀。"""
        mixed = AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "ok"}, "id": "c1", "type": "tool_call"}],
            invalid_tool_calls=[
                {
                    "name": "echo",
                    "args": "{bad json",
                    "id": "cX",
                    "error": "oops",
                    "type": "invalid_tool_call",
                }
            ],
        )
        agent, fake = make_agent([mixed, text_response("结论")])

        agent.run("x")

        malformed = [
            m
            for m in fake.requests[1]
            if isinstance(m, ToolMessage) and "could not be executed" in m.content
        ]
        assert len(malformed) == 1
        assert malformed[0].content.startswith("错误：")
        # 正本仍以有效调用收尾；合成 ToolMessage 的 id 只在 invalid_tool_calls 中
        # （框架行为：不入 API 序列化，对严格校验的服务端存在孤儿风险）
        assert agent.messages[-1].content == "结论"
        valid_tool = [m for m in agent.messages if isinstance(m, ToolMessage) and m.tool_call_id == "c1"]
        assert len(valid_tool) == 1


class TestTruncateRounds:
    def test_under_threshold_untouched(self) -> None:
        messages = make_round("u1", with_tools=False)
        result = truncate_rounds(messages, usage_total_tokens=100, max_tokens=200)
        assert result == messages
        assert result is not messages  # 未触发截断也返回副本（纯函数语义）

    def test_none_usage_untouched(self) -> None:
        messages = make_round("u1", with_tools=False)
        assert truncate_rounds(messages, usage_total_tokens=None, max_tokens=10) == messages

    def test_exactly_at_threshold_untouched(self) -> None:
        """边界：usage == 阈值不触发截断（仅"超过"才丢轮次）。"""
        messages = make_round("u1", with_tools=False)

        result = truncate_rounds(messages, usage_total_tokens=200, max_tokens=200)

        assert result == messages

    def test_one_token_over_threshold_drops_oldest(self) -> None:
        messages = [*make_round("u1", with_tools=False), *make_round("u2", with_tools=False)]

        result = truncate_rounds(messages, usage_total_tokens=201, max_tokens=200)

        assert result[0].content == "u2"  # 超过 1 token 也按轮次整体丢最旧一轮

    def test_estimate_boundary_stops_dropping(self) -> None:
        """边界：估算降到阈值以内即停（两轮各恰 1000 字符 ≈ 600 估算 tokens，阈值 600）。

        估算含 1.2 倍安全余量（1000 字符 × 0.6）；丢一轮后估算 == 阈值，不再继续丢。
        """
        messages = [
            *make_round("u" * 996, with_tools=False),
            *make_round("v" * 996, with_tools=False),
        ]

        result = truncate_rounds(messages, usage_total_tokens=100_000, max_tokens=600)

        assert sum(isinstance(m, HumanMessage) for m in result) == 1
        assert result[-1].content == "done"

    def test_over_threshold_drops_oldest_round_atomically(self) -> None:
        messages = [*make_round("u1" * 200), *make_round("u2", with_tools=False)]
        original = list(messages)

        result = truncate_rounds(messages, usage_total_tokens=50_000, max_tokens=100)

        assert result[0].content == "u2"  # 第一轮整体消失
        assert any(m.content == "done" for m in result)  # 最后一轮完整保留
        assert_rounds_atomic(result)  # tool_calls/tool 配对不被拆散
        assert messages == original  # 纯函数：不改入参

    def test_never_drops_last_round(self) -> None:
        messages = make_round("u1" * 400)

        result = truncate_rounds(messages, usage_total_tokens=999_999, max_tokens=10)

        assert result == messages  # 单轮不可拆：宁可超限也不产生非法结构
        assert result is not messages  # 且始终返回副本（纯函数语义）

    def test_estimate_keeps_dropping_until_under_threshold(self) -> None:
        messages = [
            *make_round("u1" * 2000, with_tools=False),
            *make_round("u2" * 2000, with_tools=False),
            *make_round("u3" * 2000, with_tools=False),
        ]

        result = truncate_rounds(messages, usage_total_tokens=100_000, max_tokens=1500)

        assert sum(isinstance(m, HumanMessage) for m in result) == 1
        assert result[-1].content == "done"


class TestContextTrimIntegration:
    def test_trim_applies_to_request_but_master_keeps_full_history(self) -> None:
        """M4 新语义锁定：截断发生在请求侧，正本（D14）保留完整历史。"""
        script = [
            tool_response("c1", "echo", {"text": "x"}),
            text_response("答1", total_tokens=50_000),
            text_response("答2", total_tokens=50_000),
        ]
        agent, fake = make_agent(script, max_context_tokens=100)

        agent.run("第一轮")
        agent.run("第二轮")

        # 第二轮首次请求：第一轮已按完整轮次从请求视图丢弃（视图首条即本轮 user）
        second_first = fake.requests[2][1]  # [0] 是 system 通道消息
        assert second_first.content == "第二轮"
        assert all(not isinstance(m, AIMessage) for m in fake.requests[2][1:])
        # 正本保留完整历史（含第一轮），仅追加本轮新增
        assert [type(m).__name__ for m in agent.messages] == [
            "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
            "HumanMessage", "AIMessage",
        ]
        assert agent.messages[-1].content == "答2"
        assert_rounds_atomic(agent.messages)

    def test_missing_usage_never_trims(self) -> None:
        """usage 缺失（兼容端点未回）安全降级：截断不触发，请求视图不裁剪。"""
        script = [
            tool_response("c1", "echo", {"text": "x"}),
            text_response("答1", total_tokens=None),
            text_response("答2", total_tokens=None),
        ]
        agent, fake = make_agent(script, max_context_tokens=10)

        agent.run("第一轮")
        agent.run("第二轮")

        assert fake.requests[2][1].content == "第一轮"  # 请求仍携带完整历史
        assert len(agent.messages) == 7

    def test_trim_middleware_unit(self) -> None:
        """middleware 直接单测：构造 ModelRequest 验证裁剪行为与请求侧语义。"""
        from langchain.agents.middleware import ModelRequest

        middleware = _ContextTrimMiddleware(max_context_tokens=100)
        state_messages = [*make_round("u1" * 200), *make_round("u2", with_tools=False)]
        for message in state_messages:
            if isinstance(message, AIMessage) and not message.tool_calls:
                message.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 50_000}
                break
        request = ModelRequest(
            model=ScriptedChatModel(responses=[]), messages=list(state_messages), state={"messages": state_messages}
        )
        captured: dict[str, Any] = {}

        def handler(request: ModelRequest) -> str:
            captured["messages"] = list(request.messages)
            return "ok"

        result = middleware.wrap_model_call(request, handler)

        assert result == "ok"
        assert captured["messages"][0].content == "u2"  # 请求视图已裁剪
        assert state_messages[0].content == "u1" * 200  # 传入的列表（正本侧）不动


class TestInterrupt:
    def test_keyboard_interrupt_rolls_back_to_prefix(self) -> None:
        agent, fake = make_agent([])
        agent.messages += make_round("u1", with_tools=False, answer="已答")  # 预置合法历史
        fake.responses.extend([tool_response("c9", "echo", {"text": "x"}), KeyboardInterrupt()])

        with pytest.raises(KeyboardInterrupt):
            agent.run("第二轮")

        # 中断发生在第二个（工具轮）请求上；正本回滚到本轮开始前的完整历史
        assert fake.cursor == 2
        assert shape(agent.messages) == [
            ("SystemMessage", SYSTEM_PROMPT),
            ("HumanMessage", "u1"),
            ("AIMessage", "已答"),
        ]

    def test_keyboard_interrupt_on_first_request_rolls_back_user_input(self) -> None:
        agent, _ = make_agent([KeyboardInterrupt()])

        with pytest.raises(KeyboardInterrupt):
            agent.run("刚输入的话")

        assert agent.messages == [SystemMessage(content=SYSTEM_PROMPT)]

    def test_interrupt_during_tool_execution_rolls_back(self) -> None:
        """工具执行中的 Ctrl+C（BaseException 不被工具兜底吞掉）：本轮整体作废。"""
        registry = make_registry()
        registry.register(
            name="ki",
            description="执行时立即中断",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=_ki_handler,
        )
        agent, _ = make_agent([tool_response("c9", "ki", {"text": "x"})], registry=registry)
        agent.messages += make_round("u1", with_tools=False, answer="已答")

        with pytest.raises(KeyboardInterrupt):
            agent.run("第二轮")

        assert shape(agent.messages) == [
            ("SystemMessage", SYSTEM_PROMPT),
            ("HumanMessage", "u1"),
            ("AIMessage", "已答"),
        ]

    def test_unexpected_error_propagates_unchanged_and_rolls_back(self) -> None:
        """非 SDK/框架异常原样上抛（不伪装成 LLMError），本轮同样作废。"""
        agent, fake = make_agent([RuntimeError("代码缺陷")])
        agent.messages += make_round("u1", with_tools=False, answer="已答")

        with pytest.raises(RuntimeError, match="代码缺陷"):
            agent.run("第二轮")

        assert shape(agent.messages) == [
            ("SystemMessage", SYSTEM_PROMPT),
            ("HumanMessage", "u1"),
            ("AIMessage", "已答"),
        ]
        assert fake.cursor == 1

    def test_recursion_limit_backstop_translated_as_llm_error(self) -> None:
        """护栏：轮次 middleware 失效时 recursion_limit 触发，翻译为可读 LLMError 并回滚。"""
        registry = make_registry()
        agent, _ = make_agent([], registry=registry, max_tool_rounds=1)
        agent._graph = _build_graph_without_rounds_guard(agent)

        with pytest.raises(llm.LLMError, match="recursion_limit"):
            agent.run("x")
        assert shape(agent.messages) == [("SystemMessage", SYSTEM_PROMPT)]


class _AlwaysToolCallsModel(BaseChatModel):
    """永远请求 echo 工具的假模型（配合失效的轮次 middleware 触发 recursion_limit）。"""

    @property
    def _llm_type(self) -> str:
        return "always-tool-calls"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_AlwaysToolCallsModel":
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        message = tool_response(f"c{len(messages)}", "echo", {"text": "y"})
        return ChatResult(generations=[ChatGeneration(message=message)])


def _build_graph_without_rounds_guard(agent: Agent) -> Any:
    """以恒调用工具的假模型重建图：轮次 middleware 置为失效值，验证护栏兜底。"""
    return create_agent(
        model=_AlwaysToolCallsModel(),
        tools=[_bridge_tool(agent.registry, tool_def) for tool_def in agent.registry.definitions()],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            _ToolCallbacksMiddleware(agent.registry),
            _MaxToolRoundsMiddleware(999),  # 故意失效：使轮次突破 agent.recursion_limit 对应范围
            _ContextTrimMiddleware(agent.max_context_tokens),
        ],
    )


class TestToolBridging:
    def test_bridge_tools_expose_registry_schemas_verbatim(self) -> None:
        """模型可见 schema 与注册表逐字段一致（保真，行为锁定点）。"""
        agent, fake = make_agent([text_response("好")])
        agent.run("x")  # 触发一次绑定

        bound = fake.bound_tools[0]
        assert bound  # 首次调用带工具
        registry_schemas = agent.registry.schemas()
        assert len(bound) == len(registry_schemas)
        for tool, schema in zip(bound, registry_schemas):
            assert convert_to_openai_tool(tool) == schema

    def test_bridge_tool_delegates_to_dispatch(self) -> None:
        """桥接工具执行统一委托注册表 dispatch（handler 结果直达模型）。"""
        agent, fake = make_agent([text_response("好")])

        agent.run("x")

        echo_tool = fake.bound_tools[0][0]
        result = echo_tool.invoke(
            {"type": "tool_call", "args": {"text": "直呼"}, "id": "cx", "name": "echo"}
        )
        assert result.content == "ECHO:直呼"


# 测试占位值(明显假值,非真实凭据);以变量构造避免密钥扫描对字面量的误报
_PLACEHOLDER_KEY = "placeholder-a"
_PLACEHOLDER_URL = "https://probe.example.com"


def _exec_registry(calls: list[str]) -> ToolRegistry:
    """带 run_python 工具的测试注册表:handler 记录被调用(供确认拦截断言)。"""

    def run(code: str) -> str:
        calls.append(code)
        return "ran"

    registry = ToolRegistry()
    registry.register(
        name="run_python",
        description="执行",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        handler=run,
    )
    return registry


class TestSwitchModel:
    """/change model 的底层能力:重建图、上下文正本保留。"""

    def _patch_create_model(
        self, monkeypatch: pytest.MonkeyPatch, new_fake: ScriptedChatModel
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_create_model(
            model_name: str, *, api_key: str | None = None, base_url: str | None = None
        ) -> ScriptedChatModel:
            captured.update(model_name=model_name, api_key=api_key, base_url=base_url)
            return new_fake

        monkeypatch.setattr(llm, "create_model", fake_create_model)
        return captured

    def test_switch_keeps_messages_and_updates_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = make_agent([text_response("第一轮"), text_response("切换后")])
        agent.run("第一问")
        before = list(agent.messages)

        new_fake = ScriptedChatModel(responses=[text_response("切换后")])
        captured = self._patch_create_model(monkeypatch, new_fake)
        resolved = config.ResolvedModel(
            model="new-model", base_url=_PLACEHOLDER_URL, api_key=_PLACEHOLDER_KEY, provider="prov"
        )
        agent.switch_model(resolved)
        agent.run("切换后再问")

        assert agent.messages[: len(before)] == before  # 正本保留,切换后新消息追加在后
        assert agent.model == "new-model"
        assert captured == {
            "model_name": "new-model",
            "api_key": _PLACEHOLDER_KEY,
            "base_url": _PLACEHOLDER_URL,
        }
        assert new_fake.requests  # 切换后的 run 走新模型

    def test_switch_failure_keeps_old_model_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """构建失败(如 key 缺失)时旧模型与图保持可用(先构建后赋值)。"""
        agent, _ = make_agent([text_response("第一轮"), text_response("旧模型兜底")])
        agent.run("第一问")

        def broken_create_model(*_args: Any, **_kwargs: Any) -> ScriptedChatModel:
            raise llm.LLMError("key 缺失")

        monkeypatch.setattr(llm, "create_model", broken_create_model)
        resolved = config.ResolvedModel(
            model="x", base_url=_PLACEHOLDER_URL, api_key=_PLACEHOLDER_KEY, provider=None
        )
        with pytest.raises(llm.LLMError):
            agent.switch_model(resolved)

        monkeypatch.undo()
        agent.run("再问")  # 旧模型仍可对话
        assert agent.model != "x"


class TestExecConfirm:
    """--ask-exec / /confirm 的底层钩子:拒绝时结构化回传模型、不执行。"""

    def test_rejected_exec_not_run_and_reported(self) -> None:
        calls: list[str] = []
        fake = ScriptedChatModel(
            responses=[
                tool_response("c1", "run_python", {"code": "x"}),
                text_response("收到,仅静态分析。"),
            ]
        )
        agent = Agent(
            model=fake,
            registry=_exec_registry(calls),
            confirm_exec=lambda _args: False,
        )

        agent.run("帮我执行")

        assert calls == []  # 未执行
        tool_messages = [m for m in fake.requests[1] if isinstance(m, ToolMessage)]
        assert tool_messages and tool_messages[0].content.startswith("错误：")
        assert "拒绝" in tool_messages[0].content

    def test_confirmed_exec_runs(self) -> None:
        calls: list[str] = []
        agent = Agent(
            model=ScriptedChatModel(
                responses=[
                    tool_response("c1", "run_python", {"code": "x"}),
                    text_response("结论。"),
                ]
            ),
            registry=_exec_registry(calls),
            confirm_exec=lambda _args: True,
        )

        agent.run("帮我执行")

        assert calls == ["x"]

    def test_confirm_none_runs_directly(self) -> None:
        """默认自动执行(--ask-exec 缺省)。"""
        calls: list[str] = []
        agent = Agent(
            model=ScriptedChatModel(
                responses=[
                    tool_response("c1", "run_python", {"code": "x"}),
                    text_response("结论。"),
                ]
            ),
            registry=_exec_registry(calls),
        )

        agent.run("帮我执行")

        assert calls == ["x"]


class TestUndoAndContext:
    """/undo 与 /context 的底层能力。"""

    def test_undo_removes_last_round(self) -> None:
        agent, _ = make_agent([text_response("答一"), text_response("答二")])
        agent.run("第一问")
        agent.run("第二问")

        assert agent.undo_turn() is True
        assert shape(agent.messages) == [
            ("SystemMessage", SYSTEM_PROMPT),
            ("HumanMessage", "第一问"),
            ("AIMessage", "答一"),
        ]

    def test_undo_without_turns_returns_false(self) -> None:
        agent, _ = make_agent([])

        assert agent.undo_turn() is False
        assert agent.messages == [SystemMessage(content=SYSTEM_PROMPT)]

    def test_context_uses_real_usage(self) -> None:
        agent, _ = make_agent([text_response("答", total_tokens=1234)], max_context_tokens=5000)
        agent.run("问")

        status = agent.context_status()

        assert status == {"used": 1234, "limit": 5000, "estimated": False}

    def test_context_estimates_without_usage(self) -> None:
        """usage 缺失(服务端未回)时字符估算降级并标注(安全降级)。"""
        agent, _ = make_agent([text_response("答", total_tokens=None)], max_context_tokens=5000)
        agent.run("问")

        status = agent.context_status()

        assert status["estimated"] is True
        assert status["used"] > 0
        assert status["limit"] == 5000


class TestSessionPersistence:
    """/save /load 的底层能力:框架序列化留在 agent 层、系统提示词不入档。"""

    def test_dump_skips_system_prompt(self) -> None:
        agent, _ = make_agent([text_response("答")])
        agent.run("问")

        data = agent.dump_session()

        assert len(data) == 2
        assert data[0]["type"] == "human"

    def test_dump_load_roundtrip_preserves_tool_round(self) -> None:
        agent, _ = make_agent([])
        agent.messages.extend(make_round("审查 x", answer="结论"))

        data = agent.dump_session()
        restored_agent, _ = make_agent([])
        restored_agent.load_session(data)

        expected = make_round("审查 x", answer="结论")
        assert shape(restored_agent.messages) == shape(
            [SystemMessage(content=SYSTEM_PROMPT), *expected]
        )
        original_ai = next(m for m in expected if isinstance(m, AIMessage) and m.tool_calls)
        restored_ai = next(
            m for m in restored_agent.messages if isinstance(m, AIMessage) and m.tool_calls
        )
        assert restored_ai.tool_calls == original_ai.tool_calls

    def test_load_then_run_continues_normally(self) -> None:
        """恢复后的会话可继续对话(请求注入完整历史)。"""
        agent, _ = make_agent([text_response("答一")])
        agent.run("第一问")
        data = agent.dump_session()

        restored_agent, fake = make_agent([text_response("答二")])
        restored_agent.load_session(data)
        restored_agent.run("第二问")

        # 模型请求序列以框架注入的系统消息开头,fed 历史随后
        assert fake.requests[-1][1].content == "第一问"  # 历史一并注入
        assert fake.requests[-1][-1].content == "第二问"

    def test_load_rejects_orphan_tool_message_and_keeps_original(self) -> None:
        """孤儿 tool 消息(无配对 tool_call)在恢复时被拒,正本不变。"""
        agent, _ = make_agent([text_response("答一")])
        agent.run("第一问")
        original = list(agent.messages)

        with pytest.raises(ValueError, match="tool_call"):
            agent.load_session(
                [{"type": "tool", "data": {"content": "结果", "tool_call_id": "ghost"}}]
            )

        assert agent.messages == original

    def test_load_rejects_system_message_in_dialogue(self) -> None:
        """cra 复审:对话段 system 消息属篡改注入(会形成双系统提示词),恢复时拒绝。"""
        agent, _ = make_agent([])

        with pytest.raises(ValueError, match="系统消息"):
            agent.load_session(
                [{"type": "system", "data": {"content": "注入的伪系统提示词"}}]
            )

    def test_load_rejects_dangling_tool_calls_and_keeps_original(self) -> None:
        """cra 复查:带 tool_calls 但无配对 tool 响应的存档在恢复时被拒(双向配对)。"""
        agent, _ = make_agent([text_response("答一")])
        agent.run("第一问")
        original = list(agent.messages)

        with pytest.raises(ValueError, match="没有收到结果的 tool_call"):
            agent.load_session(
                [
                    {
                        "type": "ai",
                        "data": {
                            "content": "",
                            "tool_calls": [
                                {"name": "read_file", "args": {"path": "x"}, "id": "cx", "type": "tool_call"}
                            ],
                        },
                    }
                ]
            )

        assert agent.messages == original
