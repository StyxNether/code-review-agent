"""llm 补全归一化：流式 tool_calls 分片组装、usage 透出、SDK 异常包装（fake 客户端，无网络）。"""

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIError, OpenAIError

from cra import llm


def _content_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def _tool_piece(
    index: int | None,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _tool_chunk(pieces: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=pieces))],
        usage=None,
    )


def _usage_chunk(total: int) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=total))


def _fake_client(create) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _fake_stream(chunks: list[SimpleNamespace]) -> Callable[[], Any]:
    """返回一个生成器工厂：生成器自带 close()，贴近真实 openai Stream 的形状。"""
    return lambda: (chunk for chunk in chunks)


class TestStreaming:
    def test_plain_content_streams_and_collects(self) -> None:
        chunks: list[SimpleNamespace] = [
            _content_chunk("你好"),
            _content_chunk("世界"),
            _usage_chunk(123),
        ]
        deltas: list[str] = []

        def create(**kwargs: object) -> Iterator[SimpleNamespace]:
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            assert kwargs["temperature"] == llm.TEMPERATURE
            return _fake_stream(chunks)()

        response = llm.complete(
            _fake_client(create), "test-model", [], stream=True, on_delta=deltas.append
        )

        assert deltas == ["你好", "世界"]
        assert response.message.content == "你好世界"
        assert response.message.tool_calls == []
        assert response.total_tokens == 123

    def test_tool_call_fragments_assembled_by_index(self) -> None:
        chunks = [
            _tool_chunk([_tool_piece(0, call_id="c1", name="read_file")]),
            _tool_chunk([_tool_piece(0, arguments='{"pa')]),
            _tool_chunk([_tool_piece(0, arguments='th": "a.py"}')]),
            _tool_chunk([_tool_piece(1, call_id="c2", name="list_dir", arguments="{}")]),
            _usage_chunk(456),
        ]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        calls = response.message.tool_calls
        assert [call.id for call in calls] == ["c1", "c2"]
        assert calls[0].name == "read_file"
        assert calls[0].arguments == '{"path": "a.py"}'
        assert calls[1].arguments == "{}"
        assert response.total_tokens == 456
        assert response.message.content is None

    def test_tool_call_with_stray_content(self) -> None:
        chunks = [
            _content_chunk("我先查一下"),
            _tool_chunk([_tool_piece(0, call_id="c1", name="list_dir", arguments="{}")]),
            _usage_chunk(7),
        ]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert response.message.content == "我先查一下"
        assert response.message.tool_calls[0].name == "list_dir"

    def test_missing_usage_degrades_to_none(self) -> None:
        chunks = [_content_chunk("正文")]  # 服务端未回 usage 也不影响正确性

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert response.message.content == "正文"
        assert response.total_tokens is None


class TestNonStreaming:
    def test_returns_normalized_message(self) -> None:
        sdk_message = SimpleNamespace(
            content="结论",
            tool_calls=[
                SimpleNamespace(
                    id="c1", function=SimpleNamespace(name="list_dir", arguments="{}")
                )
            ],
        )
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(message=sdk_message)],
            usage=SimpleNamespace(total_tokens=99),
        )

        def create(**kwargs: object) -> SimpleNamespace:
            assert not kwargs.get("stream")
            return sdk_response

        response = llm.complete(_fake_client(create), "test-model", [])

        assert response.message.content == "结论"
        assert response.message.tool_calls[0].name == "list_dir"
        assert response.total_tokens == 99

    def test_api_error_wrapped_as_llm_error(self) -> None:
        def create(**_: object) -> None:
            raise APIError("boom", SimpleNamespace(), body=None)

        with pytest.raises(llm.LLMError, match="boom"):
            llm.complete(_fake_client(create), "test-model", [])

    def test_non_api_openai_error_wrapped_as_llm_error(self) -> None:
        """Length/ContentFilterFinishReason 等非 APIError 的 SDK 异常也要被包装。"""

        def create(**_: object) -> None:
            raise OpenAIError("provider exploded")

        with pytest.raises(llm.LLMError, match="provider exploded"):
            llm.complete(_fake_client(create), "test-model", [])

    def test_empty_choices_raises_llm_error(self) -> None:
        sdk_response = SimpleNamespace(choices=[], usage=None)

        with pytest.raises(llm.LLMError, match="choices"):
            llm.complete(_fake_client(lambda **_: sdk_response), "test-model", [])

    def test_tools_none_omitted_from_request(self) -> None:
        """tools 为 None 时省略字段，不依赖 None 的序列化语义。"""
        captured: dict[str, object] = {}
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="好", tool_calls=None))],
            usage=SimpleNamespace(total_tokens=5),
        )

        def create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return sdk_response

        llm.complete(_fake_client(create), "test-model", [])
        assert "tools" not in captured

    def test_tools_passed_when_provided(self) -> None:
        captured: dict[str, object] = {}
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="好", tool_calls=None))],
            usage=SimpleNamespace(total_tokens=5),
        )
        tools = [{"type": "function", "function": {"name": "list_dir"}}]

        def create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return sdk_response

        llm.complete(_fake_client(create), "test-model", [], tools=tools)
        assert captured["tools"] == tools

    def test_stream_close_failure_does_not_mask_response(self) -> None:
        class _ExplodingClose:
            def __init__(self, chunks: list[SimpleNamespace]) -> None:
                self._chunks = iter(chunks)

            def __iter__(self) -> "_ExplodingClose":
                return self

            def __next__(self) -> SimpleNamespace:
                return next(self._chunks)

            def close(self) -> None:
                raise RuntimeError("close failed")

        chunks = [_content_chunk("正文"), _usage_chunk(9)]

        response = llm.complete(
            _fake_client(lambda **_: _ExplodingClose(chunks)), "test-model", [], stream=True
        )

        assert response.message.content == "正文"

    def test_tool_fragments_without_index_keep_arrival_order(self) -> None:
        chunks = [
            _tool_chunk([_tool_piece(None, call_id="c1", name="list_dir", arguments="{}")]),
            _usage_chunk(1),
        ]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert [call.name for call in response.message.tool_calls] == ["list_dir"]


class TestCreateClient:
    def test_missing_api_key_raises_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不依赖 SDK 的 OPENAI_API_KEY 回退（避免把 OpenAI 密钥发往 CRA_BASE_URL）。"""
        monkeypatch.delenv("SE_CodeAgent", raising=False)
        with pytest.raises(llm.LLMError, match="SE_CodeAgent"):
            llm.create_client()

    def test_sdk_retry_disabled_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SDK 隐式重试显式关闭，重试策略由 M2 在应用层实现。"""
        monkeypatch.setenv("SE_CodeAgent", "sk-test-key")
        assert llm.create_client().max_retries == 0
