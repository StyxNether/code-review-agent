"""llm 补全归一化与重试：流式 tool_calls 分片组装、usage 透出、SDK 异常包装、
指数退避重试与认证快速失败（fake 客户端，无网络）。"""

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from cra import llm


def _content_chunk(text: str, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=finish_reason
            )
        ],
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


def _tool_chunk(
    pieces: list[SimpleNamespace], finish_reason: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=pieces), finish_reason=finish_reason)
        ],
        usage=None,
    )


def _usage_chunk(total: int) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=total))


def _fake_client(create) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _fake_stream(chunks: list[SimpleNamespace]) -> Callable[[], Any]:
    """返回一个生成器工厂：生成器自带 close()，贴近真实 openai Stream 的形状。"""
    return lambda: (chunk for chunk in chunks)


def _http_error(
    cls: type[APIStatusError], status_code: int
) -> APIStatusError:
    """构造带 status_code 的 APIStatusError 系异常（response 需有 request/headers）。"""
    response = SimpleNamespace(status_code=status_code, request=SimpleNamespace(), headers={})
    return cls("http error", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(SimpleNamespace())


def _connection_error() -> APIConnectionError:
    return APIConnectionError(message="connection reset", request=SimpleNamespace())


def _ok_response(content: str = "好", finish_reason: str | None = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(total_tokens=5),
    )


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
            choices=[SimpleNamespace(message=sdk_message, finish_reason="tool_calls")],
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
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="好", tool_calls=None), finish_reason="stop"
                )
            ],
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
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="好", tool_calls=None), finish_reason="stop"
                )
            ],
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


class TestRetry:
    """§5：超时/限流/连接错误按 1/2/4s 退避最多重试 3 次；认证错误快速失败。"""

    def test_rate_limit_retried_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _http_error(RateLimitError, 429)
            return _ok_response()

        response = llm.complete(_fake_client(create), "test-model", [])

        assert response.message.content == "好"
        assert calls == 2
        assert sleeps == [1]

    def test_timeout_exhaustion_gives_friendly_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise _timeout_error()

        with pytest.raises(llm.LLMError, match="重试 3 次") as excinfo:
            llm.complete(_fake_client(create), "test-model", [])

        assert calls == 4  # 首次 + 3 次重试
        assert sleeps == [1, 2, 4]
        assert "最后错误" in str(excinfo.value)

    def test_connection_error_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _connection_error()
            return _ok_response()

        llm.complete(_fake_client(create), "test-model", [])

        assert calls == 2
        assert sleeps == [1]

    def test_auth_401_fails_fast_with_key_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise _http_error(AuthenticationError, 401)

        with pytest.raises(llm.LLMError, match="SE_CodeAgent") as excinfo:
            llm.complete(_fake_client(create), "test-model", [])

        assert calls == 1  # 认证错误绝不重试
        assert sleeps == []
        assert "401" in str(excinfo.value)

    def test_auth_403_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise _http_error(APIStatusError, 403)

        with pytest.raises(llm.LLMError, match="认证失败"):
            llm.complete(_fake_client(create), "test-model", [])

        assert calls == 1

    def test_other_status_errors_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """§5 重试范围仅限超时/限流/连接；5xx 等不重试，包装后原样上抛。"""
        sleeps: list[int] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise _http_error(APIStatusError, 500)

        with pytest.raises(llm.LLMError, match="http error"):
            llm.complete(_fake_client(create), "test-model", [])

        assert calls == 1
        assert sleeps == []

    def test_streaming_creation_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """流式仅建连（create）参与重试；迭代中途断开不重试。"""
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        calls = 0
        chunks = [_content_chunk("正文"), _usage_chunk(5)]

        def create(**kwargs: object) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _http_error(RateLimitError, 429)
            assert kwargs["stream"] is True
            return _fake_stream(chunks)()

        response = llm.complete(_fake_client(create), "test-model", [], stream=True)

        assert response.message.content == "正文"
        assert calls == 2


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


class TestFinishReason:
    """M1 挂账：finish_reason 透传（长度截断的回答不再被静默当最终结论）。"""

    def test_streaming_length_reason_passed_through(self) -> None:
        chunks = [
            _content_chunk("回答的前半"),
            _content_chunk("", finish_reason="length"),
            _usage_chunk(88),
        ]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert response.finish_reason == "length"

    def test_streaming_tool_calls_reason_passed_through(self) -> None:
        chunks = [
            _tool_chunk([_tool_piece(0, call_id="c1", name="list_dir", arguments="{}")],
                        finish_reason="tool_calls"),
            _usage_chunk(9),
        ]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert response.finish_reason == "tool_calls"

    def test_streaming_missing_reason_degrades_to_none(self) -> None:
        chunks = [_content_chunk("正文", finish_reason=None), _usage_chunk(3)]

        response = llm.complete(
            _fake_client(lambda **_: _fake_stream(chunks)()), "test-model", [], stream=True
        )

        assert response.finish_reason is None

    def test_non_streaming_reason_passed_through(self) -> None:
        response = llm.complete(
            _fake_client(lambda **_: _ok_response(finish_reason="length")), "test-model", []
        )

        assert response.finish_reason == "length"


class TestAuthGuidance:
    """M3 挂账：key 指引跨平台化——按当前平台给出对应设置命令。"""

    def test_windows_hint_mentions_setx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.sys, "platform", "win32")
        assert "setx" in llm._key_hint()

    def test_posix_hint_mentions_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.sys, "platform", "linux")
        assert "export" in llm._key_hint()

    def test_auth_error_message_uses_platform_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm.sys, "platform", "linux")
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)

        def create(**_: object) -> SimpleNamespace:
            raise _http_error(AuthenticationError, 401)

        with pytest.raises(llm.LLMError, match="export"):
            llm.complete(_fake_client(create), "test-model", [])


class TestStreamingBadRequest:
    """挂账处置记录：stream_options 在不支持的服务端可能 400——设计决定不做自动
    降级重试（DeepSeek 已实测支持；400 属参数类错误，按 §5 不在重试范围）。"""

    def test_streaming_400_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        calls = 0

        def create(**_: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise _http_error(APIStatusError, 400)

        with pytest.raises(llm.LLMError, match="http error"):
            llm.complete(_fake_client(create), "test-model", [], stream=True)

        assert calls == 1
