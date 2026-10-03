"""llm 层：ChatOpenAI 构建、SDK/框架异常翻译、认证指引跨平台化、非流式通路
（全部离线，无网络；openai 异常用离线构造的实例）。"""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from cra import llm

# 测试占位值(明显假值,非真实凭据);以变量构造避免密钥扫描误报
_EXPLICIT_KEY = "placeholder-key"


def _http_error(cls: type[APIStatusError], status_code: int) -> APIStatusError:
    """构造带 status_code 的 APIStatusError 系异常（response 需有 request/headers）。"""
    response = SimpleNamespace(status_code=status_code, request=SimpleNamespace(), headers={})
    return cls("http error", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(SimpleNamespace())


def _connection_error() -> APIConnectionError:
    return APIConnectionError(message="connection reset", request=SimpleNamespace())


class _FakeModel(BaseChatModel):
    """假模型：invoke 返回固定消息或抛出预置异常（供 invoke_once 测试）。"""

    reply: str = "答案"
    error: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class TestCreateModel:
    def test_reads_env_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setenv("CRA_MODEL", "custom-model")
        monkeypatch.setenv("CRA_BASE_URL", "https://llm.example.com")

        model = llm.create_model()

        assert model.model_name == "custom-model"
        assert model.openai_api_base == "https://llm.example.com"
        assert model.temperature == llm.TEMPERATURE
        assert model.request_timeout == llm.REQUEST_TIMEOUT_SECONDS

    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.delenv("CRA_MODEL", raising=False)
        monkeypatch.delenv("CRA_BASE_URL", raising=False)

        model = llm.create_model()

        assert model.model_name == "deepseek-flash"
        assert model.openai_api_base == "https://api.deepseek.com"

    def test_explicit_model_name_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")
        monkeypatch.setenv("CRA_MODEL", "env-model")

        assert llm.create_model(model_name="override-model").model_name == "override-model"

    def test_missing_api_key_raises_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不依赖 SDK 的 OPENAI_API_KEY 回退（避免把 OpenAI 密钥发往 CRA_BASE_URL）。"""
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SE_CodeAgent", raising=False)

        with pytest.raises(llm.LLMError, match="CRA_DEEPSEEK_API_KEY"):
            llm.create_model()

    def test_d13_retry_mapped_to_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """D13：瞬时错误重试经 SDK 承担（max_retries=3），不在应用层重复实现。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")

        assert llm.create_model().max_retries == llm.MAX_RETRIES

    def test_stream_usage_enabled_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """usage 是截断触发与 /context 的数据源，须显式开启。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-key")

        assert llm.create_model().stream_usage is True


class TestTranslateFailure:
    """SDK/框架异常 → LLMError 的翻译留在 llm 层（agent/cli 不 import SDK 类型）。"""

    def test_llm_error_passthrough(self) -> None:
        original = llm.LLMError("已是 LLMError")

        assert llm.translate_failure(original) is original

    def test_auth_401_gets_key_guidance(self) -> None:
        translated = llm.translate_failure(_http_error(AuthenticationError, 401))

        assert isinstance(translated, llm.LLMError)
        assert "401" in str(translated)
        assert "CRA_DEEPSEEK_API_KEY" in str(translated)

    def test_auth_403_fails_fast_with_guidance(self) -> None:
        translated = llm.translate_failure(_http_error(APIStatusError, 403))

        assert isinstance(translated, llm.LLMError)
        assert "认证失败" in str(translated)

    def test_other_status_errors_wrapped_without_auth_guidance(self) -> None:
        translated = llm.translate_failure(_http_error(APIStatusError, 500))

        assert isinstance(translated, llm.LLMError)
        assert "http error" in str(translated)
        assert "认证失败" not in str(translated)

    def test_timeout_wrapped(self) -> None:
        translated = llm.translate_failure(_timeout_error())

        assert isinstance(translated, llm.LLMError)
        assert translated.__cause__ is not None

    def test_connection_error_wrapped(self) -> None:
        translated = llm.translate_failure(_connection_error())

        assert isinstance(translated, llm.LLMError)
        assert "connection reset" in str(translated)

    def test_generic_openai_error_wrapped(self) -> None:
        translated = llm.translate_failure(APIError("boom", SimpleNamespace(), body=None))

        assert isinstance(translated, llm.LLMError)
        assert "boom" in str(translated)

    def test_recursion_limit_backstop_translated(self) -> None:
        """recursion_limit 护栏触发时给可读提示，而非裸 GraphRecursionError。"""
        translated = llm.translate_failure(GraphRecursionError("recursion limit reached"))

        assert isinstance(translated, llm.LLMError)
        assert "recursion_limit" in str(translated)

    def test_unexpected_exception_not_disguised(self) -> None:
        """非 SDK/框架异常原样返回——意外缺陷不应伪装成请求失败。"""
        original = RuntimeError("代码缺陷")

        assert llm.translate_failure(original) is original


class TestAuthGuidance:
    """key 指引按当前平台给出对应设置命令。"""

    def test_windows_hint_mentions_setx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.sys, "platform", "win32")
        assert "setx" in llm._key_hint()

    def test_posix_hint_mentions_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.sys, "platform", "linux")
        assert "export" in llm._key_hint()

    def test_auth_error_message_uses_platform_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm.sys, "platform", "linux")

        translated = llm.translate_failure(_http_error(AuthenticationError, 401))

        assert isinstance(translated, llm.LLMError)
        assert "export" in str(translated)


class TestInvokeOnce:
    """保留的非流式调用路径(--no-stream 与 cra config test 使用)。"""

    def test_returns_model_response(self) -> None:
        message = llm.invoke_once(_FakeModel(), [HumanMessage(content="问")])

        assert isinstance(message, AIMessage)
        assert message.content == "答案"

    def test_failure_translated_to_llm_error(self) -> None:
        with pytest.raises(llm.LLMError, match="connection reset"):
            llm.invoke_once(_FakeModel(error=_connection_error()), [HumanMessage(content="问")])


class TestCreateModelExplicitParams:
    """cra config test 按供应商显式传 key/接入点;不回退环境变量。"""

    def test_explicit_params_take_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CRA_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SE_CodeAgent", raising=False)

        model = llm.create_model(
            "some-model", api_key=_EXPLICIT_KEY, base_url="https://probe.example.com"
        )

        assert model.model_name == "some-model"
        assert model.openai_api_base == "https://probe.example.com"

    def test_env_fallback_when_explicit_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", _EXPLICIT_KEY)

        assert llm.create_model().openai_api_base == "https://api.deepseek.com"


class _ProbeFakeChatOpenAI:
    """替换真实 ChatOpenAI 的假类:记录构造参数,invoke 按脚本抛异常或返回。"""

    last_instance: "_ProbeFakeChatOpenAI | None" = None
    script: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _ProbeFakeChatOpenAI.last_instance = self

    def bind(self, **_kwargs: Any) -> "_ProbeFakeChatOpenAI":
        return self

    def invoke(self, _messages: Any) -> AIMessage:
        if _ProbeFakeChatOpenAI.script is not None:
            raise _ProbeFakeChatOpenAI.script
        return AIMessage(content="ok")


class TestProbe:
    """cra config test 的连通性自检:状态分类与本层 SDK 异常边界。"""

    @pytest.fixture(autouse=True)
    def _fake_openai(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        _ProbeFakeChatOpenAI.last_instance = None
        _ProbeFakeChatOpenAI.script = None
        monkeypatch.setattr(llm, "ChatOpenAI", _ProbeFakeChatOpenAI)

    def _run(self) -> tuple[str, str]:
        return llm.probe(
            base_url="https://probe.example.com", api_key=_EXPLICIT_KEY, model_name="m"
        )

    def test_ok(self) -> None:
        assert self._run() == ("ok", "")

    def test_empty_key_reported_without_request(self) -> None:
        """空 key 不发到服务端换误导性 401,直接归入 error 并提示。"""
        _ProbeFakeChatOpenAI.last_instance = None

        status, detail = llm.probe(base_url="https://x", api_key="", model_name="m")

        assert status == "error"
        assert "key" in detail
        assert _ProbeFakeChatOpenAI.last_instance is None  # 未发起请求

    def test_probe_uses_short_timeout_and_no_retry(self) -> None:
        """自检语义:短超时、显式 max_retries=0,不承担业务请求的重试。"""
        self._run()

        assert _ProbeFakeChatOpenAI.last_instance is not None
        assert _ProbeFakeChatOpenAI.last_instance.kwargs["timeout"] == llm.PROBE_TIMEOUT_SECONDS
        assert _ProbeFakeChatOpenAI.last_instance.kwargs["max_retries"] == 0

    def test_auth_401(self) -> None:
        _ProbeFakeChatOpenAI.script = _http_error(AuthenticationError, 401)
        status, _ = self._run()
        assert status == "auth"

    def test_auth_403(self) -> None:
        _ProbeFakeChatOpenAI.script = _http_error(APIStatusError, 403)
        status, _ = self._run()
        assert status == "auth"

    def test_timeout(self) -> None:
        _ProbeFakeChatOpenAI.script = _timeout_error()
        status, _ = self._run()
        assert status == "timeout"

    def test_connection_error_reported_as_timeout(self) -> None:
        """网络不可达与超时同属"timeout(网络不可达)"类（三分支报告）。"""
        _ProbeFakeChatOpenAI.script = _connection_error()
        status, _ = self._run()
        assert status == "timeout"

    def test_other_status_error(self) -> None:
        _ProbeFakeChatOpenAI.script = _http_error(APIStatusError, 500)
        status, _ = self._run()
        assert status == "error"

    def test_unexpected_exception_reported_not_raised(self) -> None:
        """自检兜底:任何失败都归入可报告状态而非上抛(检查命令不应崩溃)。"""
        _ProbeFakeChatOpenAI.script = RuntimeError("意外")
        status, detail = self._run()
        assert status == "error"
        assert "意外" in detail
