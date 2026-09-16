"""test_llm_client.py —— LLM 客户端独立测试（P2-2）。

覆盖：complete_structured / complete_text 的成功、
JSON 解析失败重试、structured output 校验、超时转换 LLMTimeoutError。
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.layer3.exceptions import (
    LLMTimeoutError,
    StructuredOutputValidationError,
)
from src.layer3.llm_client import LLMClient
from src.layer3.models import Layer3Settings

# ── 小型测试模型 ────────────────────────────────────────────


class _TestModel(BaseModel):
    name: str
    value: int = Field(ge=0)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def settings() -> Layer3Settings:
    return Layer3Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        llm_api_key="test-key",  # type: ignore[arg-type]
        llm_timeout_seconds=5.0,
        llm_max_retries=2,
        llm_retry_backoff_base_seconds=0.0,
    )


def _make_anthropic_mock(tool_input: dict | None = None, text: str = ""):
    """创建模拟的 Anthropic client。"""
    client = MagicMock()
    content_blocks: list = []
    if tool_input is not None:
        content_blocks.append(MagicMock(type="tool_use", input=tool_input))
    if text:
        content_blocks.append(MagicMock(type="text", text=text))
    if not content_blocks:
        # 至少一个 text block
        content_blocks.append(MagicMock(type="text", text=""))
    response = MagicMock()
    response.content = content_blocks
    response.usage = MagicMock(input_tokens=10, output_tokens=20)
    client.messages.create.return_value = response
    return client


def _make_openai_mock(
    content: str = '{"name":"t","value":42}',
    *,
    finish_reason: str = "stop",
    reasoning_content: str = "",
):
    """创建模拟的 OpenAI client。"""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(
        finish_reason=finish_reason,
        message=MagicMock(content=content, reasoning_content=reasoning_content),
    )]
    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )
    client.chat.completions.create.return_value = response
    return client


# ── Tests: Anthropic structured output ───────────────────────


class TestAnthropicStructured:
    def test_success(self, settings):
        mock_client = _make_anthropic_mock(tool_input={"name": "test", "value": 42})
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            parsed, tokens = client.complete_structured(
                "system", "user", _TestModel,
            )
            assert parsed.name == "test"
            assert parsed.value == 42
            assert tokens == 30

    def test_validation_failure_retries(self, settings):
        mock_client = _make_anthropic_mock(tool_input={"value": 42})
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            with pytest.raises(StructuredOutputValidationError):
                client.complete_structured("system", "user", _TestModel)

    def test_no_tool_use_block(self, settings):
        mock_client = _make_anthropic_mock(text="plain text response")
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            with pytest.raises(StructuredOutputValidationError):
                client.complete_structured("system", "user", _TestModel)


# ── Tests: OpenAI structured output ──────────────────────────


class TestOpenAIStructured:
    @pytest.fixture(autouse=True)
    def _use_openai(self, settings):
        self.settings = settings.model_copy(update={"llm_provider": "openai"})

    def test_success(self):
        mock_client = _make_openai_mock()
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(self.settings)
            parsed, tokens = client.complete_structured(
                "system", "user", _TestModel,
            )
            assert parsed.name == "t"
            assert parsed.value == 42
            assert tokens == 30
            usage = client.usage_snapshot()
            assert usage.call_count == 1
            assert usage.input_tokens == 20
            assert usage.output_tokens == 10

    def test_invalid_json_retries(self):
        mock_client = _make_openai_mock(content="not valid json {{{")
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(self.settings)
            with pytest.raises(StructuredOutputValidationError):
                client.complete_structured("system", "user", _TestModel)
            usage = client.usage_snapshot()
            assert usage.call_count == 3
            assert usage.retry_count == 2


# ── Tests: OpenAI-compatible providers ──────────────────────


class TestOpenAICompatible:
    def test_client_uses_base_url(self, settings):
        compatible_settings = settings.model_copy(
            update={
                "llm_provider": "openai_compatible",
                "llm_model": "deepseek-v4-pro",
                "llm_base_url": "https://api.deepseek.com",
            }
        )
        with patch("openai.OpenAI") as openai_cls:
            LLMClient(compatible_settings)
            openai_cls.assert_called_once_with(
                api_key="test-key",
                timeout=5.0,
                max_retries=0,
                base_url="https://api.deepseek.com",
            )

    def test_structured_uses_json_object(self, settings):
        compatible_settings = settings.model_copy(
            update={
                "llm_provider": "openai_compatible",
                "llm_model": "deepseek-v4-pro",
                "llm_base_url": "https://api.deepseek.com",
            }
        )
        mock_client = _make_openai_mock()
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(compatible_settings)
            parsed, tokens = client.complete_structured(
                "system", "return json", _TestModel,
            )
            assert parsed.name == "t"
            assert parsed.value == 42
            assert tokens == 30

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["max_tokens"] == 8_192
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_structured_error_reports_reasoning_truncation(self, settings):
        compatible_settings = settings.model_copy(
            update={
                "llm_provider": "openai_compatible",
                "llm_model": "deepseek-v4-pro",
                "llm_base_url": "https://api.deepseek.com",
                "llm_max_retries": 0,
            }
        )
        mock_client = _make_openai_mock(
            content="",
            finish_reason="length",
            reasoning_content="hidden reasoning",
        )
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(compatible_settings)
            with pytest.raises(
                StructuredOutputValidationError,
                match=r"finish_reason='length'; reasoning_chars=16",
            ):
                client.complete_structured("system", "return json", _TestModel)

    def test_structured_call_can_override_timeout_and_retry_budget(self, settings):
        mock_client = _make_openai_mock()
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings.model_copy(update={"llm_provider": "openai"}))
            client.complete_structured(
                "system",
                "user",
                _TestModel,
                timeout_seconds=90.0,
                max_retries=2,
            )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["timeout"] == 90.0


# ── Tests: complete_text ─────────────────────────────────────


class TestCompleteText:
    def test_anthropic_text_success(self, settings):
        mock_client = _make_anthropic_mock(text="Hello world")
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            text, tokens = client.complete_text("system", "user")
            assert text == "Hello world"
            assert tokens == 30

    def test_openai_text_success(self, settings):
        settings = settings.model_copy(update={"llm_provider": "openai"})
        mock_client = _make_openai_mock(content="Hello world")
        mock_client.messages = None  # ensure it's not called
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            text, tokens = client.complete_text("system", "user")
            assert text == "Hello world"
            assert tokens == 30


# ── Tests: SDK exceptions → LLMTimeoutError ──────────────────


class TestErrorTransformation:
    def test_sdk_exception_becomes_timeout(self, settings):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("sdk timeout")
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            with pytest.raises(LLMTimeoutError):
                client.complete_text("system", "user")

    def test_structured_error_becomes_timeout(self, settings):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("timeout")
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(settings)
            with pytest.raises(LLMTimeoutError):
                client.complete_structured("system", "user", _TestModel)

    def test_timeout_records_connection_event_at_occurrence(self, settings):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("network timeout")
        no_retry = settings.model_copy(update={"llm_max_retries": 0})
        with patch.object(LLMClient, "_build_client", return_value=mock_client):
            client = LLMClient(no_retry)
            with pytest.raises(LLMTimeoutError):
                client.complete_structured("system", "user", _TestModel)

        events = client.connection_events()
        assert len(events) == 1
        assert events[0].kind == "timeout"
        assert events[0].operation == "structured"
        assert events[0].attempt == 1
