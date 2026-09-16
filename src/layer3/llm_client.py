"""LLM 客户端统一封装。

支持 OpenAI、OpenAI-compatible 和 Anthropic provider，
提供 complete_structured（强制结构化输出）和 complete_text（纯文本输出）两种接口。
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from src.layer3.exceptions import (
    CitationValidityError,
    LLMTimeoutError,
    StructuredOutputValidationError,
)
from src.layer3.models import ConnectionEvent, Layer3Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 所有场景与裁判阶段共享同一个在飞请求上限。
_GLOBAL_LLM_SEMAPHORE = threading.BoundedSemaphore(5)
_COUNTER_LOCK = threading.Lock()
_LLM_429_COUNT = 0


@dataclass(frozen=True)
class LLMUsageSnapshot:
    """单个 LLMClient 的累计调用用量快照。"""

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0

    def delta(self, before: LLMUsageSnapshot) -> LLMUsageSnapshot:
        """计算两个累计快照之间的阶段增量。"""
        return LLMUsageSnapshot(
            call_count=self.call_count - before.call_count,
            input_tokens=self.input_tokens - before.input_tokens,
            output_tokens=self.output_tokens - before.output_tokens,
            retry_count=self.retry_count - before.retry_count,
        )


def get_llm_429_count() -> int:
    """返回当前进程累计收到的 HTTP 429 次数。"""
    with _COUNTER_LOCK:
        return _LLM_429_COUNT


def _record_429(error: Exception) -> None:
    """若异常代表限流响应，则累加计数。"""
    global _LLM_429_COUNT
    if getattr(error, "status_code", None) != 429:
        return
    with _COUNTER_LOCK:
        _LLM_429_COUNT += 1


class LLMClient:
    """LLM API 统一封装。

    支持 OpenAI (json_schema)、OpenAI-compatible (json_object)
    和 Anthropic (tool use) 三种
    structured output 方式。
    """

    def __init__(self, settings: Layer3Settings) -> None:
        self._settings = settings
        self._provider = settings.llm_provider
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._backoff_base = settings.llm_retry_backoff_base_seconds
        self._backoff_multiplier = settings.llm_retry_backoff_multiplier
        self._client: Any = self._build_client()
        self._usage_lock = threading.Lock()
        self._usage = LLMUsageSnapshot()
        self._connection_events: list[ConnectionEvent] = []

    def usage_snapshot(self) -> LLMUsageSnapshot:
        """返回线程安全的累计用量快照。"""
        with self._usage_lock:
            return self._usage

    def connection_events(self) -> list[ConnectionEvent]:
        """返回本 client 会话内记录的传输异常事件。"""
        with self._usage_lock:
            return list(self._connection_events)

    def note_retry(self) -> None:
        """记录一次 API 或上层校验重试。"""
        with self._usage_lock:
            self._usage = LLMUsageSnapshot(
                call_count=self._usage.call_count,
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
                retry_count=self._usage.retry_count + 1,
            )

    def _note_api_call(self) -> None:
        """记录一次实际 API 请求。"""
        with self._usage_lock:
            self._usage = LLMUsageSnapshot(
                call_count=self._usage.call_count + 1,
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
                retry_count=self._usage.retry_count,
            )

    def _note_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """累加一次成功响应的输入与输出 token。"""
        with self._usage_lock:
            self._usage = LLMUsageSnapshot(
                call_count=self._usage.call_count,
                input_tokens=self._usage.input_tokens + input_tokens,
                output_tokens=self._usage.output_tokens + output_tokens,
                retry_count=self._usage.retry_count,
            )

    def _note_connection_event(
        self,
        error: Exception,
        *,
        operation: Literal["structured", "text"],
        attempt: int,
    ) -> None:
        """记录发生时刻，避免用场景完成时间代替传输异常时间。"""
        detail = f"{type(error).__name__}: {error}"
        kind: Literal["timeout", "connection_error"] = (
            "timeout" if "timeout" in detail.lower() else "connection_error"
        )
        with self._usage_lock:
            self._connection_events.append(
                ConnectionEvent(
                    occurred_at=datetime.now(UTC),
                    kind=kind,
                    operation=operation,
                    attempt=attempt,
                    message=detail,
                )
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> tuple[T, int]:
        """强制 LLM 输出为 response_model 指定的 Pydantic 类型。

        Returns:
            (parsed, tokens_used)
        """
        temp = temperature if temperature is not None else 0.0
        token_limit = max_tokens or self._settings.llm_structured_max_tokens
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout
        retry_limit = max_retries if max_retries is not None else self._max_retries
        if timeout <= 0 or retry_limit < 0:
            raise ValueError("timeout_seconds must be > 0 and max_retries must be >= 0")
        last_error: Exception | None = None

        for attempt in range(retry_limit + 1):
            try:
                if self._provider in {"openai", "openai_compatible"}:
                    parsed, tokens = self._complete_structured_openai(
                        system_prompt, user_prompt, response_model,
                        temperature=temp, max_tokens=token_limit, timeout_seconds=timeout,
                    )
                else:
                    parsed, tokens = self._complete_structured_anthropic(
                        system_prompt, user_prompt, response_model,
                        temperature=temp, max_tokens=token_limit, timeout_seconds=timeout,
                    )
                logger.info(
                    "LLM structured call succeeded: model=%s attempt=%d tokens=%d",
                    self._model, attempt + 1, tokens,
                )
                return parsed, tokens
            except (StructuredOutputValidationError, CitationValidityError) as e:
                last_error = e
                if attempt < retry_limit:
                    self.note_retry()
                    logger.warning(
                        "LLM structured call failed (attempt %d/%d): %s",
                        attempt + 1, retry_limit + 1, e,
                    )
                    # 把失败信息追加到 user_prompt 让 LLM 自我修正
                    user_prompt = self._augment_with_error(user_prompt, e)
                else:
                    logger.error(
                        "LLM structured call exhausted retries: %s", e,
                    )
            except Exception as e:
                _record_429(e)
                self._note_connection_event(
                    e,
                    operation="structured",
                    attempt=attempt + 1,
                )
                last_error = e
                if attempt < retry_limit:
                    self.note_retry()
                    logger.warning(
                        "LLM call transient error (attempt %d/%d): %s",
                        attempt + 1, retry_limit + 1, e,
                    )
                    time.sleep(
                        random.uniform(self._backoff_base, 2.0 * self._backoff_base)
                        * (self._backoff_multiplier ** attempt)
                    )
                else:
                    logger.error("LLM call exhausted retries: %s", e)

        if isinstance(last_error, (StructuredOutputValidationError, CitationValidityError)):
            raise last_error
        raise LLMTimeoutError(
            f"LLM call failed after {retry_limit + 1} attempts: {last_error}"
        )

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, int]:
        """纯文本 LLM 调用。

        Returns:
            (text, tokens_used)
        """
        temp = temperature if temperature is not None else 0.0
        token_limit = max_tokens or self._settings.llm_text_max_tokens

        for attempt in range(self._max_retries + 1):
            try:
                if self._provider in {"openai", "openai_compatible"}:
                    text, tokens = self._complete_text_openai(
                        system_prompt, user_prompt,
                        temperature=temp, max_tokens=token_limit,
                    )
                else:
                    text, tokens = self._complete_text_anthropic(
                        system_prompt, user_prompt,
                        temperature=temp, max_tokens=token_limit,
                    )
                logger.info(
                    "LLM text call succeeded: model=%s tokens=%d",
                    self._model, tokens,
                )
                return text, tokens
            except Exception as e:
                _record_429(e)
                self._note_connection_event(
                    e,
                    operation="text",
                    attempt=attempt + 1,
                )
                if attempt < self._max_retries:
                    self.note_retry()
                    logger.warning(
                        "LLM text call error (attempt %d/%d): %s",
                        attempt + 1, self._max_retries + 1, e,
                    )
                    time.sleep(
                        random.uniform(self._backoff_base, 2.0 * self._backoff_base)
                        * (self._backoff_multiplier ** attempt)
                    )
                else:
                    raise LLMTimeoutError(
                        f"LLM text call failed after {self._max_retries + 1} attempts: {e}"
                    ) from e
        raise LLMTimeoutError(
            f"LLM text call failed after {self._max_retries + 1} attempts"
        )

    # ------------------------------------------------------------------
    # 内部：客户端构建
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        """根据 provider 构建对应的客户端实例。"""
        key = self._settings.llm_api_key.get_secret_value()
        # SDK 在构造客户端时就要求非空密钥，但很多离线测试只构造裁判，
        # 并不会发起网络请求。使用明确无效的占位值可保留这种离线能力；
        # 真实运行仍必须通过 LAYER3_LLM_API_KEY 提供有效密钥。
        client_key = key or "not-configured"

        if self._provider in {"openai", "openai_compatible"}:
            from openai import OpenAI
            kwargs: dict[str, Any] = {
                "api_key": client_key,
                "timeout": self._timeout,
                "max_retries": 0,
            }
            if self._settings.llm_base_url:
                kwargs["base_url"] = self._settings.llm_base_url
            return OpenAI(**kwargs)
        else:
            from anthropic import Anthropic
            return Anthropic(api_key=client_key, timeout=self._timeout)

    # ------------------------------------------------------------------
    # 内部：OpenAI structured output
    # ------------------------------------------------------------------

    def _complete_structured_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> tuple[T, int]:
        client = self._client

        messages = self._openai_messages(system_prompt, user_prompt)
        extra_kwargs = self._openai_extra_kwargs()

        with _GLOBAL_LLM_SEMAPHORE:
            self._note_api_call()
            response = client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                response_format=self._openai_structured_response_format(response_model),
                timeout=timeout_seconds,
                **extra_kwargs,
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None)
        input_tokens, output_tokens = self._openai_usage(response.usage)
        self._note_tokens(input_tokens, output_tokens)
        tokens = input_tokens + output_tokens

        try:
            data = json.loads(content)
            parsed = response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise StructuredOutputValidationError(
                f"Failed to parse structured output: {e}; "
                f"finish_reason={finish_reason!r}; "
                f"reasoning_chars={len(reasoning_content)}\nRaw: {content[:500]}"
            ) from e

        return parsed, tokens

    def _openai_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        """构造 OpenAI Chat Completions 消息。"""
        if self._provider != "openai_compatible":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        combined = f"{system_prompt}\n{user_prompt}".lower()
        if "json" in combined:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nOutput must be valid JSON.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _openai_extra_kwargs(self) -> dict[str, Any]:
        if (
            self._provider == "openai_compatible"
            and self._settings.llm_base_url
            and "api.deepseek.com" in self._settings.llm_base_url
        ):
            return {
                "extra_body": {
                    "thinking": {"type": self._settings.llm_thinking_mode},
                },
            }
        return {}

    def _openai_structured_response_format(
        self,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """返回当前 OpenAI 风格 provider 支持的结构化输出格式。"""
        if self._provider == "openai_compatible":
            return {"type": "json_object"}

        json_schema = response_model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": json_schema,
            },
        }

    # ------------------------------------------------------------------
    # 内部：Anthropic structured output (via tool use)
    # ------------------------------------------------------------------

    def _complete_structured_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> tuple[T, int]:
        client = self._client

        json_schema = response_model.model_json_schema()

        # Anthropic tool use 包装
        tool_def = {
            "name": f"output_{response_model.__name__}",
            "description": f"Structured output conforming to {response_model.__name__}",
            "input_schema": json_schema,
        }

        with _GLOBAL_LLM_SEMAPHORE:
            self._note_api_call()
            response = client.messages.create(
                model=self._model,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[tool_def],
                tool_choice={"type": "tool", "name": tool_def["name"]},
                timeout=timeout_seconds,
            )

        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        self._note_tokens(input_tokens, output_tokens)
        tokens = input_tokens + output_tokens

        # 提取 tool_use 块中的 JSON
        raw_json: dict[str, Any] | None = None
        for block in response.content:
            if block.type == "tool_use":
                raw_json = block.input
                break

        if raw_json is None:
            raise StructuredOutputValidationError(
                "Anthropic response did not contain tool_use block"
            )

        try:
            parsed = response_model.model_validate(raw_json)
        except ValueError as e:
            raise StructuredOutputValidationError(
                f"Failed to validate structured output: {e}\nData: {json.dumps(raw_json)[:500]}"
            ) from e

        return parsed, tokens

    # ------------------------------------------------------------------
    # 内部：OpenAI text
    # ------------------------------------------------------------------

    def _complete_text_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        client = self._client
        extra_kwargs = self._openai_extra_kwargs()

        with _GLOBAL_LLM_SEMAPHORE:
            self._note_api_call()
            response = client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **extra_kwargs,
            )

        content = response.choices[0].message.content or ""
        input_tokens, output_tokens = self._openai_usage(response.usage)
        self._note_tokens(input_tokens, output_tokens)
        tokens = input_tokens + output_tokens
        return content, tokens

    # ------------------------------------------------------------------
    # 内部：Anthropic text
    # ------------------------------------------------------------------

    def _complete_text_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        client = self._client

        with _GLOBAL_LLM_SEMAPHORE:
            self._note_api_call()
            response = client.messages.create(
                model=self._model,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_prompt}],
            )

        # Anthropic 返回 text blocks
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        text = "".join(text_parts)
        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        self._note_tokens(input_tokens, output_tokens)
        tokens = input_tokens + output_tokens
        return text, tokens

    @staticmethod
    def _openai_usage(usage: Any) -> tuple[int, int]:
        """兼容读取 OpenAI 风格响应的输入与输出 token。"""
        if usage is None:
            return 0, 0
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion
        total = getattr(usage, "total_tokens", 0)
        return (int(total), 0) if isinstance(total, int) else (0, 0)

    # ------------------------------------------------------------------
    # 内部：错误增强
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_with_error(user_prompt: str, error: Exception) -> str:
        """将上一次错误信息追加到 prompt 中，帮助 LLM 自我修正。"""
        return (
            f"{user_prompt}\n\n"
            f"[系统提示] 上一次输出校验失败：{error}\n"
            f"请修正上述错误，确保输出严格符合要求的格式和约束。"
        )
