"""Layer 3 自定义异常层级。"""

from __future__ import annotations


class Layer3Error(Exception):
    """Layer 3 所有自定义异常的基类。"""


class ContextTooLargeError(Layer3Error):
    """上下文截断后仍超过 prompt 预算。"""


class LLMTimeoutError(Layer3Error):
    """LLM 调用超时或重试耗尽。"""


class StructuredOutputValidationError(Layer3Error):
    """LLM structured output 无法解析或不满足接口约束。"""


class CitationValidityError(Layer3Error):
    """运行期引用了不在可引用集合内的 rule_id。"""

    def __init__(
        self,
        message: str,
        *,
        scene_id: str = "",
        stage: str = "",
        cited_ids: set[str] | None = None,
        allowed_ids: set[str] | None = None,
    ) -> None:
        if cited_ids is None:
            cited_ids = set()
        if allowed_ids is None:
            allowed_ids = set()
        super().__init__(message)
        self.scene_id = scene_id
        self.stage = stage
        self.cited_ids = cited_ids
        self.allowed_ids = allowed_ids


class CitationAccuracyEvaluationError(Layer3Error):
    """实验阶段 citation accuracy 评估失败。"""


class JudgeFatalError(Layer3Error):
    """不可恢复错误；Facade 应返回 status=FAILED 的 stub label。"""
