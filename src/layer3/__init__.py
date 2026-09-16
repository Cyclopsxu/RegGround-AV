"""Layer 3：逻辑裁判与 pairwise 偏好打标层。"""

from src.layer1.models import JudgeInput
from src.layer2.models import RuleSubgraph
from src.layer3.judge import AuditJudge
from src.layer3.models import (
    AuditLabel,
    CompletionSource,
    ConnectionEvent,
    DecisionSource,
    DeferredAttempt,
    JudgeContext,
    JudgeStage,
    Layer3Settings,
    PairwisePreference,
    ReasoningStep,
    ReportStatus,
    StageDiagnostics,
    TrajectoryVerdict,
    VetoResult,
    VetoStatus,
)


def judge(
    judge_input: JudgeInput,
    subgraph: RuleSubgraph,
    *,
    settings: Layer3Settings | None = None,
    judger: AuditJudge | None = None,
) -> AuditLabel:
    """便捷函数：单次调用完成 Layer 3 全流程评判。"""
    if judger is None:
        if settings is None:
            settings = Layer3Settings()
        judger = AuditJudge(settings)
    return judger.judge(judge_input, subgraph)


__all__ = [
    "AuditJudge",
    "AuditLabel",
    "CompletionSource",
    "ConnectionEvent",
    "DecisionSource",
    "DeferredAttempt",
    "JudgeContext",
    "JudgeInput",
    "JudgeStage",
    "Layer3Settings",
    "PairwisePreference",
    "ReasoningStep",
    "ReportStatus",
    "StageDiagnostics",
    "TrajectoryVerdict",
    "VetoResult",
    "VetoStatus",
    "judge",
]
