"""Layer 3 数据模型定义。"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.layer1.models import JudgeInput, SceneFactsDigest, TrajectoryFeatures


class VetoStatus(StrEnum):
    VETOED = "vetoed"
    CLEARED = "cleared"
    UNCERTAIN = "uncertain"


class JudgeStage(StrEnum):
    CONTEXT_BUILDING = "context_building"
    HARD_FILTER = "hard_filter"
    PAIRWISE_RANKING = "pairwise_ranking"
    LABEL_GENERATION = "label_generation"


class ReportStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class SelectionOutcome(StrEnum):
    """候选选择结果，与报告流程状态相互独立。"""

    SELECTED = "selected"
    NO_CHOOSABLE_CANDIDATE = "no_choosable_candidate"
    UNAVAILABLE = "unavailable"


class PartialReason(StrEnum):
    """报告降级原因；status 必须由本列表推导。"""

    UNCERTAIN_VERDICT = "uncertain_verdict"
    LOW_CONFIDENCE_PAIR = "low_confidence_pair"
    CITATION_REPAIR = "citation_repair"
    STAGE_DEGRADED = "stage_degraded"
    FATAL_ERROR = "fatal_error"


class DecisionSource(StrEnum):
    RULE_ENGINE = "rule_engine"
    LLM = "llm"
    FALLBACK = "fallback"


class CompletionSource(StrEnum):
    """标记判决在哪个运行阶段完成，独立于判决来源。"""

    PRIMARY = "primary"
    DEFERRED_RETRY = "deferred_retry"
    FALLBACK = "fallback"


class Layer3Settings(BaseSettings):
    """Layer 3 配置，支持环境变量 LAYER3_ 前缀覆盖。"""

    llm_provider: Literal["openai", "anthropic", "openai_compatible"] = "openai_compatible"
    llm_model: str = "deepseek-v4-pro"
    llm_model_version: str = "DeepSeek-V4-Pro"
    llm_thinking_mode: Literal["enabled", "disabled"] = "enabled"
    # 真实密钥只能通过 LAYER3_LLM_API_KEY 环境变量传入，禁止写进源码。
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str | None = "https://api.deepseek.com"
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 1
    llm_retry_backoff_base_seconds: float = Field(default=3.0, ge=0.0)
    llm_retry_backoff_multiplier: float = Field(default=4.0, ge=1.0)
    llm_structured_max_tokens: int = Field(default=8_192, gt=0)
    llm_text_max_tokens: int = Field(default=4_096, gt=0)
    # hard-filter 的单次逻辑调用网络等待预算为
    # (hard_filter_max_retries + 1) * hard_filter_timeout_seconds；
    # 重试退避的少量额外等待不计入该网络超时预算。
    hard_filter_timeout_seconds: float = Field(default=180.0, gt=0.0)
    hard_filter_max_retries: int = Field(default=2, ge=0)
    deferred_retry_rounds: int = Field(default=3, ge=1)
    deferred_retry_delay_seconds: float = Field(default=5.0, ge=0.0)
    llm_input_price_per_million_usd: float | None = Field(default=None, ge=0.0)
    llm_output_price_per_million_usd: float | None = Field(default=None, ge=0.0)

    hard_filter_temperature: float = 0.0
    pairwise_ranker_temperature: float = 0.0
    label_temperature: float = 0.2

    max_trajectories_per_judge: int = 10
    max_rules_in_context: int = 20
    max_pairwise_comparisons: int = 45
    max_prompt_tokens: int = 8_000
    phase_validity_horizon_s: float = Field(default=2.0, gt=0.0)
    enable_cot: bool = True
    archive_llm_io: bool = False
    # 仅影响 hard filter 阶段：置真时规则引擎不再直裁，全部候选交给 LLM。
    # 影子评估与 2×2 消融的 no_rule_engine 条件共用这一开关。
    disable_rule_engine: bool = False

    model_config = SettingsConfigDict(env_prefix="LAYER3_")


class JudgeContext(BaseModel):
    """Prompt 构建后的 Layer 3 内部上下文。"""

    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    scene_narrative: str
    scene_facts_digest: SceneFactsDigest
    scene_facts_text: str
    hard_rules_text: str
    soft_rules_text: str
    trajectories_text: str
    hard_rule_ids: list[str]
    soft_rule_ids: list[str]
    hard_rule_categories: list[str] = Field(default_factory=list)
    soft_rule_categories: list[str] = Field(default_factory=list)
    trajectory_ids: list[str]
    total_prompt_tokens: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_rule_ids(self) -> list[str]:
        return sorted(set(self.hard_rule_ids + self.soft_rule_ids))


class DeferredAttempt(BaseModel):
    """候选级清扫轮的一次尝试，供运行后审计。"""

    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    round_index: int = Field(ge=1)
    started_at: datetime
    finished_at: datetime
    error_message: str | None = None


class VetoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    status: VetoStatus
    violated_rule_ids: list[str] = Field(default_factory=list)
    reason: str
    decided_by: DecisionSource
    completed_by: CompletionSource = CompletionSource.PRIMARY
    deferred_attempts: list[DeferredAttempt] = Field(default_factory=list)

    @field_validator("violated_rule_ids")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate rule_ids in violated_rule_ids")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> VetoResult:
        if self.status == VetoStatus.VETOED and not self.violated_rule_ids:
            raise ValueError("vetoed trajectory must cite at least one hard rule")
        if self.status in {VetoStatus.CLEARED, VetoStatus.UNCERTAIN} and self.violated_rule_ids:
            raise ValueError("cleared/uncertain trajectory must not cite violated rules")
        return self


class PairwisePreference(BaseModel):
    model_config = ConfigDict(frozen=True)

    preferred_id: str
    dispreferred_id: str
    basis: Literal["compliance", "pairwise_judgment"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    referenced_rule_ids: list[str] = Field(default_factory=list)
    decided_by: DecisionSource

    @field_validator("referenced_rule_ids")
    @classmethod
    def _unique_refs(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate rule_ids in referenced_rule_ids")
        return v

    @model_validator(mode="after")
    def _ids_are_distinct(self) -> PairwisePreference:
        if self.preferred_id == self.dispreferred_id:
            raise ValueError("preferred_id and dispreferred_id must differ")
        return self


class ReasoningStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: JudgeStage
    content: str
    referenced_rule_ids: list[str] = Field(default_factory=list)

    @field_validator("referenced_rule_ids")
    @classmethod
    def _unique_refs(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate rule_ids in referenced_rule_ids")
        return v


class TrajectoryVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    veto: VetoResult
    rank: int | None = None

    @model_validator(mode="after")
    def _consistency(self) -> TrajectoryVerdict:
        if self.veto.status == VetoStatus.VETOED and self.rank is not None:
            raise ValueError("vetoed trajectory must not have rank among cleared items")
        if self.veto.status == VetoStatus.UNCERTAIN and self.rank is not None:
            raise ValueError("uncertain trajectory must not have pairwise rank")
        return self


class StageDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: JudgeStage
    duration_ms: int
    llm_tokens_used: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_call_count: int = 0
    llm_retry_count: int = 0
    planned_pairwise_comparisons: int = Field(default=0, ge=0)
    completed_pairwise_comparisons: int = Field(default=0, ge=0)
    deferred_attempts: list[DeferredAttempt] = Field(default_factory=list)
    recovered_count: int = Field(default=0, ge=0)
    feature_text_hashes: dict[str, str] = Field(default_factory=dict)
    citation_failures: list[CitationRepairDetail] = Field(default_factory=list)
    succeeded: bool
    error_message: str | None = None


class CitationRepairDetail(BaseModel):
    """引用校验失败及回退修复明细。"""

    model_config = ConfigDict(frozen=True)

    stage: JudgeStage
    raw_citations: list[str]
    fallback_citations: list[str]
    allowed_citations: list[str]
    error_message: str


class ConnectionEvent(BaseModel):
    """LLM 传输层异常的发生时间，避免事后用记录完成时间猜测。"""

    model_config = ConfigDict(frozen=True)

    occurred_at: datetime
    kind: Literal["timeout", "connection_error"]
    operation: Literal["structured", "text"]
    attempt: int = Field(ge=1)
    message: str


class AuditLabel(BaseModel):
    """Layer 3 最终可审计偏好标签。"""

    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    partial_reasons: list[PartialReason] = Field(default_factory=list)
    chosen_trajectory_id: str | None
    preference_ranking: list[str]
    preference_pairs: list[PairwisePreference]
    verdicts: list[TrajectoryVerdict]
    natural_language_summary: str
    legal_basis: list[str]
    reasoning_chain: list[ReasoningStep]
    stage_diagnostics: list[StageDiagnostics]
    connection_events: list[ConnectionEvent] = Field(default_factory=list)
    citation_validity: float = Field(ge=0.0, le=1.0)
    total_duration_ms: int
    tied_pairs_count: int = Field(default=0, ge=0)
    generated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> ReportStatus:
        """由降级原因唯一推导报告状态。"""
        if PartialReason.FATAL_ERROR in self.partial_reasons:
            return ReportStatus.FAILED
        if self.partial_reasons:
            return ReportStatus.PARTIAL
        return ReportStatus.COMPLETE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def selection_outcome(self) -> SelectionOutcome:
        """根据有效裁决与 cleared 候选计算选择结果。"""
        if self.status == ReportStatus.FAILED or not self.verdicts:
            return SelectionOutcome.UNAVAILABLE
        if self.cleared_count == 0:
            return SelectionOutcome.NO_CHOOSABLE_CANDIDATE
        return SelectionOutcome.SELECTED

    @field_validator("partial_reasons")
    @classmethod
    def _unique_partial_reasons(cls, v: list[PartialReason]) -> list[PartialReason]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate partial_reasons")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vetoed_count(self) -> int:
        return sum(1 for v in self.verdicts if v.veto.status == VetoStatus.VETOED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cleared_count(self) -> int:
        return sum(1 for v in self.verdicts if v.veto.status == VetoStatus.CLEARED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uncertain_count(self) -> int:
        return sum(1 for v in self.verdicts if v.veto.status == VetoStatus.UNCERTAIN)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decided_by_rule_engine_count(self) -> int:
        return sum(
            1 for v in self.verdicts if v.veto.decided_by == DecisionSource.RULE_ENGINE
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decided_by_llm_count(self) -> int:
        return sum(1 for v in self.verdicts if v.veto.decided_by == DecisionSource.LLM)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decided_by_fallback_count(self) -> int:
        return sum(1 for v in self.verdicts if v.veto.decided_by == DecisionSource.FALLBACK)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pairwise_comparisons_count(self) -> int:
        return sum(1 for p in self.preference_pairs if p.basis == "pairwise_judgment")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chosen_position(self) -> int | None:
        if self.chosen_trajectory_id is None:
            return None
        try:
            return self.preference_ranking.index(self.chosen_trajectory_id)
        except ValueError:
            return None

    @field_validator("legal_basis")
    @classmethod
    def _dedupe_and_sort(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _self_consistency(self) -> AuditLabel:
        verdict_ids = {v.trajectory_id for v in self.verdicts}

        if self.cleared_count > 0 and self.chosen_trajectory_id is None:
            raise ValueError("a cleared trajectory requires chosen_trajectory_id")
        if self.cleared_count == 0 and self.chosen_trajectory_id is not None:
            raise ValueError("chosen_trajectory_id requires a cleared trajectory")

        if self.chosen_trajectory_id is not None:
            if self.chosen_trajectory_id not in verdict_ids:
                raise ValueError("chosen_trajectory_id must appear in verdicts")
            chosen = next(v for v in self.verdicts if v.trajectory_id == self.chosen_trajectory_id)
            if chosen.veto.status != VetoStatus.CLEARED:
                raise ValueError("chosen_trajectory_id must refer to a cleared trajectory")

        if set(self.preference_ranking) != verdict_ids:
            raise ValueError("preference_ranking must contain exactly the verdict ids")

        status_order = {
            VetoStatus.CLEARED: 0,
            VetoStatus.UNCERTAIN: 1,
            VetoStatus.VETOED: 2,
        }
        verdict_by_id = {v.trajectory_id: v for v in self.verdicts}
        ranking_layers = [
            status_order[verdict_by_id[trajectory_id].veto.status]
            for trajectory_id in self.preference_ranking
        ]
        if ranking_layers != sorted(ranking_layers):
            raise ValueError(
                "preference_ranking must order cleared before uncertain before vetoed"
            )

        uncertain_ids = {
            v.trajectory_id for v in self.verdicts if v.veto.status == VetoStatus.UNCERTAIN
        }
        for pair in self.preference_pairs:
            if pair.preferred_id not in verdict_ids:
                raise ValueError(f"preferred_id={pair.preferred_id!r} not in verdicts")
            if pair.dispreferred_id not in verdict_ids:
                raise ValueError(f"dispreferred_id={pair.dispreferred_id!r} not in verdicts")
            if pair.preferred_id in uncertain_ids or pair.dispreferred_id in uncertain_ids:
                raise ValueError("uncertain trajectories must not appear in preference_pairs")

        if PartialReason.STAGE_DEGRADED not in self.partial_reasons:
            cleared_count = self.cleared_count
            vetoed_count = self.vetoed_count
            pairwise_count = sum(
                pair.basis == "pairwise_judgment" for pair in self.preference_pairs
            )
            compliance_count = sum(
                pair.basis == "compliance" for pair in self.preference_pairs
            )
            if pairwise_count != cleared_count * (cleared_count - 1) // 2:
                raise ValueError("pairwise count must equal C(cleared, 2)")
            if compliance_count != cleared_count * vetoed_count:
                raise ValueError("compliance pair count must equal cleared * vetoed")

        cited = self._collect_cited_rule_ids()
        uncovered = cited - set(self.legal_basis)
        if uncovered:
            raise ValueError(f"legal_basis does not cover cited rule_ids: {uncovered}")

        summary_refs = set(re.findall(r"\[(R-[A-Za-z]+-\d+)\]", self.natural_language_summary))
        unknown_refs = summary_refs - set(self.legal_basis)
        if unknown_refs:
            raise ValueError(
                f"natural_language_summary references ids outside legal_basis: {unknown_refs}"
            )

        return self

    def _collect_cited_rule_ids(self) -> set[str]:
        cited: set[str] = set()
        for verdict in self.verdicts:
            cited.update(verdict.veto.violated_rule_ids)
        for pair in self.preference_pairs:
            cited.update(pair.referenced_rule_ids)
        for step in self.reasoning_chain:
            cited.update(step.referenced_rule_ids)
        cited.update(re.findall(r"\[(R-[A-Za-z]+-\d+)\]", self.natural_language_summary))
        return cited


__all__ = [
    "AuditLabel",
    "CitationRepairDetail",
    "CompletionSource",
    "ConnectionEvent",
    "DecisionSource",
    "DeferredAttempt",
    "JudgeContext",
    "JudgeInput",
    "JudgeStage",
    "Layer3Settings",
    "PairwisePreference",
    "PartialReason",
    "ReasoningStep",
    "ReportStatus",
    "SelectionOutcome",
    "StageDiagnostics",
    "TrajectoryFeatures",
    "TrajectoryVerdict",
    "VetoResult",
    "VetoStatus",
]
