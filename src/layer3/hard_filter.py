"""硬性过滤模块。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.layer1.models import TrajectoryFeatures
from src.layer3.citation_validator import CitationValidator
from src.layer3.exceptions import (
    CitationValidityError,
    LLMTimeoutError,
    StructuredOutputValidationError,
)
from src.layer3.leakage_guard import LeakageGuard
from src.layer3.llm_client import LLMClient
from src.layer3.models import (
    CitationRepairDetail,
    CompletionSource,
    DecisionSource,
    DeferredAttempt,
    JudgeContext,
    JudgeStage,
    Layer3Settings,
    ReasoningStep,
    VetoResult,
    VetoStatus,
)
from src.layer3.prompt_loader import load_few_shot_examples, load_system_prompt
from src.layer3.rule_engine import DisabledRuleEngine, RuleEngine, RuleEngineDecision
from src.trajectory_feature_text import feature_text_hash, render_feature_text

logger = logging.getLogger(__name__)


class _FilterItem(BaseModel):
    trajectory_id: str
    status: VetoStatus
    violated_rule_ids: list[str] = Field(default_factory=list)
    reason: str


class _LLMFilterResponse(BaseModel):
    results: list[_FilterItem]


@dataclass
class _PendingCandidate:
    """尚未得到 LLM 回答、仅允许在当前 judge 会话内补跑的候选。"""

    index: int
    feature: TrajectoryFeatures
    fallback: VetoResult
    attempts: list[DeferredAttempt]


class HardFilter:
    """规则引擎 + LLM 兜底的一票否决判定。"""

    def __init__(
        self,
        settings: Layer3Settings,
        llm_client: LLMClient,
        *,
        rule_engine: RuleEngine | DisabledRuleEngine | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._engine = rule_engine or RuleEngine(settings.phase_validity_horizon_s)
        self._citation_validator = citation_validator or CitationValidator()
        self.last_fallback_reason: str | None = None
        self.last_feature_text_hashes: dict[str, str] = {}
        self.last_citation_failures: list[CitationRepairDetail] = []
        self.last_deferred_attempts: list[DeferredAttempt] = []
        self.last_recovered_count = 0

    def filter(
        self,
        context: JudgeContext,
        features: list[TrajectoryFeatures],
    ) -> tuple[list[VetoResult], list[ReasoningStep]]:
        self.last_fallback_reason = None
        self.last_citation_failures = []
        self.last_deferred_attempts = []
        self.last_recovered_count = 0
        self.last_feature_text_hashes = {
            self._trajectory_id(context, idx): feature_text_hash(feature)
            for idx, feature in enumerate(features)
        }
        reasoning: list[ReasoningStep] = []
        results: dict[str, VetoResult] = {}
        pending: list[_PendingCandidate] = []

        for idx, feature in enumerate(features):
            traj_id = self._trajectory_id(context, idx)
            decision = self._engine.check(feature, context, traj_id)
            if decision.is_decisive:
                results[traj_id] = self._from_engine_decision(decision)
            else:
                try:
                    results[traj_id] = self._call_llm_candidate(context, idx, feature)
                except Exception as exc:
                    logger.warning("Hard filter candidate %s pending: %s", traj_id, exc)
                    pending.append(
                        _PendingCandidate(
                            index=idx,
                            feature=feature,
                            fallback=self._fallback_result(traj_id, exc),
                            attempts=[],
                        )
                    )

        self._repair_pending_candidates(context, pending, results)

        ordered: list[VetoResult] = []
        for idx in range(len(features)):
            traj_id = self._trajectory_id(context, idx)
            result = results.get(traj_id)
            if result is None:
                result = VetoResult(
                    trajectory_id=traj_id,
                    status=VetoStatus.UNCERTAIN,
                    reason="Hard filter did not return a decision",
                    decided_by=DecisionSource.FALLBACK,
                )
            ordered.append(result)
            reasoning.append(
                ReasoningStep(
                    stage=JudgeStage.HARD_FILTER,
                    content=f"轨迹 {traj_id}: {result.status.value} - {result.reason}",
                    referenced_rule_ids=result.violated_rule_ids,
                )
            )

        return ordered, reasoning

    def _call_llm_filter(
        self,
        context: JudgeContext,
        undecided: list[tuple[int, TrajectoryFeatures]],
    ) -> _LLMFilterResponse:
        system_prompt = load_system_prompt("hard_filter")
        few_shot_text = load_few_shot_examples("hard_filter")
        features_text = self._build_undecided_features_text(undecided, context)

        user_prompt = self.build_evidence_prompt(context, undecided)
        if few_shot_text:
            user_prompt += f"## 参考示例\n{few_shot_text}\n\n"
        user_prompt += (
            "判定必须同时依据义务是否触发（场景要件）和义务是否履行（轨迹特征）；"
            "要件为未知或缺失时如实返回 uncertain。"
            "请逐条判定待判定轨迹，输出 JSON 对象："
            "results 数组内每项包含 trajectory_id、status、violated_rule_ids、reason。"
        )
        LeakageGuard.assert_runtime_text_safe(
            context.scene_facts_text,
            context.scene_narrative,
            features_text,
        )

        allowed_ids = set(context.hard_rule_ids)
        last_error: Exception | None = None
        for attempt in range(self._settings.hard_filter_max_retries + 1):
            try:
                parsed, _tokens = self._llm.complete_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=_LLMFilterResponse,
                    temperature=self._settings.hard_filter_temperature,
                    timeout_seconds=self._settings.hard_filter_timeout_seconds,
                    max_retries=self._settings.hard_filter_max_retries,
                )
                for item in parsed.results:
                    self._citation_validator.validate_validity(
                        item.violated_rule_ids,
                        allowed_ids,
                        scene_id=context.scene_id,
                        stage=JudgeStage.HARD_FILTER.value,
                    )
                if self.last_citation_failures:
                    repaired = sorted({
                        rid
                        for item in parsed.results
                        for rid in item.violated_rule_ids
                    })
                    self.last_citation_failures = [
                        detail.model_copy(update={"fallback_citations": repaired})
                        for detail in self.last_citation_failures
                    ]
                return parsed
            except CitationValidityError as e:
                last_error = e
                raw_citations = sorted(e.cited_ids)
                self.last_citation_failures.append(CitationRepairDetail(
                    stage=JudgeStage.HARD_FILTER,
                    raw_citations=raw_citations,
                    fallback_citations=[],
                    allowed_citations=sorted(allowed_ids),
                    error_message=str(e),
                ))
                if attempt < self._settings.hard_filter_max_retries:
                    note_retry = getattr(self._llm, "note_retry", None)
                    if callable(note_retry):
                        note_retry()
                    user_prompt = LLMClient._augment_with_error(user_prompt, e)
                else:
                    raise
        raise LLMTimeoutError(f"Hard filter LLM call failed: {last_error}")

    def _call_llm_candidate(
        self,
        context: JudgeContext,
        idx: int,
        feature: TrajectoryFeatures,
    ) -> VetoResult:
        """一次 LLM 调用只服务一个非确定性候选，故障不会连坐其他候选。"""
        traj_id = self._trajectory_id(context, idx)
        response = self._call_llm_filter(context, [(idx, feature)])
        self._validate_llm_result_ids(response, {traj_id}, context.scene_id)
        item = response.results[0]
        try:
            return VetoResult(
                trajectory_id=item.trajectory_id,
                status=item.status,
                violated_rule_ids=item.violated_rule_ids,
                reason=item.reason,
                decided_by=DecisionSource.LLM,
            )
        except ValueError as exc:
            raise StructuredOutputValidationError(str(exc)) from exc

    def _repair_pending_candidates(
        self,
        context: JudgeContext,
        pending: list[_PendingCandidate],
        results: dict[str, VetoResult],
    ) -> None:
        """在同一 judge 会话内有界补跑基础设施 fallback，绝不重跑 LLM uncertain。"""
        remaining = pending
        for round_index in range(1, self._settings.deferred_retry_rounds + 1):
            if not remaining:
                break
            if round_index > 1 and self._settings.deferred_retry_delay_seconds:
                time.sleep(self._settings.deferred_retry_delay_seconds)

            next_round: list[_PendingCandidate] = []
            for candidate in remaining:
                # 红线：只有没有得到回答的 fallback 才有补跑资格。
                if candidate.fallback.decided_by != DecisionSource.FALLBACK:
                    results[candidate.fallback.trajectory_id] = candidate.fallback
                    continue
                started_at = datetime.now(UTC)
                try:
                    result = self._call_llm_candidate(
                        context,
                        candidate.index,
                        candidate.feature,
                    )
                except Exception as exc:
                    finished_at = datetime.now(UTC)
                    attempt = DeferredAttempt(
                        trajectory_id=candidate.fallback.trajectory_id,
                        round_index=round_index,
                        started_at=started_at,
                        finished_at=finished_at,
                        error_message=str(exc),
                    )
                    candidate.attempts.append(attempt)
                    self.last_deferred_attempts.append(attempt)
                    candidate.fallback = self._fallback_result(
                        candidate.fallback.trajectory_id,
                        exc,
                    )
                    next_round.append(candidate)
                else:
                    finished_at = datetime.now(UTC)
                    attempt = DeferredAttempt(
                        trajectory_id=result.trajectory_id,
                        round_index=round_index,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                    candidate.attempts.append(attempt)
                    self.last_deferred_attempts.append(attempt)
                    results[result.trajectory_id] = result.model_copy(
                        update={
                            "completed_by": CompletionSource.DEFERRED_RETRY,
                            "deferred_attempts": candidate.attempts,
                        }
                    )
                    self.last_recovered_count += 1
            remaining = next_round

        if remaining:
            self.last_fallback_reason = remaining[0].fallback.reason
        for candidate in remaining:
            results[candidate.fallback.trajectory_id] = candidate.fallback.model_copy(
                update={
                    "completed_by": CompletionSource.FALLBACK,
                    "deferred_attempts": candidate.attempts,
                }
            )

    @staticmethod
    def _fallback_result(trajectory_id: str, error: Exception) -> VetoResult:
        return VetoResult(
            trajectory_id=trajectory_id,
            status=VetoStatus.UNCERTAIN,
            reason=f"Hard filter LLM unavailable: {error}",
            decided_by=DecisionSource.FALLBACK,
            completed_by=CompletionSource.FALLBACK,
        )

    @staticmethod
    def _from_engine_decision(decision: RuleEngineDecision) -> VetoResult:
        return VetoResult(
            trajectory_id=decision.trajectory_id,
            status=decision.status,
            violated_rule_ids=decision.violated_rule_ids,
            reason=decision.reason,
            decided_by=DecisionSource.RULE_ENGINE,
        )

    @staticmethod
    def build_evidence_prompt(
        context: JudgeContext,
        undecided: list[tuple[int, TrajectoryFeatures]],
    ) -> str:
        """LLM 实际读到的证据区（few-shot 与固定指令段之前的部分）。

        盲标包的证据列与影子评估的 prompt 摘要都从这里取，避免在别处重写一份
        拼装逻辑后与生产悄悄漂移。
        """
        features_text = HardFilter._build_undecided_features_text(undecided, context)
        return (
            f"## 场景要件\n{context.scene_facts_text}\n"
            f"{context.hard_rules_text}\n"
            f"## 场景描述\n{context.scene_narrative}\n\n"
            f"{features_text}\n\n"
        )

    @staticmethod
    def _build_undecided_features_text(
        undecided: list[tuple[int, TrajectoryFeatures]],
        context: JudgeContext,
    ) -> str:
        lines = ["[待判定轨迹特征]"]
        for idx, feature in undecided:
            traj_id = HardFilter._trajectory_id(context, idx)
            lines.append(f"[{traj_id}] {HardFilter._feature_text(feature)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _feature_text(feature: TrajectoryFeatures) -> str:
        """生成 HardFilter 实际可见的候选特征文本。"""
        return render_feature_text(feature)

    @staticmethod
    def _validate_llm_result_ids(
        llm_response: _LLMFilterResponse,
        expected_ids: set[str],
        scene_id: str,
    ) -> None:
        actual_ids = {item.trajectory_id for item in llm_response.results}
        if actual_ids != expected_ids:
            raise StructuredOutputValidationError(
                "hard_filter result ids must match undecided trajectory ids: "
                f"actual={actual_ids}, expected={expected_ids}, scene_id={scene_id}"
            )

    @staticmethod
    def _trajectory_id(context: JudgeContext, idx: int) -> str:
        if idx < len(context.trajectory_ids):
            return context.trajectory_ids[idx]
        return f"traj_{idx}"


__all__ = ["HardFilter", "RuleEngine", "RuleEngineDecision", "_LLMFilterResponse"]
