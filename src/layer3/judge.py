"""Layer 3 Facade：逻辑裁判编排器。"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from src.layer1.models import JudgeInput, TrajectoryFeatures
from src.layer2.models import RuleSubgraph
from src.layer3.citation_validator import CitationValidator
from src.layer3.context_builder import ContextBuilder
from src.layer3.exceptions import (
    CitationValidityError,
    ContextTooLargeError,
    LLMTimeoutError,
    StructuredOutputValidationError,
)
from src.layer3.hard_filter import HardFilter
from src.layer3.label_generator import LabelGenerator
from src.layer3.llm_client import LLMClient, LLMUsageSnapshot
from src.layer3.models import (
    AuditLabel,
    CitationRepairDetail,
    ConnectionEvent,
    DecisionSource,
    DeferredAttempt,
    JudgeContext,
    JudgeStage,
    Layer3Settings,
    PairwisePreference,
    PartialReason,
    ReasoningStep,
    StageDiagnostics,
    TrajectoryVerdict,
    VetoResult,
    VetoStatus,
)
from src.layer3.preference_ranker import PreferenceRanker
from src.layer3.rule_engine import build_rule_engine

logger = logging.getLogger(__name__)


class AuditJudge:
    """Layer 3 编排 facade；业务异常降级为 AuditLabel。"""

    def __init__(
        self,
        settings: Layer3Settings,
        *,
        llm_client: LLMClient | None = None,
        context_builder: ContextBuilder | None = None,
        hard_filter: HardFilter | None = None,
        preference_ranker: PreferenceRanker | None = None,
        label_generator: LabelGenerator | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client or LLMClient(settings)
        self._citation_validator = citation_validator or CitationValidator()
        self._context_builder = context_builder or ContextBuilder(settings)
        self._hard_filter = hard_filter or HardFilter(
            settings,
            self._llm,
            rule_engine=build_rule_engine(
                phase_validity_horizon_s=settings.phase_validity_horizon_s,
                disabled=settings.disable_rule_engine,
            ),
            citation_validator=self._citation_validator,
        )
        self._preference_ranker = preference_ranker or PreferenceRanker(
            settings,
            self._llm,
            citation_validator=self._citation_validator,
        )
        self._label_generator = label_generator or LabelGenerator(settings, self._llm)

    def judge(self, judge_input: JudgeInput, subgraph: RuleSubgraph) -> AuditLabel:
        start = time.perf_counter()
        diagnostics: list[StageDiagnostics] = []
        reasoning_chain: list[ReasoningStep] = []
        partial_reasons: set[PartialReason] = set()

        if not subgraph.active_rules:
            diagnostics.append(
                StageDiagnostics(
                    stage=JudgeStage.CONTEXT_BUILDING,
                    duration_ms=0,
                    succeeded=False,
                    error_message="未检索到可用法规子图",
                )
            )
            return self._build_failure_label(
                judge_input,
                diagnostics,
                "未检索到可用法规子图",
                start,
            )

        ctx, diag = self._stage_context(judge_input, subgraph)
        diagnostics.append(diag)
        if ctx is None:
            return self._build_failure_label(
                judge_input,
                diagnostics,
                "Context building failed",
                start,
            )

        features = judge_input.trajectory_features[: len(ctx.trajectory_ids)]
        veto_results, diag, hard_reasoning = self._stage_hard_filter(ctx, features)
        diagnostics.append(diag)
        if diag.citation_failures:
            partial_reasons.add(PartialReason.CITATION_REPAIR)
        reasoning_chain.extend(hard_reasoning)
        if not diag.succeeded:
            partial_reasons.add(PartialReason.STAGE_DEGRADED)

        cleared_ids, cleared_features = self._get_cleared_features(
            features,
            veto_results,
            ctx,
        )
        ranking_cleared, pairwise_pairs, diag, rank_reasoning = self._stage_pairwise_ranker(
            ctx,
            cleared_ids,
            cleared_features,
        )
        diagnostics.append(diag)
        if diag.citation_failures:
            partial_reasons.add(PartialReason.CITATION_REPAIR)
        reasoning_chain.extend(rank_reasoning)
        if not diag.succeeded:
            partial_reasons.add(PartialReason.STAGE_DEGRADED)

        compliance_pairs = self._build_compliance_pairs(veto_results)
        preference_pairs = compliance_pairs + pairwise_pairs
        ranking = self._merge_ranking(ranking_cleared, veto_results, ctx.trajectory_ids)
        verdicts = self._build_verdicts(veto_results, ranking_cleared)
        if any(verdict.veto.status == VetoStatus.UNCERTAIN for verdict in verdicts):
            partial_reasons.add(PartialReason.UNCERTAIN_VERDICT)
        if any(
            self._preference_ranker.is_indistinguishable_pair(pair)
            for pair in pairwise_pairs
        ):
            partial_reasons.add(PartialReason.LOW_CONFIDENCE_PAIR)
        legal_basis = self._collect_legal_basis(verdicts, preference_pairs, reasoning_chain)

        summary, diag = self._stage_label_generator(
            judge_input,
            subgraph,
            verdicts,
            preference_pairs,
            reasoning_chain,
        )
        diagnostics.append(diag)
        if not diag.succeeded:
            partial_reasons.add(PartialReason.STAGE_DEGRADED)

        summary, legal_basis, citation_validity, citation_repair = self._validate_summary(
            summary,
            legal_basis,
            subgraph,
            verdicts,
            preference_pairs,
        )
        if citation_repair is not None:
            partial_reasons.add(PartialReason.CITATION_REPAIR)
            diagnostics[-1] = diagnostics[-1].model_copy(
                update={"citation_failures": [citation_repair]}
            )

        try:
            return AuditLabel(
                scene_id=judge_input.scene_id,
                frame_token=judge_input.frame_token,
                partial_reasons=[
                    reason for reason in PartialReason if reason in partial_reasons
                ],
                chosen_trajectory_id=ranking_cleared[0] if ranking_cleared else None,
                preference_ranking=ranking,
                preference_pairs=preference_pairs,
                verdicts=verdicts,
                natural_language_summary=summary,
                legal_basis=legal_basis,
                reasoning_chain=reasoning_chain,
                stage_diagnostics=diagnostics,
                connection_events=self._connection_events(),
                citation_validity=citation_validity,
                total_duration_ms=int((time.perf_counter() - start) * 1000),
                tied_pairs_count=self._preference_ranker.last_tied_pairs_count,
                generated_at=datetime.now(UTC),
            )
        except Exception as exc:
            logger.warning("AuditLabel assembly failed: %s", exc)
            diagnostics.append(
                StageDiagnostics(
                    stage=JudgeStage.LABEL_GENERATION,
                    duration_ms=0,
                    succeeded=False,
                    error_message=f"AuditLabel assembly failed: {exc}",
                )
            )
            return self._build_failure_label(
                judge_input,
                diagnostics,
                str(exc),
                start,
            )

    def _stage_context(
        self,
        judge_input: JudgeInput,
        subgraph: RuleSubgraph,
    ) -> tuple[JudgeContext | None, StageDiagnostics]:
        t0 = time.perf_counter()
        try:
            ctx = self._context_builder.build(judge_input, subgraph)
            return ctx, StageDiagnostics(
                stage=JudgeStage.CONTEXT_BUILDING,
                duration_ms=self._elapsed_ms(t0),
                succeeded=True,
            )
        except ContextTooLargeError as exc:
            return None, StageDiagnostics(
                stage=JudgeStage.CONTEXT_BUILDING,
                duration_ms=self._elapsed_ms(t0),
                succeeded=False,
                error_message=str(exc),
            )
        except Exception as exc:
            return None, StageDiagnostics(
                stage=JudgeStage.CONTEXT_BUILDING,
                duration_ms=self._elapsed_ms(t0),
                succeeded=False,
                error_message=str(exc),
            )

    def _stage_hard_filter(
        self,
        ctx: JudgeContext,
        features: list[TrajectoryFeatures],
    ) -> tuple[list[VetoResult], StageDiagnostics, list[ReasoningStep]]:
        t0 = time.perf_counter()
        usage_before = self._usage_snapshot()
        try:
            veto_results, reasoning = self._hard_filter.filter(ctx, features)
            fallback_reason = self._hard_filter.last_fallback_reason
            return veto_results, self._stage_diagnostics(
                JudgeStage.HARD_FILTER,
                t0,
                usage_before,
                succeeded=fallback_reason is None,
                error_message=fallback_reason,
                feature_text_hashes=self._hard_filter.last_feature_text_hashes,
                citation_failures=self._hard_filter.last_citation_failures,
                deferred_attempts=self._hard_filter.last_deferred_attempts,
                recovered_count=self._hard_filter.last_recovered_count,
            ), reasoning
        except Exception as exc:
            logger.warning("Hard filter failed: %s", exc)
            veto_results = self._uncertain_for_all(features, ctx, str(exc))
            reasoning = [
                ReasoningStep(
                    stage=JudgeStage.HARD_FILTER,
                    content=f"Hard filter failed: {exc}",
                )
            ]
            return veto_results, self._stage_diagnostics(
                JudgeStage.HARD_FILTER,
                t0,
                usage_before,
                succeeded=False,
                error_message=str(exc),
                feature_text_hashes=self._hard_filter.last_feature_text_hashes,
                citation_failures=self._hard_filter.last_citation_failures,
                deferred_attempts=self._hard_filter.last_deferred_attempts,
                recovered_count=self._hard_filter.last_recovered_count,
            ), reasoning

    def _stage_pairwise_ranker(
        self,
        ctx: JudgeContext,
        cleared_ids: list[str],
        cleared_features: list[TrajectoryFeatures],
    ) -> tuple[list[str], list[PairwisePreference], StageDiagnostics, list[ReasoningStep]]:
        t0 = time.perf_counter()
        usage_before = self._usage_snapshot()
        planned_comparisons = len(cleared_features) * (len(cleared_features) - 1) // 2
        if len(cleared_features) <= 1:
            return list(cleared_ids), [], StageDiagnostics(
                stage=JudgeStage.PAIRWISE_RANKING,
                duration_ms=0,
                planned_pairwise_comparisons=planned_comparisons,
                completed_pairwise_comparisons=0,
                succeeded=True,
            ), []
        try:
            ranking, pairs, reasoning = self._preference_ranker.rank(
                ctx,
                cleared_features,
                cleared_ids,
            )
            return ranking, pairs, self._stage_diagnostics(
                JudgeStage.PAIRWISE_RANKING,
                t0,
                usage_before,
                succeeded=True,
                planned_pairwise_comparisons=planned_comparisons,
                completed_pairwise_comparisons=len(pairs),
                feature_text_hashes=self._preference_ranker.last_feature_text_hashes,
                citation_failures=self._preference_ranker.last_citation_failures,
            ), reasoning
        except (
            CitationValidityError,
            LLMTimeoutError,
            StructuredOutputValidationError,
        ) as exc:
            logger.warning("Pairwise ranker failed: %s", exc)
            return list(cleared_ids), [], self._stage_diagnostics(
                JudgeStage.PAIRWISE_RANKING,
                t0,
                usage_before,
                succeeded=False,
                planned_pairwise_comparisons=planned_comparisons,
                completed_pairwise_comparisons=0,
                error_message=str(exc),
                feature_text_hashes=self._preference_ranker.last_feature_text_hashes,
                citation_failures=self._preference_ranker.last_citation_failures,
            ), [
                ReasoningStep(
                    stage=JudgeStage.PAIRWISE_RANKING,
                    content=f"Pairwise ranking failed; using input order: {exc}",
                )
            ]
        except Exception as exc:
            logger.warning("Pairwise ranker unexpected error: %s", exc)
            return list(cleared_ids), [], self._stage_diagnostics(
                JudgeStage.PAIRWISE_RANKING,
                t0,
                usage_before,
                succeeded=False,
                planned_pairwise_comparisons=planned_comparisons,
                completed_pairwise_comparisons=0,
                error_message=str(exc),
                feature_text_hashes=self._preference_ranker.last_feature_text_hashes,
                citation_failures=self._preference_ranker.last_citation_failures,
            ), []

    def _stage_label_generator(
        self,
        judge_input: JudgeInput,
        subgraph: RuleSubgraph,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
        reasoning_chain: list[ReasoningStep],
    ) -> tuple[str, StageDiagnostics]:
        t0 = time.perf_counter()
        usage_before = self._usage_snapshot()
        try:
            summary = self._label_generator.generate(
                judge_input,
                subgraph,
                verdicts,
                preference_pairs,
                reasoning_chain,
            )
            return summary, self._stage_diagnostics(
                JudgeStage.LABEL_GENERATION,
                t0,
                usage_before,
                succeeded=True,
            )
        except Exception as exc:
            logger.warning("Label generator failed: %s", exc)
            summary = self._label_generator.fallback_summary(verdicts, preference_pairs)
            return summary, self._stage_diagnostics(
                JudgeStage.LABEL_GENERATION,
                t0,
                usage_before,
                succeeded=False,
                error_message=str(exc),
            )

    @staticmethod
    def _get_cleared_features(
        features: list[TrajectoryFeatures],
        veto_results: list[VetoResult],
        ctx: JudgeContext,
    ) -> tuple[list[str], list[TrajectoryFeatures]]:
        veto_by_id = {v.trajectory_id: v for v in veto_results}
        cleared_ids: list[str] = []
        cleared_features: list[TrajectoryFeatures] = []
        for idx, feature in enumerate(features):
            traj_id = ctx.trajectory_ids[idx]
            verdict = veto_by_id.get(traj_id)
            if verdict and verdict.status == VetoStatus.CLEARED:
                cleared_ids.append(traj_id)
                cleared_features.append(feature)
        return cleared_ids, cleared_features

    @staticmethod
    def _uncertain_for_all(
        features: list[TrajectoryFeatures],
        ctx: JudgeContext,
        reason: str,
    ) -> list[VetoResult]:
        return [
            VetoResult(
                trajectory_id=ctx.trajectory_ids[idx],
                status=VetoStatus.UNCERTAIN,
                reason=reason,
                decided_by=DecisionSource.FALLBACK,
            )
            for idx, _feature in enumerate(features)
        ]

    @staticmethod
    def _build_compliance_pairs(veto_results: list[VetoResult]) -> list[PairwisePreference]:
        cleared = [v for v in veto_results if v.status == VetoStatus.CLEARED]
        vetoed = [v for v in veto_results if v.status == VetoStatus.VETOED]
        pairs: list[PairwisePreference] = []
        for cleared_result in cleared:
            for vetoed_result in vetoed:
                pairs.append(
                    PairwisePreference(
                        preferred_id=cleared_result.trajectory_id,
                        dispreferred_id=vetoed_result.trajectory_id,
                        basis="compliance",
                        confidence=1.0,
                        reasoning=(
                            f"{cleared_result.trajectory_id} cleared hard rules; "
                            f"{vetoed_result.trajectory_id} was vetoed"
                        ),
                        referenced_rule_ids=vetoed_result.violated_rule_ids,
                        decided_by=DecisionSource.RULE_ENGINE,
                    )
                )
        return pairs

    @staticmethod
    def _merge_ranking(
        ranking_cleared: list[str],
        veto_results: list[VetoResult],
        trajectory_ids: list[str],
    ) -> list[str]:
        status_by_id = {v.trajectory_id: v.status for v in veto_results}
        cleared_input = [
            tid for tid in trajectory_ids if status_by_id.get(tid) == VetoStatus.CLEARED
        ]
        ranked = [tid for tid in ranking_cleared if tid in cleared_input]
        ranked.extend(tid for tid in cleared_input if tid not in ranked)
        uncertain = [
            tid for tid in trajectory_ids if status_by_id.get(tid) == VetoStatus.UNCERTAIN
        ]
        vetoed = [
            tid for tid in trajectory_ids if status_by_id.get(tid) == VetoStatus.VETOED
        ]
        return ranked + uncertain + vetoed

    @staticmethod
    def _build_verdicts(
        veto_results: list[VetoResult],
        ranking_cleared: list[str],
    ) -> list[TrajectoryVerdict]:
        rank_by_id = {traj_id: rank + 1 for rank, traj_id in enumerate(ranking_cleared)}
        return [
            TrajectoryVerdict(
                trajectory_id=result.trajectory_id,
                veto=result,
                rank=rank_by_id.get(result.trajectory_id)
                if result.status == VetoStatus.CLEARED
                else None,
            )
            for result in veto_results
        ]

    @staticmethod
    def _collect_legal_basis(
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
        reasoning_chain: list[ReasoningStep],
    ) -> list[str]:
        ids: set[str] = set()
        for verdict in verdicts:
            ids.update(verdict.veto.violated_rule_ids)
        for pair in preference_pairs:
            ids.update(pair.referenced_rule_ids)
        for step in reasoning_chain:
            ids.update(step.referenced_rule_ids)
        return sorted(ids)

    def _validate_summary(
        self,
        summary: str,
        legal_basis: list[str],
        subgraph: RuleSubgraph,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
    ) -> tuple[str, list[str], float, CitationRepairDetail | None]:
        allowed = {rule.node_id for rule in subgraph.active_rules}
        summary_refs = set(self._citation_validator.extract_summary_citations(summary))
        legal_basis_set = set(legal_basis)
        try:
            self._citation_validator.validate_validity(
                legal_basis_set | summary_refs,
                allowed,
                stage=JudgeStage.LABEL_GENERATION.value,
            )
            if summary_refs - legal_basis_set:
                raise CitationValidityError(
                    "summary citations must be included in legal_basis",
                    cited_ids=summary_refs,
                    allowed_ids=legal_basis_set,
                    stage=JudgeStage.LABEL_GENERATION.value,
                )
            return summary, sorted(legal_basis_set), 1.0, None
        except CitationValidityError as exc:
            logger.warning("Citation validity failed in summary: %s", exc)
            fallback = self._label_generator.fallback_summary(verdicts, preference_pairs)
            fallback_refs = set(self._citation_validator.extract_summary_citations(fallback))
            legal_basis_set.update(fallback_refs & allowed)
            repair = CitationRepairDetail(
                stage=JudgeStage.LABEL_GENERATION,
                raw_citations=sorted(summary_refs),
                fallback_citations=sorted(fallback_refs),
                allowed_citations=sorted(allowed),
                error_message=str(exc),
            )
            return fallback, sorted(legal_basis_set), 1.0, repair

    @staticmethod
    def _build_failure_label(
        judge_input: JudgeInput,
        diagnostics: list[StageDiagnostics],
        error_message: str,
        start: float,
    ) -> AuditLabel:
        return AuditLabel(
            scene_id=judge_input.scene_id,
            frame_token=judge_input.frame_token,
            partial_reasons=[PartialReason.FATAL_ERROR],
            chosen_trajectory_id=None,
            preference_ranking=[],
            preference_pairs=[],
            verdicts=[],
            natural_language_summary=(
                f"评判流程失败，无法生成有效标签。错误：{error_message}"
            ),
            legal_basis=[],
            reasoning_chain=[
                ReasoningStep(
                    stage=JudgeStage.CONTEXT_BUILDING,
                    content=f"Fatal error: {error_message}",
                )
            ],
            stage_diagnostics=diagnostics,
            citation_validity=1.0,
            total_duration_ms=int((time.perf_counter() - start) * 1000),
            generated_at=datetime.now(UTC),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    def _usage_snapshot(self) -> LLMUsageSnapshot:
        """兼容没有 telemetry 接口的测试 LLM。"""
        snapshot = getattr(self._llm, "usage_snapshot", None)
        return snapshot() if callable(snapshot) else LLMUsageSnapshot()

    def _connection_events(self) -> list[ConnectionEvent]:
        """兼容不提供事件接口的离线测试 LLM。"""
        events = getattr(self._llm, "connection_events", None)
        return events() if callable(events) else []

    def _stage_diagnostics(
        self,
        stage: JudgeStage,
        start: float,
        usage_before: LLMUsageSnapshot,
        *,
        succeeded: bool,
        error_message: str | None = None,
        feature_text_hashes: dict[str, str] | None = None,
        citation_failures: list[CitationRepairDetail] | None = None,
        deferred_attempts: list[DeferredAttempt] | None = None,
        recovered_count: int = 0,
        planned_pairwise_comparisons: int = 0,
        completed_pairwise_comparisons: int = 0,
    ) -> StageDiagnostics:
        """构造带 LLM 用量增量的阶段诊断。"""
        usage = self._usage_snapshot().delta(usage_before)
        return StageDiagnostics(
            stage=stage,
            duration_ms=self._elapsed_ms(start),
            llm_tokens_used=usage.input_tokens + usage.output_tokens,
            llm_input_tokens=usage.input_tokens,
            llm_output_tokens=usage.output_tokens,
            llm_call_count=usage.call_count,
            llm_retry_count=usage.retry_count,
            planned_pairwise_comparisons=planned_pairwise_comparisons,
            completed_pairwise_comparisons=completed_pairwise_comparisons,
            deferred_attempts=deferred_attempts or [],
            recovered_count=recovered_count,
            feature_text_hashes=feature_text_hashes or {},
            citation_failures=citation_failures or [],
            succeeded=succeeded,
            error_message=error_message,
        )
