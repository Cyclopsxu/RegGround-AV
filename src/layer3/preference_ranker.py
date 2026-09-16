"""Pairwise 偏好排序模块。"""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from pydantic import BaseModel, Field

from src.layer1.models import TrajectoryFeatures
from src.layer3.citation_validator import CitationValidator
from src.layer3.exceptions import CitationValidityError, StructuredOutputValidationError
from src.layer3.leakage_guard import LeakageGuard
from src.layer3.llm_client import LLMClient
from src.layer3.models import (
    CitationRepairDetail,
    DecisionSource,
    JudgeContext,
    JudgeStage,
    Layer3Settings,
    PairwisePreference,
    ReasoningStep,
)
from src.layer3.pairwise_aggregator import PairwiseAggregator
from src.layer3.prompt_loader import load_few_shot_examples, load_system_prompt
from src.trajectory_feature_text import feature_text_hash, render_feature_text

logger = logging.getLogger(__name__)


class _LLMPairwiseResponse(BaseModel):
    preferred_id: str
    dispreferred_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    referenced_rule_ids: list[str] = Field(default_factory=list)


class PreferenceRanker:
    """对 CLEARED 轨迹做 pairwise LLM 比较并聚合排序。"""

    def __init__(
        self,
        settings: Layer3Settings,
        llm_client: LLMClient,
        *,
        aggregator: PairwiseAggregator | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._aggregator = aggregator or PairwiseAggregator()
        self._citation_validator = citation_validator or CitationValidator()
        self.last_tied_pairs_count = 0
        self.last_feature_text_hashes: dict[str, str] = {}
        self.last_citation_failures: list[CitationRepairDetail] = []
        self._diagnostics_lock = Lock()

    def rank(
        self,
        context: JudgeContext,
        cleared_features: list[TrajectoryFeatures],
        cleared_ids: list[str] | None = None,
    ) -> tuple[list[str], list[PairwisePreference], list[ReasoningStep]]:
        self.last_tied_pairs_count = 0
        self.last_feature_text_hashes = {}
        self.last_citation_failures = []
        if not cleared_features:
            return [], [], []

        if cleared_ids is None:
            cleared_ids = context.trajectory_ids[: len(cleared_features)]
        if len(cleared_ids) != len(cleared_features):
            raise StructuredOutputValidationError(
                "cleared_ids must align with cleared_features"
            )
        if len(cleared_features) == 1:
            return list(cleared_ids), [], []

        pairs: list[PairwisePreference] = []
        reasoning: list[ReasoningStep] = []
        feature_by_id = dict(zip(cleared_ids, cleared_features, strict=True))
        self.last_feature_text_hashes = {
            traj_id: feature_text_hash(feature)
            for traj_id, feature in feature_by_id.items()
        }
        comparisons = list(itertools.combinations(cleared_ids, 2))
        comparisons = comparisons[: self._settings.max_pairwise_comparisons]

        def compare(
            pair: tuple[str, str],
        ) -> tuple[_LLMPairwiseResponse, list[CitationRepairDetail]]:
            left_id, right_id = pair
            return self._call_llm_pairwise(
                context,
                left_id,
                feature_by_id[left_id],
                right_id,
                feature_by_id[right_id],
            )

        # 比较彼此独立；全局 LLM 信号量负责跨场景统一限流。
        if comparisons:
            with ThreadPoolExecutor(max_workers=min(5, len(comparisons))) as executor:
                llm_items = list(executor.map(compare, comparisons))
        else:
            llm_items = []

        for llm_item, citation_failures in llm_items:
            self.last_citation_failures.extend(citation_failures)
            preference = PairwisePreference(
                preferred_id=llm_item.preferred_id,
                dispreferred_id=llm_item.dispreferred_id,
                basis="pairwise_judgment",
                confidence=llm_item.confidence,
                reasoning=llm_item.reasoning,
                referenced_rule_ids=llm_item.referenced_rule_ids,
                decided_by=DecisionSource.LLM,
            )
            pairs.append(preference)
            reasoning.append(
                ReasoningStep(
                    stage=JudgeStage.PAIRWISE_RANKING,
                    content=(
                        f"{preference.preferred_id} preferred over "
                        f"{preference.dispreferred_id}: {preference.reasoning}"
                    ),
                    referenced_rule_ids=preference.referenced_rule_ids,
                )
            )
            if self.is_indistinguishable_pair(preference):
                self.last_tied_pairs_count += 1

        ranking = self._aggregator.aggregate(cleared_ids, pairs)
        return ranking, pairs, reasoning

    @staticmethod
    def is_indistinguishable_pair(preference: PairwisePreference) -> bool:
        """判断低置信比较是否明确表达无法区分。"""
        if preference.confidence > 0.5:
            return False
        text = preference.reasoning.lower()
        phrases = (
            "无法区分",
            "无差异",
            "完全一致",
            "随机",
            "identical",
            "cannot distinguish",
            "no discriminatory",
            "no preference",
            "no difference",
        )
        return any(phrase in text for phrase in phrases)

    def _call_llm_pairwise(
        self,
        context: JudgeContext,
        left_id: str,
        left_feature: TrajectoryFeatures,
        right_id: str,
        right_feature: TrajectoryFeatures,
    ) -> tuple[_LLMPairwiseResponse, list[CitationRepairDetail]]:
        system_prompt = load_system_prompt("preference_ranker")
        few_shot_text = load_few_shot_examples("preference_ranker")
        user_prompt = (
            f"## 场景要件\n{context.scene_facts_text}\n"
            f"{context.soft_rules_text}\n"
            f"## 场景描述\n{context.scene_narrative}\n\n"
            f"## 比较对象\n"
            f"[{left_id}] {self._feature_text(left_feature)}\n"
            f"[{right_id}] {self._feature_text(right_feature)}\n\n"
        )
        if few_shot_text:
            user_prompt += f"## 参考示例\n{few_shot_text}\n\n"
        user_prompt += (
            "只比较以上两条轨迹哪条更优。输出 JSON："
            '{"preferred_id":"...", "dispreferred_id":"...", '
            '"confidence":0.0, "reasoning":"...", "referenced_rule_ids":[]}'
        )
        LeakageGuard.assert_runtime_text_safe(
            context.scene_facts_text,
            context.scene_narrative,
            self._feature_text(left_feature),
            self._feature_text(right_feature),
        )

        allowed_rule_ids = set(context.active_rule_ids)
        compared_ids = {left_id, right_id}
        citation_failures: list[CitationRepairDetail] = []
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                parsed, _tokens = self._llm.complete_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=_LLMPairwiseResponse,
                    temperature=self._settings.pairwise_ranker_temperature,
                )
                self._validate_pairwise_output(parsed, compared_ids, context.scene_id)
                self._citation_validator.validate_validity(
                    parsed.referenced_rule_ids,
                    allowed_rule_ids,
                    scene_id=context.scene_id,
                    stage=JudgeStage.PAIRWISE_RANKING.value,
                )
                if citation_failures:
                    citation_failures = [
                        detail.model_copy(
                            update={"fallback_citations": sorted(parsed.referenced_rule_ids)}
                        )
                        for detail in citation_failures
                    ]
                return parsed, citation_failures
            except (CitationValidityError, StructuredOutputValidationError) as e:
                if isinstance(e, CitationValidityError):
                    citation_failures.append(CitationRepairDetail(
                        stage=JudgeStage.PAIRWISE_RANKING,
                        raw_citations=sorted(e.cited_ids),
                        fallback_citations=[],
                        allowed_citations=sorted(allowed_rule_ids),
                        error_message=str(e),
                    ))
                if attempt < self._settings.llm_max_retries:
                    note_retry = getattr(self._llm, "note_retry", None)
                    if callable(note_retry):
                        note_retry()
                    user_prompt = LLMClient._augment_with_error(user_prompt, e)
                else:
                    if citation_failures:
                        with self._diagnostics_lock:
                            self.last_citation_failures.extend(citation_failures)
                    raise
        raise StructuredOutputValidationError("pairwise ranker exhausted retries")

    @staticmethod
    def _validate_pairwise_output(
        parsed: _LLMPairwiseResponse,
        compared_ids: set[str],
        scene_id: str,
    ) -> None:
        output_ids = {parsed.preferred_id, parsed.dispreferred_id}
        if output_ids != compared_ids:
            raise StructuredOutputValidationError(
                "pairwise output ids must match compared ids: "
                f"output={output_ids}, compared={compared_ids}, scene_id={scene_id}"
            )

    @staticmethod
    def _feature_text(feature: TrajectoryFeatures) -> str:
        return render_feature_text(feature)


__all__ = ["PreferenceRanker", "_LLMPairwiseResponse"]
