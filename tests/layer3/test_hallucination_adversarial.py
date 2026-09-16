"""Citation validity 对抗测试。"""

import pytest

from src.layer3.exceptions import CitationValidityError
from src.layer3.hard_filter import HardFilter, _LLMFilterResponse
from src.layer3.models import DecisionSource, VetoStatus
from src.layer3.preference_ranker import PreferenceRanker, _LLMPairwiseResponse
from src.layer3.rule_engine import RuleEngine, RuleEngineDecision


class PassThroughEngine(RuleEngine):
    def check(self, features, context, traj_id):
        return RuleEngineDecision(
            trajectory_id=traj_id,
            status=VetoStatus.UNCERTAIN,
            reason="force LLM",
            is_decisive=False,
        )


class StaticLLM:
    def __init__(self, response) -> None:
        self.response = response

    def complete_structured(self, *args, **kwargs):
        return self.response, 10

    def complete_text(self, *args, **kwargs):
        return "", 0


def test_hard_filter_invalid_citation_degrades(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    response = _LLMFilterResponse(
        results=[
            {
                "trajectory_id": "traj_a",
                "status": "vetoed",
                "violated_rule_ids": ["R-999"],
                "reason": "bad",
            },
            {
                "trajectory_id": "traj_b",
                "status": "vetoed",
                "violated_rule_ids": ["R-999"],
                "reason": "bad",
            },
            {
                "trajectory_id": "traj_c",
                "status": "vetoed",
                "violated_rule_ids": ["R-999"],
                "reason": "bad",
            },
        ]
    )
    hard_filter = HardFilter(
        layer3_settings,
        StaticLLM(response),
        rule_engine=PassThroughEngine(),
    )
    results, _reasoning = hard_filter.filter(sample_judge_context, sample_features)

    assert all(result.status == VetoStatus.UNCERTAIN for result in results)
    assert all(result.decided_by == DecisionSource.FALLBACK for result in results)


def test_pairwise_invalid_citation_raises(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    response = _LLMPairwiseResponse(
        preferred_id="traj_a",
        dispreferred_id="traj_b",
        confidence=0.8,
        reasoning="bad citation",
        referenced_rule_ids=["R-999"],
    )
    ranker = PreferenceRanker(layer3_settings, StaticLLM(response))
    with pytest.raises(CitationValidityError):
        ranker.rank(sample_judge_context, sample_features[:2], ["traj_a", "traj_b"])
