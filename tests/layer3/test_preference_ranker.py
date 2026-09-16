"""PreferenceRanker v3.0 pairwise 测试。"""

import pytest

from src.layer3.exceptions import CitationValidityError, StructuredOutputValidationError
from src.layer3.models import VetoStatus
from src.layer3.preference_ranker import PreferenceRanker, _LLMPairwiseResponse


class FakePairwiseLLM:
    def __init__(self, responses: list[_LLMPairwiseResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete_structured(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0), 20

    def complete_text(self, *args, **kwargs):
        return "", 0


def test_zero_and_one_cleared_do_not_call_llm(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakePairwiseLLM([])
    ranker = PreferenceRanker(layer3_settings, llm)

    assert ranker.rank(sample_judge_context, []) == ([], [], [])
    ranking, pairs, reasoning = ranker.rank(
        sample_judge_context,
        [sample_features[0]],
        ["traj_a"],
    )

    assert ranking == ["traj_a"]
    assert pairs == []
    assert reasoning == []
    assert llm.calls == 0


def test_pairwise_ranking_normal(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakePairwiseLLM(
        [
            _LLMPairwiseResponse(
                preferred_id="traj_a",
                dispreferred_id="traj_b",
                confidence=0.9,
                reasoning="traj_a stops earlier with larger margin",
                referenced_rule_ids=["R-SIG-03"],
            )
        ]
    )
    ranker = PreferenceRanker(layer3_settings, llm)
    ranking, pairs, reasoning = ranker.rank(
        sample_judge_context,
        sample_features[:2],
        ["traj_a", "traj_b"],
    )

    assert ranking == ["traj_a", "traj_b"]
    assert len(pairs) == 1
    assert pairs[0].basis == "pairwise_judgment"
    assert pairs[0].referenced_rule_ids == ["R-SIG-03"]
    assert reasoning[0].stage.value == "pairwise_ranking"


def test_low_confidence_indistinguishable_pair_is_counted_as_tie(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakePairwiseLLM([
        _LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.5,
            reasoning="两条轨迹特征完全一致，无法区分",
        )
    ])
    ranker = PreferenceRanker(layer3_settings, llm)
    ranking, pairs, _reasoning = ranker.rank(
        sample_judge_context,
        [sample_features[0], sample_features[0]],
        ["traj_b", "traj_a"],
    )

    assert ranking == ["traj_a", "traj_b"]
    assert len(pairs) == 1
    assert llm.calls == 1
    assert ranker.last_tied_pairs_count == 1
    assert set(ranker.last_feature_text_hashes) == {"traj_a", "traj_b"}


def test_pairwise_output_ids_must_match_compared_ids(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakePairwiseLLM(
        [
            _LLMPairwiseResponse(
                preferred_id="traj_a",
                dispreferred_id="traj_x",
                confidence=0.5,
                reasoning="bad id",
            )
        ]
    )
    ranker = PreferenceRanker(layer3_settings, llm)
    with pytest.raises(StructuredOutputValidationError):
        ranker.rank(sample_judge_context, sample_features[:2], ["traj_a", "traj_b"])


def test_pairwise_citation_validity(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakePairwiseLLM(
        [
            _LLMPairwiseResponse(
                preferred_id="traj_a",
                dispreferred_id="traj_b",
                confidence=0.5,
                reasoning="bad citation",
                referenced_rule_ids=["R-FAKE-01"],
            )
        ]
    )
    ranker = PreferenceRanker(layer3_settings, llm)
    with pytest.raises(CitationValidityError):
        ranker.rank(sample_judge_context, sample_features[:2], ["traj_a", "traj_b"])
    assert ranker.last_citation_failures[0].raw_citations == ["R-FAKE-01"]


def test_imported_status_enum_still_available_for_veto_flow():
    assert VetoStatus.UNCERTAIN.value == "uncertain"
