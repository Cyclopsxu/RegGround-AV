"""PairwiseAggregator 测试。"""

from src.layer3.models import DecisionSource, PairwisePreference
from src.layer3.pairwise_aggregator import PairwiseAggregator


def _pref(preferred: str, dispreferred: str) -> PairwisePreference:
    return PairwisePreference(
        preferred_id=preferred,
        dispreferred_id=dispreferred,
        basis="pairwise_judgment",
        confidence=0.8,
        reasoning="test",
        decided_by=DecisionSource.LLM,
    )


def test_majority_wins():
    ranking = PairwiseAggregator().aggregate_majority(
        ["traj_a", "traj_b", "traj_c"],
        [_pref("traj_b", "traj_a"), _pref("traj_b", "traj_c"), _pref("traj_a", "traj_c")],
    )
    assert ranking == ["traj_b", "traj_a", "traj_c"]


def test_tie_break_uses_trajectory_id_order():
    ranking = PairwiseAggregator().aggregate_majority(
        ["traj_b", "traj_a"],
        [],
    )
    assert ranking == ["traj_a", "traj_b"]


def test_bradley_terry_entrypoint():
    ranking = PairwiseAggregator().aggregate(
        ["traj_a", "traj_b"],
        [_pref("traj_b", "traj_a")],
        method="bradley_terry",
    )
    assert ranking[0] == "traj_b"
