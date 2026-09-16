"""Pairwise preference 聚合器。"""

from __future__ import annotations

from src.layer3.models import PairwisePreference


class PairwiseAggregator:
    """将 pairwise 偏好聚合为全序 ranking。"""

    def aggregate(
        self,
        trajectory_ids: list[str],
        preferences: list[PairwisePreference],
        *,
        method: str = "majority",
    ) -> list[str]:
        if method == "bradley_terry":
            return self.aggregate_bradley_terry(trajectory_ids, preferences)
        return self.aggregate_majority(trajectory_ids, preferences)

    def aggregate_majority(
        self,
        trajectory_ids: list[str],
        preferences: list[PairwisePreference],
    ) -> list[str]:
        wins = {tid: 0 for tid in trajectory_ids}
        for pref in preferences:
            if pref.preferred_id in wins and pref.dispreferred_id in wins:
                wins[pref.preferred_id] += 1
        return sorted(trajectory_ids, key=lambda tid: (-wins[tid], tid))

    def aggregate_bradley_terry(
        self,
        trajectory_ids: list[str],
        preferences: list[PairwisePreference],
    ) -> list[str]:
        wins = {tid: 0 for tid in trajectory_ids}
        losses = {tid: 0 for tid in trajectory_ids}
        for pref in preferences:
            if pref.preferred_id in wins and pref.dispreferred_id in wins:
                wins[pref.preferred_id] += 1
                losses[pref.dispreferred_id] += 1

        def score(tid: str) -> float:
            return (wins[tid] + 0.5) / (wins[tid] + losses[tid] + 1.0)

        return sorted(trajectory_ids, key=lambda tid: (-score(tid), tid))
