"""候选轨迹匿名化与 benchmark 标签隔离。"""

from __future__ import annotations

import random
from typing import Literal

from src.layer1.benchmark_builder import BENCHMARK_DATA_VERSION
from src.layer1.models import (
    AugmentedTrajectory,
    BenchmarkLabels,
    CandidateBenchmarkLabel,
    Trajectory,
    TrajectoryVariantType,
)


class CandidateAnonymizer:
    """将候选轨迹 ID 改写为 traj_a、traj_b，并固定随机顺序。"""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def anonymize(
        self,
        candidates: list[AugmentedTrajectory],
        *,
        scene_id: str = "",
        frame_token: str = "",
    ) -> tuple[list[Trajectory], BenchmarkLabels]:
        """匿名化候选轨迹并返回 benchmark-only 标签映射。"""
        order = list(range(len(candidates)))
        # 每个 keyframe 使用不同但可复现的排列，避免跨场景位置恒定。
        random.Random(f"{self.seed}:{frame_token}").shuffle(order)

        anonymized: list[Trajectory] = []
        labels: list[CandidateBenchmarkLabel] = []
        for out_idx, original_idx in enumerate(order):
            candidate = candidates[original_idx]
            anonymous_id = self._anonymous_id(out_idx)
            trajectory = candidate.trajectory.model_copy(update={"traj_id": anonymous_id})
            anonymized.append(trajectory)
            labels.append(
                CandidateBenchmarkLabel(
                    anonymous_id=anonymous_id,
                    original_internal_id=candidate.trajectory.traj_id,
                    variant_type=candidate.variant_type,
                    expected_verdict=candidate.expected_verdict,
                    expected_verdict_reason=candidate.expected_verdict_reason,
                    exclude_reason=candidate.exclude_reason,
                    predicate_version=candidate.predicate_version,
                    feature_text_hash=candidate.feature_text_hash,
                    is_degenerate=candidate.is_degenerate,
                    difficulty=self._difficulty(
                        candidate.variant_type,
                        candidate.expected_verdict != "exclude",
                    ),
                    gt_precheck_status=(
                        candidate.gt_precheck_status
                    ),
                )
            )

        return anonymized, BenchmarkLabels(
            scene_id=scene_id,
            frame_token=frame_token,
            data_version=BENCHMARK_DATA_VERSION,
            labels=labels,
        )

    @staticmethod
    def _anonymous_id(index: int) -> str:
        letters: list[str] = []
        n = index
        while True:
            letters.append(chr(ord("a") + (n % 26)))
            n = n // 26 - 1
            if n < 0:
                break
        return "traj_" + "".join(reversed(letters))

    @staticmethod
    def _difficulty(
        variant_type: TrajectoryVariantType,
        is_decisive: bool,
    ) -> Literal["easy", "medium", "hard"]:
        if variant_type == TrajectoryVariantType.HARD_CASE:
            return "hard"
        return "easy" if is_decisive else "medium"
