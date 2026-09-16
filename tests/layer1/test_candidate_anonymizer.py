"""Candidate anonymization and benchmark-label isolation tests."""

from __future__ import annotations

from src.layer1.candidate_anonymizer import CandidateAnonymizer
from src.layer1.models import (
    AugmentedTrajectory,
    Trajectory,
    TrajectorySource,
    TrajectoryVariantType,
    Waypoint,
)


def _trajectory(traj_id: str) -> Trajectory:
    return Trajectory(
        traj_id=traj_id,
        source=TrajectorySource.SYNTHETIC,
        waypoints=[
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=1.0, y=0.0, t=0.5),
        ],
    )


def _candidate(variant_type: TrajectoryVariantType) -> AugmentedTrajectory:
    return AugmentedTrajectory(
        trajectory=_trajectory(f"scene_1_synthetic_{variant_type.value}_0"),
        variant_type=variant_type,
        expected_verdict="vetoed" if variant_type == TrajectoryVariantType.ILLEGAL else "cleared",
        gt_precheck_status=(
            "passed" if variant_type == TrajectoryVariantType.GROUND_TRUTH else "not_gt"
        ),
    )


def _candidates() -> list[AugmentedTrajectory]:
    return [_candidate(variant_type) for variant_type in TrajectoryVariantType]


class TestCandidateAnonymizer:
    def test_ids_are_traj_letters_and_order_is_seeded(self) -> None:
        anonymized, labels = CandidateAnonymizer(seed=42).anonymize(
            _candidates(),
            scene_id="scene_1",
            frame_token="frame_1",
        )

        assert [trajectory.traj_id for trajectory in anonymized] == [
            "traj_a",
            "traj_b",
            "traj_c",
            "traj_d",
            "traj_e",
            "traj_f",
        ]
        first_order = [label.original_internal_id for label in labels.labels]
        repeated = CandidateAnonymizer(seed=42).anonymize(
            _candidates(), scene_id="scene_1", frame_token="frame_1"
        )[1]
        other_frame = CandidateAnonymizer(seed=42).anonymize(
            _candidates(), scene_id="scene_1", frame_token="frame_2"
        )[1]
        assert first_order == [label.original_internal_id for label in repeated.labels]
        assert first_order != [label.original_internal_id for label in other_frame.labels]

    def test_same_seed_reproducible_and_different_seed_changes_mapping(self) -> None:
        first = CandidateAnonymizer(seed=7).anonymize(_candidates())[1]
        second = CandidateAnonymizer(seed=7).anonymize(_candidates())[1]
        third = CandidateAnonymizer(seed=8).anonymize(_candidates())[1]

        assert first.model_dump() == second.model_dump()
        assert first.model_dump() != third.model_dump()

    def test_variant_labels_stay_only_in_benchmark_labels(self) -> None:
        anonymized, labels = CandidateAnonymizer(seed=42).anonymize(_candidates())

        assert all(not hasattr(trajectory, "variant_type") for trajectory in anonymized)
        assert {label.variant_type for label in labels.labels} == set(TrajectoryVariantType)

    def test_difficulty_and_gt_precheck_status(self) -> None:
        _, labels = CandidateAnonymizer(seed=42).anonymize(_candidates())
        by_variant = {label.variant_type: label for label in labels.labels}

        assert by_variant[TrajectoryVariantType.HARD_CASE].difficulty == "hard"
        assert by_variant[TrajectoryVariantType.ILLEGAL].difficulty == "easy"
        assert by_variant[TrajectoryVariantType.SUBOPTIMAL].difficulty == "easy"
        assert by_variant[TrajectoryVariantType.GROUND_TRUTH].gt_precheck_status == "passed"
        assert by_variant[TrajectoryVariantType.AGGRESSIVE].gt_precheck_status == "not_gt"

    def test_anonymous_id_rolls_over_after_z(self) -> None:
        assert CandidateAnonymizer._anonymous_id(0) == "traj_a"
        assert CandidateAnonymizer._anonymous_id(25) == "traj_z"
        assert CandidateAnonymizer._anonymous_id(26) == "traj_aa"
