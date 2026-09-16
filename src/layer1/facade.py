"""Layer 1 facade：RawScene / dataset 到 ParsedSceneBundle。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from src.layer1.agent_interaction import AgentInteractionAnalyzer, render_agent_interactions
from src.layer1.benchmark_builder import (
    build_benchmark_candidates,
    label_benchmark_candidates,
)
from src.layer1.candidate_anonymizer import CandidateAnonymizer
from src.layer1.dataset_loader import DriveLMDatasetLoader
from src.layer1.exceptions import SceneParseError, TrajectoryAugmentationError
from src.layer1.interface_adapter import InterfaceAdapter
from src.layer1.models import (
    AugmentedTrajectory,
    ParsedSceneBundle,
    PathTrajectory,
    RawScene,
    ScenarioType,
    SceneContext,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectoryVariantType,
)
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.scene_text_extractor import SceneTextExtractor
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor
from src.layer1.trajectory_extractor import TrajectoryExtractor

logger = logging.getLogger(__name__)

_PEDESTRIAN_SLOWDOWN_SCALES = (0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
_PEDESTRIAN_SPEEDUP_SCALES = (
    1.10,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    6.00,
    8.00,
)


def parse_scene(raw_scene: RawScene, *, seed: int = 42) -> ParsedSceneBundle:
    """解析单个 keyframe，输出 v3.1 ParsedSceneBundle。"""
    fact_extractor = SceneFactExtractor()
    text_extractor = SceneTextExtractor()
    trajectory_extractor = TrajectoryExtractor()
    augmentor = TrajectoryAugmentor(seed=seed)
    anonymizer = CandidateAnonymizer(seed=seed)
    analyzer = TrajectoryAnalyzer()
    adapter = InterfaceAdapter()

    facts = fact_extractor.extract(raw_scene)
    try:
        description = text_extractor.extract(raw_scene, facts)
    except SceneParseError:
        description = SceneDescription()
    ego_state = text_extractor.extract_ego_state(raw_scene)
    scenario_type = text_extractor.infer_scenario_type(raw_scene, facts)

    ground_truth = trajectory_extractor.extract(raw_scene)
    path_trajectory = trajectory_extractor.extract_path(raw_scene)
    drivable_area_polygons = trajectory_extractor.local_drivable_polygons(raw_scene)
    # 短窗口场景会在 admission 中整体弃权，且 benchmark 已固定标为
    # not_evaluable_short_window。不要让一个不会进入 LLM 的场景先被候选地图校验阻断；
    # 其余场景仍保留生产 drivable-area 硬校验。
    candidate_validation_polygons = (
        [] if facts.window_insufficient else drivable_area_polygons
    )
    candidates = augmentor.augment(
        ground_truth,
        scenario_type,
        facts,
        scene_id=raw_scene.scene_id,
        frame_token=raw_scene.frame_token,
        path_trajectory=path_trajectory,
        drivable_area_polygons=candidate_validation_polygons,
    )
    candidates = build_benchmark_candidates(
        candidates,
        analyzer,
        description,
        facts,
        scenario_type,
    )
    interaction_analyzer = AgentInteractionAnalyzer()
    candidates = [
        _enrich_interactions(candidate, interaction_analyzer, raw_scene, facts)
        for candidate in candidates
    ]
    candidates = _ensure_pedestrian_illegal_target(
        candidates,
        augmentor=augmentor,
        analyzer=analyzer,
        interaction_analyzer=interaction_analyzer,
        raw_scene=raw_scene,
        facts=facts,
        scenario_type=scenario_type,
        description=description,
        ground_truth=ground_truth,
        path_trajectory=path_trajectory,
        drivable_area_polygons=candidate_validation_polygons,
    )
    candidates = label_benchmark_candidates(candidates, facts, scenario_type)
    anonymized_trajectories, benchmark_labels = anonymizer.anonymize(
        candidates,
        scene_id=raw_scene.scene_id,
        frame_token=raw_scene.frame_token,
    )
    features_by_internal_id = {}
    for candidate in candidates:
        if candidate.features is None:
            raise SceneParseError("benchmark 构建后候选缺少轨迹特征")
        features_by_internal_id[candidate.trajectory.traj_id] = candidate.features
    trajectory_features = [
        features_by_internal_id[label.original_internal_id]
        for label in benchmark_labels.labels
    ]

    context = SceneContext(
        scene_id=raw_scene.scene_id,
        frame_token=raw_scene.frame_token,
        location=raw_scene.location,
        description=description,
        ego_state=ego_state,
        scenario_type=scenario_type,
        scene_facts=facts,
        candidate_trajectories=anonymized_trajectories,
        trajectory_features=trajectory_features,
    )
    scene_query = adapter.to_scene_query(context)
    judge_input = adapter.to_judge_input(context)

    return ParsedSceneBundle(
        context=context,
        scene_query=scene_query,
        judge_input=judge_input,
        benchmark_labels=benchmark_labels,
    )


def _enrich_interactions(
    candidate: AugmentedTrajectory,
    interaction_analyzer: AgentInteractionAnalyzer,
    raw_scene: RawScene,
    facts: SceneFacts,
) -> AugmentedTrajectory:
    if candidate.features is None:
        raise SceneParseError("benchmark 构建后候选缺少轨迹特征")
    interactions = interaction_analyzer.analyze(candidate.trajectory, raw_scene, facts)
    interaction_text = render_agent_interactions(interactions)
    feature = candidate.features.model_copy(update={
        "agent_interactions": interactions,
        "natural_language_summary": (
            candidate.features.natural_language_summary + interaction_text
        ),
    })
    return candidate.model_copy(update={"features": feature})


def _ensure_pedestrian_illegal_target(
    candidates: list[AugmentedTrajectory],
    *,
    augmentor: TrajectoryAugmentor,
    analyzer: TrajectoryAnalyzer,
    interaction_analyzer: AgentInteractionAnalyzer,
    raw_scene: RawScene,
    facts: SceneFacts,
    scenario_type: ScenarioType,
    description: SceneDescription,
    ground_truth: Trajectory,
    path_trajectory: PathTrajectory | Trajectory,
    drivable_area_polygons: list[list[tuple[float, float]]],
) -> list[AugmentedTrajectory]:
    if (
        scenario_type != ScenarioType.PEDESTRIAN
        or not facts.pedestrian_in_forward_crosswalk
    ):
        return candidates
    illegal_index = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.variant_type == TrajectoryVariantType.ILLEGAL
        ),
        None,
    )
    if illegal_index is None:
        return candidates
    illegal = candidates[illegal_index]
    target = augmentor.pedestrian_illegal_gap_target_s
    probe = _pedestrian_illegal_candidate_at_scale(
        speed_scale=1.0,
        augmentor=augmentor,
        analyzer=analyzer,
        interaction_analyzer=interaction_analyzer,
        raw_scene=raw_scene,
        facts=facts,
        scenario_type=scenario_type,
        description=description,
        ground_truth=ground_truth,
        path_trajectory=path_trajectory,
        drivable_area_polygons=drivable_area_polygons,
    )
    slowdown_scales, speedup_scales = _pedestrian_timing_search_scales(probe)
    for direction, scales in (("减速", slowdown_scales), ("加速", speedup_scales)):
        for speed_scale in scales:
            try:
                retry = _pedestrian_illegal_candidate_at_scale(
                    speed_scale=speed_scale,
                    augmentor=augmentor,
                    analyzer=analyzer,
                    interaction_analyzer=interaction_analyzer,
                    raw_scene=raw_scene,
                    facts=facts,
                    scenario_type=scenario_type,
                    description=description,
                    ground_truth=ground_truth,
                    path_trajectory=path_trajectory,
                    drivable_area_polygons=drivable_area_polygons,
                )
            except TrajectoryAugmentationError:
                continue
            gap = _minimum_pedestrian_gap(retry)
            if not _pedestrian_illegal_target_met(retry, target):
                continue
            retry = retry.model_copy(update={
                "generation_note": (
                    f"行人违规时机搜索：{direction}至 {speed_scale:.3f}x，"
                    f"min_time_gap_s={gap:.3f}"
                )
            })
            updated = list(candidates)
            updated[illegal_index] = retry
            return updated

    updated = list(candidates)
    updated[illegal_index] = illegal.model_copy(update={
        "generation_note": (
            "no_violation_injectable：双向时机搜索后未同时满足 "
            f"min_time_gap_s <= {target:.2f} s 与明确未区前停车"
        )
    })
    return updated


def _pedestrian_illegal_candidate_at_scale(
    *,
    speed_scale: float,
    augmentor: TrajectoryAugmentor,
    analyzer: TrajectoryAnalyzer,
    interaction_analyzer: AgentInteractionAnalyzer,
    raw_scene: RawScene,
    facts: SceneFacts,
    scenario_type: ScenarioType,
    description: SceneDescription,
    ground_truth: Trajectory,
    path_trajectory: PathTrajectory | Trajectory,
    drivable_area_polygons: list[list[tuple[float, float]]],
) -> AugmentedTrajectory:
    retry_candidates = augmentor.augment(
        ground_truth,
        scenario_type,
        facts,
        scene_id=raw_scene.scene_id,
        frame_token=raw_scene.frame_token,
        path_trajectory=path_trajectory,
        drivable_area_polygons=drivable_area_polygons,
        pedestrian_illegal_speed_scale=speed_scale,
    )
    retry = next(
        candidate
        for candidate in retry_candidates
        if candidate.variant_type == TrajectoryVariantType.ILLEGAL
    )
    retry = retry.model_copy(update={
        "features": analyzer.analyze(retry.trajectory, description, facts)
    })
    return _enrich_interactions(retry, interaction_analyzer, raw_scene, facts)


def _pedestrian_timing_search_scales(
    probe: AugmentedTrajectory,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """由 1x 到达时刻反解缩放，并用有限边界点覆盖非线性误差。"""
    derived_slowdown: list[float] = []
    derived_speedup: list[float] = []
    if probe.features is not None:
        for interaction in probe.features.agent_interactions:
            if (
                interaction.agent_kind != "pedestrian"
                or interaction.ego_zone_arrival_s is None
                or interaction.agent_zone_window_s is None
            ):
                continue
            start_s, end_s = interaction.agent_zone_window_s
            target_times = {
                max(0.05, start_s + 0.3),
                max(0.05, (start_s + end_s) / 2.0),
                max(0.05, end_s - 0.3),
            }
            for target_time_s in sorted(target_times):
                solved = interaction.ego_zone_arrival_s / target_time_s
                for bracket in (0.9, 1.0, 1.1):
                    scale = solved * bracket
                    if 0.05 <= scale < 1.0:
                        derived_slowdown.append(scale)
                    elif 1.0 < scale <= 8.0:
                        derived_speedup.append(scale)

    slowdown = _unique_scales((*derived_slowdown, *_PEDESTRIAN_SLOWDOWN_SCALES))
    speedup = _unique_scales((*derived_speedup, *_PEDESTRIAN_SPEEDUP_SCALES))
    return slowdown, speedup


def _unique_scales(scales: tuple[float, ...]) -> tuple[float, ...]:
    unique: list[float] = []
    seen: set[float] = set()
    for scale in scales:
        rounded = round(scale, 4)
        if rounded not in seen:
            seen.add(rounded)
            unique.append(rounded)
    return tuple(unique)


def _minimum_pedestrian_gap(candidate: AugmentedTrajectory) -> float | None:
    if candidate.features is None:
        return None
    gaps = [
        interaction.min_time_gap_s
        for interaction in candidate.features.agent_interactions
        if interaction.agent_kind == "pedestrian"
        and interaction.min_time_gap_s is not None
    ]
    return min(gaps) if gaps else None


def _pedestrian_illegal_target_met(
    candidate: AugmentedTrajectory,
    target_gap_s: float,
) -> bool:
    if candidate.features is None:
        return False
    gap = _minimum_pedestrian_gap(candidate)
    crossing = next(
        (
            metric
            for metric in candidate.features.conflict_zone_metrics
            if metric.kind == "ped_crossing"
        ),
        None,
    )
    return (
        gap is not None
        and gap <= target_gap_s
        and crossing is not None
        and crossing.stopped_before_zone is False
    )


def parse_dataset(
    nuscenes_root: Path,
    *,
    drivelm_qa_path: Path,
    future_horizon_s: float = 6.0,
    path_horizon_s: float = 12.0,
    on_scene_error: Callable[[Exception, RawScene], None] | None = None,
) -> Iterator[ParsedSceneBundle]:
    """遍历 Boston ∩ DriveLM keyframes 并逐帧解析。"""
    loader = DriveLMDatasetLoader(
        nuscenes_root,
        drivelm_qa_path,
        future_horizon_s=future_horizon_s,
        path_horizon_s=path_horizon_s,
    )
    for raw_scene in loader.iter_keyframes():
        try:
            bundle = parse_scene(raw_scene)
            if bundle.context.scenario_type in {
                ScenarioType.YELLOW_LIGHT,
                ScenarioType.EMERGENCY,
            }:
                logger.info(
                    "Scope-cut keyframe %s (%s)",
                    raw_scene.frame_token,
                    bundle.context.scenario_type.value,
                )
                continue
            yield bundle
        except Exception as exc:
            if on_scene_error is None:
                raise
            on_scene_error(exc, raw_scene)
