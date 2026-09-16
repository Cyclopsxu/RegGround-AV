"""复核冻结 preflight 中 no_violation_injectable 的双向时机可达域。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.layer1.agent_interaction import AgentInteractionAnalyzer
from src.layer1.facade import (
    _minimum_pedestrian_gap,
    _pedestrian_illegal_candidate_at_scale,
    _pedestrian_illegal_target_met,
    _pedestrian_timing_search_scales,
    parse_scene,
)
from src.layer1.models import RawScene, TrajectoryVariantType
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor
from src.layer1.trajectory_extractor import TrajectoryExtractor


def _range(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def _crossing_stop_state(candidate: Any) -> bool | None | str:
    crossing = next(
        (
            metric
            for metric in candidate.features.conflict_zone_metrics
            if metric.kind == "ped_crossing"
        ),
        None,
    )
    return "missing" if crossing is None else crossing.stopped_before_zone


def _direction_domain(
    rows: list[tuple[float, Any]],
    target_gap_s: float,
) -> dict[str, Any]:
    arrivals = [
        interaction.ego_zone_arrival_s
        for _, candidate in rows
        for interaction in candidate.features.agent_interactions
        if interaction.agent_kind == "pedestrian"
        and interaction.ego_zone_arrival_s is not None
    ]
    gaps = [
        gap
        for _, candidate in rows
        if (gap := _minimum_pedestrian_gap(candidate)) is not None
    ]
    stop_states = {_crossing_stop_state(candidate) for _, candidate in rows}
    return {
        "arrival_interval_s": _range(arrivals),
        "min_gap_interval_s": _range(gaps),
        "tested_scales": [scale for scale, _ in rows],
        "shared_zone_scale_count": sum(
            any(
                interaction.agent_kind == "pedestrian"
                and interaction.ego_zone_arrival_s is not None
                for interaction in candidate.features.agent_interactions
            )
            for _, candidate in rows
        ),
        "gap_target_reached": any(gap <= target_gap_s for gap in gaps),
        "violation_target_reached": any(
            _pedestrian_illegal_target_met(candidate, target_gap_s)
            for _, candidate in rows
        ),
        "stopped_before_zone_values": sorted(
            stop_states,
            key=lambda value: (value is not None, str(value)),
        ),
    }


def _per_track_domains(
    slowdown: list[tuple[float, Any]],
    speedup: list[tuple[float, Any]],
) -> list[dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for direction, rows in (("slowdown", slowdown), ("speedup", speedup)):
        for _, candidate in rows:
            for interaction in candidate.features.agent_interactions:
                if interaction.agent_kind != "pedestrian":
                    continue
                track = tracks.setdefault(interaction.agent_track_id, {
                    "agent_track_id": interaction.agent_track_id,
                    "agent_zone_windows_s": set(),
                    "slowdown_arrivals_s": [],
                    "speedup_arrivals_s": [],
                })
                if interaction.agent_zone_window_s is not None:
                    track["agent_zone_windows_s"].add(interaction.agent_zone_window_s)
                if interaction.ego_zone_arrival_s is not None:
                    track[f"{direction}_arrivals_s"].append(
                        interaction.ego_zone_arrival_s
                    )
    return [
        {
            "agent_track_id": track["agent_track_id"],
            "agent_zone_windows_s": sorted(track["agent_zone_windows_s"]),
            "slowdown_arrival_interval_s": _range(track["slowdown_arrivals_s"]),
            "speedup_arrival_interval_s": _range(track["speedup_arrivals_s"]),
        }
        for track in sorted(tracks.values(), key=lambda item: item["agent_track_id"])
    ]


def _diagnose(raw_scene: RawScene, source_index: int) -> dict[str, Any]:
    bundle = parse_scene(raw_scene)
    facts = SceneFactExtractor().extract(raw_scene)
    extractor = TrajectoryExtractor()
    ground_truth = extractor.extract(raw_scene)
    path = extractor.extract_path(raw_scene)
    augmentor = TrajectoryAugmentor()
    analyzer = TrajectoryAnalyzer()
    interaction_analyzer = AgentInteractionAnalyzer()

    def candidate(scale: float) -> Any:
        return _pedestrian_illegal_candidate_at_scale(
            speed_scale=scale,
            augmentor=augmentor,
            analyzer=analyzer,
            interaction_analyzer=interaction_analyzer,
            raw_scene=raw_scene,
            facts=facts,
            scenario_type=bundle.context.scenario_type,
            description=bundle.context.description,
            ground_truth=ground_truth,
            path_trajectory=path,
            drivable_area_polygons=[],
        )

    probe = candidate(1.0)
    slowdown_scales, speedup_scales = _pedestrian_timing_search_scales(probe)
    slowdown = [(scale, candidate(scale)) for scale in slowdown_scales]
    speedup = [(scale, candidate(scale)) for scale in speedup_scales]
    target = augmentor.pedestrian_illegal_gap_target_s
    slowdown_domain = _direction_domain(slowdown, target)
    speedup_domain = _direction_domain(speedup, target)
    target_reached = (
        slowdown_domain["violation_target_reached"]
        or speedup_domain["violation_target_reached"]
    )
    illegal_label = next(
        label
        for label in bundle.benchmark_labels.labels
        if label.variant_type == TrajectoryVariantType.ILLEGAL
    )

    if target_reached or illegal_label.expected_verdict == "vetoed":
        conclusion = "可注,已修复"
        basis = "bidirectional_search_reached_decisive_violation_target"
    elif (
        slowdown_domain["arrival_interval_s"] is None
        and speedup_domain["arrival_interval_s"] is None
    ):
        conclusion = "双向不可达,真不可注"
        basis = "no_shared_conflict_zone_in_either_direction"
    elif (
        slowdown_domain["gap_target_reached"]
        or speedup_domain["gap_target_reached"]
    ) and False not in {
        *slowdown_domain["stopped_before_zone_values"],
        *speedup_domain["stopped_before_zone_values"],
    }:
        conclusion = "双向不可达,真不可注"
        basis = "starts_inside_crosswalk_pre_stop_relation_unknown"
    else:
        conclusion = "双向不可达,真不可注"
        basis = "pedestrian_window_outside_bidirectional_arrival_domain"

    return {
        "record_index_zero_based": source_index,
        "frame_token": raw_scene.frame_token,
        "candidate_variant": "illegal",
        "candidate_anonymous_id": illegal_label.anonymous_id,
        "a_lat_max_mps2": augmentor.a_lat_max_mps2,
        "pedestrian_gap_target_s": target,
        "slowdown_domain": slowdown_domain,
        "speedup_domain_under_a_lat_limit": speedup_domain,
        "per_pedestrian_track": _per_track_domains(slowdown, speedup),
        "final_illegal_expected": illegal_label.expected_verdict,
        "final_illegal_reason": illegal_label.expected_verdict_reason,
        "diagnostic_basis": basis,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    failed_records = [
        record
        for record in preflight["records"]
        if any(
            label["variant"] == "illegal"
            and label["exclude_reason"] == "no_violation_injectable"
            for label in record["labels"]
        )
    ]
    target_tokens = {record["frame_token"] for record in failed_records}
    raw_records = json.loads(args.input.read_text(encoding="utf-8"))
    by_token = {
        record["frame_token"]: record
        for record in raw_records
        if record["frame_token"] in target_tokens
    }
    missing = target_tokens - by_token.keys()
    if missing:
        raise ValueError(f"prepared input 缺少 frame_token: {sorted(missing)}")

    source_diagnostics = {
        item["frame_token"]: item
        for item in preflight.get("pedestrian_timing_diagnostics", [])
    }
    source_has_bidirectional_evidence = all(
        "slowdown_domain" in source_diagnostics.get(token, {})
        and "speedup_domain" in source_diagnostics.get(token, {})
        for token in target_tokens
    )
    report = {
        "source_preflight": str(args.preflight),
        "source_no_violation_injectable_count": len(failed_records),
        "implementation_relative_to_source_preflight": (
            "implemented_before_preflight"
            if source_has_bidirectional_evidence
            else "not_provable_from_artifact"
        ),
        "timing_search_order": ["slowdown", "speedup_under_a_lat_limit"],
        "diagnostics": [
            _diagnose(
                RawScene.model_validate(by_token[record["frame_token"]]),
                record["index"],
            )
            for record in failed_records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "failed_candidates": len(report["diagnostics"]),
        "conclusions": [row["conclusion"] for row in report["diagnostics"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
