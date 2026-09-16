"""共享规则谓词的 v2 边界回归。"""

from src.compliance_predicates import (
    PredicateDecision,
    pedestrian_yield_verdict,
    red_light_crossing_verdict,
)
from src.layer1.models import (
    AgentInteraction,
    ConflictZoneMetrics,
    SceneFactsDigest,
    TrajectoryFeatures,
)


def _digest(**updates: object) -> SceneFactsDigest:
    values = {
        "signal_phase": "red",
        "phase_source": "drivelm_status",
        "ego_turn_intent": "straight",
        "governing_stop_line_signed_m": 5.0,
        "pedestrian_in_forward_crosswalk": False,
        "pedestrian_moving": None,
        "oncoming_vehicle_moving": False,
        "location_is_intersection": True,
    }
    values.update(updates)
    return SceneFactsDigest(**values)  # type: ignore[arg-type]


def _feature(metric: ConflictZoneMetrics, *, full_stop: bool = False) -> TrajectoryFeatures:
    return TrajectoryFeatures(
        max_speed_mps=4.0,
        min_speed_mps=0.0,
        mean_speed_mps=2.0,
        total_distance_m=10.0,
        max_lateral_offset_m=0.0,
        n_waypoints=3,
        full_stop=full_stop,
        stopped_before_zone=metric.stopped_before_zone,
        conflict_zone_metrics=[metric],
    )


def test_red_predicate_covers_trigger_exemptions_and_phase_horizon() -> None:
    crossing = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        entry_time_s=1.5,
        penetration_depth_m=0.5,
        stopped_before_zone=False,
    )
    feature = _feature(crossing)

    assert red_light_crossing_verdict(
        feature, _digest(governing_stop_line_signed_m=None)
    ).reason == "no_governing_signal_or_line"
    beyond = red_light_crossing_verdict(
        feature, _digest(governing_stop_line_signed_m=-2.0)
    )
    assert beyond.decision == PredicateDecision.NOT_DECIDABLE
    assert red_light_crossing_verdict(
        feature, _digest(ego_turn_intent="right")
    ).reason == "right_turn_on_red_exemption"
    assert red_light_crossing_verdict(feature, _digest()).decision == (
        PredicateDecision.DECISIVE_VETO
    )
    assert red_light_crossing_verdict(
        feature, _digest(), phase_validity_horizon_s=1.0
    ).decision == PredicateDecision.NOT_DECIDABLE

    shallow_crossing = crossing.model_copy(update={"penetration_depth_m": 0.10})
    assert red_light_crossing_verdict(
        _feature(shallow_crossing), _digest()
    ).decision == PredicateDecision.NOT_DECIDABLE

    stopped = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.5,
        entered=False,
        stopped_before_zone=True,
    )
    assert red_light_crossing_verdict(
        _feature(stopped, full_stop=True), _digest()
    ).decision == PredicateDecision.DECISIVE_CLEAR


def test_pedestrian_predicate_uses_interaction_and_prior_stop() -> None:
    crossing = ConflictZoneMetrics(
        kind="ped_crossing",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        entry_time_s=2.5,
        stopped_before_zone=False,
    )
    feature = _feature(crossing).model_copy(update={
        "agent_interactions": [AgentInteraction(
            agent_kind="pedestrian",
            agent_track_id="ped-1",
            ego_zone_arrival_s=2.5,
            agent_zone_window_s=(2.0, 5.0),
            min_time_gap_s=-1.0,
            min_spatiotemporal_gap_m=1.0,
            crossing_order="temporal_overlap",
        )]
    })
    digest = _digest(
        pedestrian_in_forward_crosswalk=True,
        pedestrian_moving=True,
    )
    assert pedestrian_yield_verdict(feature, digest).decision == (
        PredicateDecision.DECISIVE_VETO
    )

    zero_gap = feature.model_copy(update={
        "agent_interactions": [feature.agent_interactions[0].model_copy(update={
            "min_time_gap_s": -0.0,
        })]
    })
    assert pedestrian_yield_verdict(zero_gap, digest).decision == (
        PredicateDecision.DECISIVE_VETO
    )

    starts_inside = zero_gap.model_copy(update={
        "conflict_zone_metrics": [crossing.model_copy(update={
            "entry_time_s": 0.0,
            "stopped_before_zone": None,
        })]
    })
    assert pedestrian_yield_verdict(starts_inside, digest).decision == (
        PredicateDecision.NOT_DECIDABLE
    )

    clear_feature = feature.model_copy(update={
        "agent_interactions": [feature.agent_interactions[0].model_copy(update={
            "min_time_gap_s": 3.5,
            "crossing_order": "agent_first",
        })]
    })
    assert pedestrian_yield_verdict(clear_feature, digest).decision == (
        PredicateDecision.DECISIVE_CLEAR
    )
