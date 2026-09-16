"""智能体交互的手算时间窗与数量上限。"""

from pathlib import Path

from src.layer1.agent_interaction import AgentInteractionAnalyzer
from src.layer1.models import (
    ConflictZone,
    RawScene,
    SceneFacts,
    Trajectory,
    TrajectorySource,
    Waypoint,
)


def _trajectory() -> Trajectory:
    return Trajectory(
        traj_id="internal",
        source=TrajectorySource.PLANNING,
        waypoints=[
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=5.0, y=0.0, t=2.5),
            Waypoint(x=9.0, y=0.0, t=4.5),
        ],
    )


def test_f1_f2_hand_calculated_windows() -> None:
    analyzer = AgentInteractionAnalyzer()
    assert analyzer._window_gap((4.5, 4.5), (1.0, 3.0)) == (1.5, "agent_first")
    gap, order = analyzer._window_gap((2.5, 2.5), (2.0, 5.0))
    assert gap < 0.0
    assert order == "temporal_overlap"


def test_f3_oncoming_paths_use_each_agents_arrival_time() -> None:
    interaction = AgentInteractionAnalyzer()._oncoming_interaction(
        _trajectory(),
        {"instance_token": "car-1"},
        [(0.0, 10.0, 0.0), (1.0, 5.0, 0.0), (2.0, 0.0, 0.0)],
    )
    assert interaction.ego_zone_arrival_s == 2.5
    assert interaction.agent_zone_window_s == (0.5, 1.5)
    assert interaction.crossing_order == "agent_first"
    assert interaction.min_time_gap_s == 0.5


def test_track_truncation_and_three_agent_limit() -> None:
    annotations = [
        {
            "instance_token": f"ped-{index}",
            "category_name": "human.pedestrian.adult",
            "track": [
                {"t": 1.0, "translation": [5.0, 0.0, 0.0]},
                {"t": 2.0, "translation": [5.0, 0.0, 0.0]},
            ],
        }
        for index in range(4)
    ]
    raw = RawScene(
        scene_id="scene",
        frame_token="frame",
        source_file=Path("/tmp/input.json"),
        ego_poses=[{
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": 0,
        }],
        annotations=annotations,
    )
    facts = SceneFacts(conflict_zones=[ConflictZone(
        kind="ped_crossing",
        polygon_xy=[(4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)],
        source_layer="ped_crossing",
    )])
    interactions = AgentInteractionAnalyzer().analyze(_trajectory(), raw, facts)
    assert len(interactions) == 3
    # Track 在 2s 消失，之后不外推，因此同时刻距离计算只覆盖 [1s, 2s]。
    assert all(item.agent_zone_window_s == (1.0, 2.0) for item in interactions)
