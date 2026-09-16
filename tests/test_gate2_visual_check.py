"""Gate 2 人工核对页中文事实句的回归测试。"""

from scripts.build_gate2_visual_check import _chinese_explanation
from src.layer1.models import ConflictZoneMetrics, TrajectoryFeatures


def _feature(*metrics: ConflictZoneMetrics) -> TrajectoryFeatures:
    return TrajectoryFeatures(
        max_speed_mps=4.0,
        min_speed_mps=1.0,
        mean_speed_mps=2.0,
        total_distance_m=10.0,
        max_lateral_offset_m=0.0,
        n_waypoints=3,
        conflict_zone_metrics=list(metrics),
    )


def test_pedestrian_explanation_selects_crosswalk_metric() -> None:
    stop_line = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.4,
        entered=False,
    )
    crosswalk = ConflictZoneMetrics(
        kind="ped_crossing",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        entry_time_s=0.32,
        stopped_before_zone=False,
    )

    explanation = _chinese_explanation(
        _feature(stop_line, crosswalk),
        "pedestrian",
        final_speed_mps=2.0,
    )

    assert "0.3 s 后进入前方人行横道区域" in explanation
    assert "停止线" not in explanation


def test_endpoint_inside_zone_is_distinguished_from_full_stop() -> None:
    stop_line = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        ended_inside=True,
        entry_time_s=0.65,
        stopped_before_zone=False,
    )

    explanation = _chinese_explanation(
        _feature(stop_line),
        "red_light",
        final_speed_mps=1.12,
    )

    assert "预测窗终点仍位于本向停止线区域" in explanation
    assert "末段速度 1.1 m/s" in explanation
    assert "末段仍在移动，并非完整停车" in explanation


def test_endpoint_inside_other_stop_line_proxy_is_annotated() -> None:
    stop_line = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        ended_inside_other_zone=True,
        entry_time_s=0.4,
        stopped_before_zone=False,
    )

    explanation = _chinese_explanation(
        _feature(stop_line),
        "red_light",
        final_speed_mps=0.88,
    )

    assert "预测窗终点位于另一停止线代理区域（非本向）" in explanation
    assert "末段速度 0.9 m/s" in explanation
    assert "末段仍在移动，并非完整停车" in explanation
