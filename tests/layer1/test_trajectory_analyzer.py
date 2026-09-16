"""轨迹特征分析器测试。

使用合成 Trajectory fixture 验证 TrajectoryAnalyzer 各方法，
不依赖真实 nuScenes 数据。
"""

from __future__ import annotations

import pytest

from src.layer1.models import (
    ConflictZone,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectorySource,
    Waypoint,
)
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer

# =============================================================================
# 合成 fixture 构造辅助
# =============================================================================


def _make_trajectory(
    n_waypoints: int = 10,
    speed_mps: float = 5.0,
    dt: float = 0.5,
    lateral_offset: float = 0.0,
) -> Trajectory:
    """构造匀速直线（或小幅偏移）合成轨迹。

    Args:
        n_waypoints: 航点数量。
        speed_mps: 匀速速度 (m/s)。
        dt: 航点间时间间隔 (s)。
        lateral_offset: 每步横向偏移增量 (m)。
    """
    wps: list[Waypoint] = []
    for i in range(n_waypoints):
        t = float(i) * dt
        x = float(i) * speed_mps * dt
        y = float(i) * lateral_offset
        wps.append(Waypoint(x=x, y=y, t=t))
    return Trajectory(
        traj_id="test_traj",
        source=TrajectorySource.SYNTHETIC,
        waypoints=wps,
    )


def _make_deceleration_trajectory(
    n_waypoints: int = 10,
    initial_speed: float = 10.0,
    final_speed: float = 0.0,
    dt: float = 0.5,
) -> Trajectory:
    """构造线性减速轨迹。

    Args:
        n_waypoints: 航点数量。
        initial_speed: 初始速度 (m/s)。
        final_speed: 最终速度 (m/s)。
        dt: 时间间隔 (s)。
    """
    wps: list[Waypoint] = []
    x = 0.0
    for i in range(n_waypoints):
        t = float(i) * dt
        # 线性减速
        current_speed = initial_speed + (final_speed - initial_speed) * i / (n_waypoints - 1)
        x += current_speed * dt
        wps.append(Waypoint(x=x, y=0.0, t=t))
    return Trajectory(
        traj_id="decel_traj",
        source=TrajectorySource.SYNTHETIC,
        waypoints=wps,
    )


def _make_default_scene() -> SceneDescription:
    """构造默认场景描述。"""
    return SceneDescription(
        narrative="测试场景",
        keywords=["测试"],
    )


# =============================================================================
# 基本行为测试
# =============================================================================


class TestAnalyzeBasic:
    """TrajectoryAnalyzer.analyze 基本行为测试。"""

    def test_constant_speed_correct(self):
        """匀速直线轨迹：速度特征数值正确。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, speed_mps=5.0, dt=0.5)
        features = analyzer.analyze(traj, _make_default_scene())

        # 匀速 5 m/s
        assert features.mean_speed_mps == pytest.approx(5.0, rel=0.01)
        assert features.max_speed_mps == pytest.approx(5.0, rel=0.01)
        assert features.min_speed_mps == pytest.approx(5.0, rel=0.01)

    def test_total_distance_correct(self):
        """总行程数值正确。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, speed_mps=5.0, dt=0.5)
        features = analyzer.analyze(traj, _make_default_scene())

        # 9 段 × 5 m/s × 0.5 s = 22.5 m
        expected_distance = 9 * 5.0 * 0.5
        assert features.total_distance_m == pytest.approx(expected_distance, rel=0.01)

    def test_n_waypoints_correct(self):
        """n_waypoints 应与输入一致。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=15)
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.n_waypoints == 15

    def test_max_lateral_offset_zero_for_straight(self):
        """直线轨迹最大横向偏移为 0。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, lateral_offset=0.0)
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.max_lateral_offset_m == pytest.approx(0.0, abs=0.01)

    def test_max_lateral_offset_detected(self):
        """含横向偏移的轨迹应检测到非零 max_lateral_offset。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, lateral_offset=0.2)
        features = analyzer.analyze(traj, _make_default_scene())
        # 终点的 y = 9 * 0.2 = 1.8 m
        assert features.max_lateral_offset_m > 1.0

    def test_deceleration_detected(self):
        """减速轨迹应检测到最低速度 < 平均速度。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_deceleration_trajectory(
            n_waypoints=10, initial_speed=10.0, final_speed=0.0,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.min_speed_mps < features.mean_speed_mps
        assert features.max_speed_mps > features.min_speed_mps

    def test_two_waypoints_minimal_works(self):
        """恰好 2 个 waypoints 时正常分析（最小值）。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=2)
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.n_waypoints == 2
        assert features.mean_speed_mps == pytest.approx(5.0)

    def test_summary_not_empty(self):
        """natural_language_summary 不应为空。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10)
        features = analyzer.analyze(traj, _make_default_scene())
        assert len(features.natural_language_summary) > 10

    def test_conflict_zone_behavior_not_empty(self):
        """conflict_zone_behavior 不应为空。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10)
        features = analyzer.analyze(traj, _make_default_scene())
        assert len(features.conflict_zone_behavior) > 0

    def test_conflict_zone_features_use_polygon_geometry(self):
        """冲突区特征应来自真实多边形，而不是轨迹后半段近似。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, speed_mps=2.0, dt=0.5)
        facts = SceneFacts(conflict_zones=[
            ConflictZone(
                kind="ped_crossing",
                polygon_xy=[(3.5, -1.0), (5.5, -1.0), (5.5, 1.0), (3.5, 1.0)],
                source_layer="ped_crossing",
            ),
        ])

        features = analyzer.analyze(traj, _make_default_scene(), facts)

        assert features.entered_conflict_zone is True
        assert features.min_distance_to_conflict_m == pytest.approx(0.0)
        assert features.min_speed_in_zone_mps == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("polygon_x", "expected"),
        [((9.0, 10.0), True), ((3.5, 5.5), False)],
    )
    def test_conflict_zone_metric_records_whether_endpoint_is_inside(
        self,
        polygon_x: tuple[float, float],
        expected: bool,
    ) -> None:
        """区分“轨迹曾进入区域”和“预测终点仍在区域”。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, speed_mps=2.0, dt=0.5)
        x_min, x_max = polygon_x
        facts = SceneFacts(conflict_zones=[
            ConflictZone(
                kind="ped_crossing",
                polygon_xy=[
                    (x_min, -1.0),
                    (x_max, -1.0),
                    (x_max, 1.0),
                    (x_min, 1.0),
                ],
                source_layer="ped_crossing",
            ),
        ])

        features = analyzer.analyze(traj, _make_default_scene(), facts)

        metric = features.conflict_zone_metrics[0]
        assert metric.entered is True
        assert metric.ended_inside is expected

    def test_endpoint_in_non_governing_stop_line_zone_is_recorded(self) -> None:
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, speed_mps=2.0, dt=0.5)
        facts = SceneFacts(
            governing_stop_line_signed_m=1.0,
            conflict_zones=[
                ConflictZone(
                    kind="stop_line",
                    polygon_xy=[
                        (0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (0.5, 1.0),
                    ],
                    source_layer="stop_line",
                ),
                ConflictZone(
                    kind="stop_line",
                    polygon_xy=[
                        (8.5, -1.0), (9.5, -1.0), (9.5, 1.0), (8.5, 1.0),
                    ],
                    source_layer="stop_line",
                ),
            ],
        )

        features = analyzer.analyze(traj, _make_default_scene(), facts)

        metric = features.conflict_zone_metrics[0]
        assert metric.governing is True
        assert metric.ended_inside is False
        assert metric.ended_inside_other_zone is True


# =============================================================================
# 合规词检查测试（§4.4 / §5.5 —— 强制）
# =============================================================================


class TestBannedWords:
    """natural_language_summary 合规词隔离测试。"""

    BANNED = ["违规", "违反", "合规", "非法", "应该", "必须"]

    def test_no_banned_words_in_summary_straight(self):
        """匀速直线轨迹 summary 不含禁止词。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10)
        features = analyzer.analyze(traj, _make_default_scene())
        for word in self.BANNED:
            assert word not in features.natural_language_summary, (
                f"禁止词 '{word}' 出现在 summary 中: {features.natural_language_summary}"
            )

    def test_no_banned_words_in_summary_deceleration(self):
        """减速轨迹 summary 不含禁止词。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_deceleration_trajectory(n_waypoints=10)
        features = analyzer.analyze(traj, _make_default_scene())
        for word in self.BANNED:
            assert word not in features.natural_language_summary, (
                f"禁止词 '{word}' 出现在 summary 中: {features.natural_language_summary}"
            )

    def test_no_banned_words_in_summary_lateral(self):
        """横向偏移轨迹 summary 不含禁止词。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_trajectory(n_waypoints=10, lateral_offset=0.3)
        features = analyzer.analyze(traj, _make_default_scene())
        for word in self.BANNED:
            assert word not in features.natural_language_summary, (
                f"禁止词 '{word}' 出现在 summary 中: {features.natural_language_summary}"
            )

    def test_no_banned_words_in_conflict_zone_behavior(self):
        """conflict_zone_behavior 不含禁止词。"""
        analyzer = TrajectoryAnalyzer()
        traj = _make_deceleration_trajectory(n_waypoints=20)
        features = analyzer.analyze(traj, _make_default_scene())
        for word in self.BANNED:
            assert word not in features.conflict_zone_behavior, (
                f"禁止词 '{word}' 出现在 conflict_zone_behavior 中"
            )

    def test_check_banned_words_utility(self):
        """check_banned_words 静态方法正确检测禁止词。"""
        assert TrajectoryAnalyzer.check_banned_words("正常描述") == []
        assert set(TrajectoryAnalyzer.check_banned_words("这违反了规则")) == {"违反"}
        assert set(TrajectoryAnalyzer.check_banned_words("应该停车 违规")) == {"应该", "违规"}


# =============================================================================
# 数值正确性详细测试
# =============================================================================


class TestNumericalCorrectness:
    """速度/距离数值精确性验证。"""

    def test_speed_calculation_precise(self):
        """已知坐标与时间 → 验证速度计算精度。"""
        analyzer = TrajectoryAnalyzer()
        # 手动构造：x = [0, 3, 6], t = [0, 1, 2]
        # speed = [3, 3] m/s
        wps = [
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=3.0, y=0.0, t=1.0),
            Waypoint(x=6.0, y=0.0, t=2.0),
        ]
        traj = Trajectory(
            traj_id="precise",
            source=TrajectorySource.SYNTHETIC,
            waypoints=wps,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.max_speed_mps == pytest.approx(3.0)
        assert features.mean_speed_mps == pytest.approx(3.0)

    def test_two_waypoints_works(self):
        """2 个 waypoint 时也能正常分析。"""
        analyzer = TrajectoryAnalyzer()
        wps = [
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=5.0, y=0.0, t=1.0),
        ]
        traj = Trajectory(
            traj_id="two",
            source=TrajectorySource.SYNTHETIC,
            waypoints=wps,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.n_waypoints == 2
        assert features.mean_speed_mps == pytest.approx(5.0)

    def test_lateral_distance_not_affecting_longitudinal(self):
        """横向偏移不影响纵向行程计算。"""
        analyzer = TrajectoryAnalyzer()
        # 曲折轨迹：总行程 > 纵向跨度
        wps = [
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=3.0, y=4.0, t=1.0),  # 对角线 5 m
        ]
        traj = Trajectory(
            traj_id="diagonal",
            source=TrajectorySource.SYNTHETIC,
            waypoints=wps,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        # 总行程应为 5 m（3-4-5 三角形）
        assert features.total_distance_m == pytest.approx(5.0)

    def test_stationary_trajectory(self):
        """静止轨迹所有速度为 0。"""
        analyzer = TrajectoryAnalyzer()
        wps = [
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=0.0, y=0.0, t=1.0),
            Waypoint(x=0.0, y=0.0, t=2.0),
        ]
        traj = Trajectory(
            traj_id="stationary",
            source=TrajectorySource.SYNTHETIC,
            waypoints=wps,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        assert features.max_speed_mps == pytest.approx(0.0)
        assert features.min_speed_mps == pytest.approx(0.0)
        assert features.total_distance_m == pytest.approx(0.0)

    def test_duplicate_timestamps_no_speed_explosion(self):
        """相邻 waypoint 时间戳相同时，速度不应爆炸（应为 0，dt==0 段被忽略）。"""
        analyzer = TrajectoryAnalyzer()
        # 两个 waypoint 位置不同但时间戳相同 → dt==0
        wps = [
            Waypoint(x=0.0, y=0.0, t=1.0),
            Waypoint(x=5.0, y=0.0, t=1.0),  # 相同时间戳！
            Waypoint(x=10.0, y=0.0, t=2.0),
        ]
        traj = Trajectory(
            traj_id="dup_time",
            source=TrajectorySource.SYNTHETIC,
            waypoints=wps,
        )
        features = analyzer.analyze(traj, _make_default_scene())
        # dt==0 的速度段被标记为 NaN 并过滤，不应产生虚高速度
        assert features.max_speed_mps < 100.0, (
            f"速度爆炸: max_speed={features.max_speed_mps}"
        )
        # 有效的速度段（t=1.0→2.0，位移 5m，dt=1s）= 5 m/s
        assert features.mean_speed_mps == pytest.approx(5.0)
