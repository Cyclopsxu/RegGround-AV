"""availability spike 统计与报告测试。"""

from src.eval_set import EvalSetEntry
from src.layer1.availability_spike import (
    AvailabilityStats,
    CountDistribution,
    _distribution,
    render_spike_report,
    select_representative_frames,
)


def test_distribution_reports_discrete_p95() -> None:
    distribution = _distribution([1, 2, 3, 4, 100])

    assert distribution.minimum == 1
    assert distribution.median == 3.0
    assert distribution.p95 == 100
    assert distribution.maximum == 100


def test_spike_report_contains_admitted_scene_counts_and_metadata() -> None:
    stats = AvailabilityStats(
        boston_drivelm_keyframes=30,
        total_keyframes=30,
        unique_scenes=6,
        location_keyframes={"boston-seaport": 30, "singapore-onenorth": 20},
        scenario_counts={"red_light": 25, "unknown": 5},
        scenario_unique_scene_counts={"red_light": 1, "unknown": 5},
        evaluable_scenario_counts={"red_light": 18},
        admitted_scenario_unique_scene_counts={"red_light": 1},
        admitted_pair_count=2,
        mini_admitted_pair_count=1,
        traffic_status_sources={"drivelm_status": 25, "none": 5},
        admitted_keyframes=25,
        admitted_unique_scenes=4,
        stationary_admitted_keyframes=7,
        qa_pair_counts=CountDistribution(minimum=10, median=20.0, p95=30, maximum=40),
        key_object_counts=CountDistribution(minimum=1, median=2.0, p95=4, maximum=5),
        selected_eval_entries=[EvalSetEntry(
            frame_token="frame_1",
            scene_token="scene_1",
            scenario_type="red_light",
            horizon_displacement_m=9.5,
            timestamp=1,
        )],
        notes=["red_light 非静止可评估样本 < 20；按冻结的薄层处置执行。"],
    )

    report = render_spike_report(stats)

    assert "| red_light | 25 | 1 | 18 | 1 |" in report
    assert "配对总量：2" in report
    assert "mini 涉及的准入配对数：1" in report
    assert "| none | 5 | 16.7% |" in report
    assert "| frame_1 | scene_1 | red_light | 9.500 |" in report
    assert "singapore-onenorth" in report
    assert "10/20.0/30/40" in report
    assert "薄层处置执行" in report
    assert "准入后 n=12" in report
    assert report.index("## 决策提示") < report.index("## 正式评估集候选")
    assert "## 正式评估集（已冻结）" in render_spike_report(
        stats,
        eval_set_state="frozen",
    )
    assert "输入基线：prepared input is the formal-run baseline" in render_spike_report(
        stats,
        input_provenance="prepared input is the formal-run baseline",
        determinism_verified=True,
    )
    assert "确定性验证：两次独立生成" in render_spike_report(
        stats,
        determinism_verified=True,
    )


def test_representative_frame_selection_is_deterministic_and_scene_level() -> None:
    frames = [
        EvalSetEntry(
            frame_token="frame_2",
            scene_token="scene_a",
            scenario_type="red_light",
            horizon_displacement_m=8.0,
            timestamp=2,
        ),
        EvalSetEntry(
            frame_token="frame_1",
            scene_token="scene_a",
            scenario_type="red_light",
            horizon_displacement_m=8.0,
            timestamp=1,
        ),
        EvalSetEntry(
            frame_token="frame_3",
            scene_token="scene_b",
            scenario_type="red_light",
            horizon_displacement_m=9.0,
            timestamp=3,
        ),
    ]

    first = select_representative_frames(frames)
    second = select_representative_frames(list(reversed(frames)))

    assert first == second
    assert len(first) == 2
    assert [entry.frame_token for entry in first] == ["frame_1", "frame_3"]
