"""SceneFacts v2 的方向、停止线与前向横道判定。"""

import math
from pathlib import Path

from src.layer1.models import ConflictZone, RawScene
from src.layer1.scene_fact_extractor import SceneFactExtractor


def _yaw_quaternion(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def test_f4_start_beyond_line_and_no_unrelated_fallback() -> None:
    aligned = ConflictZone(
        kind="stop_line",
        polygon_xy=[(-2.2, -2.0), (-1.8, -2.0), (-1.8, 2.0), (-2.2, 2.0)],
        source_layer="stop_line",
    )
    unrelated = ConflictZone(
        kind="stop_line",
        polygon_xy=[(4.0, 1.8), (8.0, 1.8), (8.0, 2.2), (4.0, 2.2)],
        source_layer="stop_line",
    )
    assert SceneFactExtractor._governing_stop_line_signed_m([aligned]) == -2.0
    assert SceneFactExtractor._governing_stop_line_signed_m([unrelated]) is None


def test_f5_cumulative_minus_sixty_degrees_is_right_turn() -> None:
    raw = RawScene(
        scene_id="scene",
        frame_token="frame",
        source_file=Path("/tmp/input.json"),
        ego_poses=[
            {"rotation": _yaw_quaternion(angle), "translation": [0.0, 0.0, 0.0]}
            for angle in (0.0, -20.0, -40.0, -60.0)
        ],
    )
    assert SceneFactExtractor._ego_turn_intent(raw) == "right"


def test_forward_path_buffer_rejects_distant_crosswalk() -> None:
    path = [(0.0, 0.0), (10.0, 0.0)]
    near = [(4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)]
    far = [(4.0, 5.0), (6.0, 5.0), (6.0, 7.0), (4.0, 7.0)]
    assert SceneFactExtractor._path_intersects_polygon_buffer(path, near, 3.0)
    assert not SceneFactExtractor._path_intersects_polygon_buffer(path, far, 3.0)


def test_nominal_six_second_sample_support_is_not_short() -> None:
    raw = RawScene(
        scene_id="scene",
        frame_token="frame",
        source_file=Path("/tmp/input.json"),
        ego_poses=[
            {
                "timestamp": index * 50_000,
                "translation": [float(index), 0.0, 0.0],
                "rotation": _yaw_quaternion(0.0),
            }
            for index in range(120)
        ],
    )
    assert SceneFactExtractor._window_insufficient(raw) is False
