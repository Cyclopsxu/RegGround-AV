"""离线 RawScene 构建器的数据完整性回归。"""

from input import build_map_index, collect_annotation_track, collect_map_records


def test_collect_annotation_track_follows_next_within_horizon() -> None:
    annotations = {
        "a0": {
            "token": "a0",
            "sample_token": "s0",
            "translation": [1.0, 2.0, 0.0],
            "next": "a1",
        },
        "a1": {
            "token": "a1",
            "sample_token": "s1",
            "translation": [2.0, 3.0, 0.0],
            "next": "a2",
        },
        "a2": {
            "token": "a2",
            "sample_token": "s2",
            "translation": [3.0, 4.0, 0.0],
            "next": "",
        },
    }
    samples = {
        "s0": {"timestamp": 1_000_000},
        "s1": {"timestamp": 2_000_000},
        "s2": {"timestamp": 8_000_000},
    }

    track = collect_annotation_track(
        annotations["a0"],
        annotations,
        samples,
        start_timestamp=1_000_000,
        future_horizon_s=6.0,
    )

    assert track == [
        {"t": 0.0, "translation": [1.0, 2.0]},
        {"t": 1.0, "translation": [2.0, 3.0]},
    ]


def test_drivable_area_follows_path_without_expanding_other_layers() -> None:
    polygons = {
        "contains_ego": [(-100.0, -100.0), (100.0, -100.0), (100.0, 100.0), (-100.0, 100.0)],
        "path_only": [(65.0, -5.0), (75.0, -5.0), (75.0, 5.0), (65.0, 5.0)],
    }
    nodes: list[dict] = []
    polygon_rows: list[dict] = []
    for polygon_token, points in polygons.items():
        node_tokens = []
        for index, (x, y) in enumerate(points):
            node_token = f"{polygon_token}_{index}"
            node_tokens.append(node_token)
            nodes.append({"token": node_token, "x": x, "y": y})
        polygon_rows.append({
            "token": polygon_token,
            "exterior_node_tokens": node_tokens,
        })
    map_index = build_map_index({
        "node": nodes,
        "polygon": polygon_rows,
        "drivable_area": [{
            "token": "drive",
            "polygon_tokens": ["contains_ego", "path_only"],
        }],
        "stop_line": [{"token": "far_stop", "polygon_token": "path_only"}],
    })

    records = collect_map_records(
        [{"translation": [0.0, 0.0, 0.0]}],
        map_index,
        50.0,
        path_ego_poses=[
            {"translation": [0.0, 0.0, 0.0]},
            {"translation": [70.0, 0.0, 0.0]},
        ],
    )

    assert len(records["drivable_area"][0]["polygon_xy_list"]) == 2
    assert "stop_line" not in records
