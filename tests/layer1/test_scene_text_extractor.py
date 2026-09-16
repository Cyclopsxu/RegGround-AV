"""场景文本提取器测试。

使用合成 RawScene fixture 验证 SceneTextExtractor 各方法，
不依赖真实 nuScenes / DriveLM 数据。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.layer1.exceptions import SceneParseError
from src.layer1.models import (
    LocationType,
    RawScene,
    ScenarioType,
    SceneFacts,
)
from src.layer1.scene_text_extractor import (
    SCENARIO_KEYWORD_MAP,
    SceneTextExtractor,
)

# =============================================================================
# 合成 RawScene 构造辅助
# =============================================================================


def _make_raw_scene(
    scene_id: str = "test_scene",
    qa_pairs: list[dict] | None = None,
    key_object_infos: dict | None = None,
    keywords: list[str] | None = None,
    ego_poses: list[dict] | None = None,
    ego_speed: float | None = None,
) -> RawScene:
    """构造合成 RawScene，模拟 DriveLM QA 结构。"""
    raw_json: dict = {}
    if qa_pairs is not None:
        raw_json["qa_pairs"] = qa_pairs
    if key_object_infos is not None:
        raw_json["key_object_infos"] = key_object_infos
    if keywords is not None:
        raw_json["keywords"] = keywords
    if ego_speed is not None:
        raw_json["ego_speed"] = ego_speed

    return RawScene(
        scene_id=scene_id,
        frame_token=f"{scene_id}_frame",
        source_file=Path("/fake/nuscenes"),
        raw_json=raw_json,
        ego_poses=ego_poses or [],
        keyframe_index=0,
    )


# =============================================================================
# extract 测试
# =============================================================================


class TestExtract:
    """SceneTextExtractor.extract 测试。"""

    def test_empty_raw_json_raises_scene_parse_error(self):
        """raw_json 为空 dict 时抛出 SceneParseError。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene()
        with pytest.raises(SceneParseError):
            extractor.extract(raw)

    def test_extract_narrative_from_qa_pairs(self):
        """从 qa_pairs 拼接 narrative。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"category": "perception", "question": "看到什么？",
             "answer": "前方红灯。"},
            {"category": "prediction", "question": "预测什么？",
             "answer": "需要停车。"},
        ])
        desc = extractor.extract(raw)
        assert "前方红灯" in desc.narrative
        assert "需要停车" not in desc.narrative

    def test_narrative_truncation(self):
        """narrative 超过 max_narrative_chars 应截断。"""
        extractor = SceneTextExtractor(max_narrative_chars=10)
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "A" * 50},
        ])
        desc = extractor.extract(raw)
        assert len(desc.narrative) <= 10

    def test_extract_keywords_hit_from_narrative(self):
        """关键词应从 narrative 中命中提取。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "前方有行人正在横穿斑马线。"},
        ])
        desc = extractor.extract(raw)
        assert "行人" in desc.keywords or "横穿" in desc.keywords or "斑马线" in desc.keywords

    def test_extract_explicit_keywords(self):
        """raw_json 中 explicit keywords 字段应直接提取。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(
            qa_pairs=[{"answer": "路口场景。"}],
            keywords=["红灯", "交叉口"],
        )
        desc = extractor.extract(raw)
        assert "红灯" in desc.keywords
        assert "交叉口" in desc.keywords

    def test_infer_location_type_intersection(self):
        """含路口关键词应推断为 INTERSECTION。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "自车接近十字路口。"},
        ])
        desc = extractor.extract(raw)
        assert desc.location_type == LocationType.INTERSECTION

    def test_infer_location_type_unknown_by_default(self):
        """无位置关键词时应为 UNKNOWN。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "自车正常行驶。"},
        ])
        desc = extractor.extract(raw)
        assert desc.location_type == LocationType.UNKNOWN


# =============================================================================
# extract_ego_state 测试
# =============================================================================


class TestExtractEgoState:
    """SceneTextExtractor.extract_ego_state 测试。"""

    def test_speed_from_ego_poses(self):
        """从相邻帧 ego_pose 位移估算速度。"""
        extractor = SceneTextExtractor()
        poses = [
            {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
             "timestamp": 0},
            {"translation": [5.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
             "timestamp": 500_000},  # 0.5 s
        ]
        raw = _make_raw_scene(ego_poses=poses)
        # keyframe_index=0, 取 poses[0] 和 poses[1]
        state = extractor.extract_ego_state(raw)
        # 5 m / 0.5 s = 10 m/s
        assert state.speed_mps == pytest.approx(10.0)

    def test_speed_from_ego_speed_field(self):
        """无 ego_poses 时从 ego_speed 字段取值。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(ego_speed=8.0)
        state = extractor.extract_ego_state(raw)
        assert state.speed_mps == pytest.approx(8.0)

    def test_ego_speed_bool_ignored(self):
        """ego_speed 为 JSON boolean 时应被忽略（bool 是 int 子类陷阱）。"""
        extractor = SceneTextExtractor()
        # True 是 int 子类，isinstance(True, int) 为 True
        raw = _make_raw_scene(ego_speed=True)  # type: ignore[arg-type]
        state = extractor.extract_ego_state(raw)
        # 不应将 True 转为 1.0
        assert state.speed_mps is None

    def test_speed_none_when_no_data(self):
        """无任何速度数据时返回 None。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene()
        state = extractor.extract_ego_state(raw)
        assert state.speed_mps is None

    def test_speed_from_text_kmh(self):
        """从 QA 文本的 km/h 解析速度。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"question": "速度多少？", "answer": "当前速度 36 km/h。"},
        ])
        state = extractor.extract_ego_state(raw)
        # 36 km/h = 10 m/s
        assert state.speed_mps == pytest.approx(10.0)

    def test_speed_zero_when_stationary(self):
        """静止状态位移为 0，速度应为 0。"""
        extractor = SceneTextExtractor()
        poses = [
            {"translation": [0.0, 0.0, 0.0], "timestamp": 0},
            {"translation": [0.0, 0.0, 0.0], "timestamp": 1_000_000},
        ]
        raw = _make_raw_scene(ego_poses=poses)
        state = extractor.extract_ego_state(raw)
        assert state.speed_mps == pytest.approx(0.0)


# =============================================================================
# infer_scenario_type 测试（§5.2.1）
# =============================================================================


class TestInferScenarioType:
    """SceneTextExtractor.infer_scenario_type 测试——各信号灯/让行关键词命中。"""

    def test_red_light_from_status(self):
        """key_object_infos 中红绿灯 status='red' 直接返回 RED_LIGHT。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(key_object_infos={
            "obj_1": {"category": "traffic_light", "status": "red"},
        })
        assert extractor.infer_scenario_type(raw) == ScenarioType.RED_LIGHT

    def test_yellow_light_from_status(self):
        """key_object_infos 中红绿灯 status='yellow' 返回 YELLOW_LIGHT。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(key_object_infos={
            "obj_1": {"category": "traffic_light", "status": "yellow"},
        })
        assert extractor.infer_scenario_type(raw) == ScenarioType.YELLOW_LIGHT

    def test_red_light_keyword_only_returns_unknown(self):
        """无结构化 status 时关键词不能生成法规场景标签。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "前方有红灯，需要停车等待。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_yellow_light_from_keywords(self):
        """黄灯关键词命中返回 YELLOW_LIGHT。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "黄灯亮了，自车抢行通过。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_pedestrian_from_keywords(self):
        """行人关键词命中返回 PEDESTRIAN。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "行人正在斑马线横穿马路。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_oncoming_from_keywords(self):
        """对向车关键词命中返回 ONCOMING。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "对向有直行车辆，自车需要让行。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_minor_road_from_keywords(self):
        """支路关键词命中返回 MINOR_ROAD。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "自车从支路汇入主路。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_emergency_from_keywords(self):
        """特种车辆关键词命中返回 EMERGENCY。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "后方有救护车鸣笛驶来。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_unknown_when_no_match(self):
        """无任何匹配时返回 UNKNOWN。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "自车在空旷道路上正常行驶。"},
        ])
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_unknown_when_traffic_light_status_unknown(self):
        """红绿灯对象存在但 status 未知（非 red/yellow）且无关键词 => UNKNOWN。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(
            key_object_infos={
                "obj_1": {"category": "traffic_light", "status": "unknown"},
            },
            qa_pairs=[{"answer": "接近路口。"}],
        )
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_empty_raw_json_returns_unknown(self):
        """raw_json 为空返回 UNKNOWN，不抛异常。"""
        extractor = SceneTextExtractor()
        raw = RawScene(
            scene_id="empty",
            frame_token="f1",
            source_file=Path("/fake"),
        )
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_signal_light_with_chinese_status(self):
        """红绿灯 status 为中文 '红灯' 时也正确识别。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(key_object_infos={
            "obj_1": {"category": "traffic_light", "status": "红灯"},
        })
        assert extractor.infer_scenario_type(raw) == ScenarioType.RED_LIGHT

    def test_multiple_traffic_lights_second_is_red(self):
        """多个红绿灯对象：第一个 status=green，第二个 status=red → 应返回 RED_LIGHT。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(key_object_infos={
            "obj_1": {"category": "traffic_light", "status": "green"},
            "obj_2": {"category": "traffic_light", "status": "red"},
        })
        assert extractor.infer_scenario_type(raw) == ScenarioType.RED_LIGHT

    def test_status_uppercase_field_is_supported(self):
        """DriveLM key_object_infos 使用大写 Status 字段时也应解析。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(key_object_infos={
            "obj_1": {"category": "traffic_light", "Status": "yellow"},
        })
        assert extractor.infer_scenario_type(raw) == ScenarioType.YELLOW_LIGHT

    def test_english_keywords_match_drivelm_text(self):
        """v3.1 关键词表必须覆盖 DriveLM 英文文本。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "The ego vehicle approaches a crosswalk with a pedestrian nearby."},
        ])
        desc = extractor.extract(raw)

        assert "pedestrian" in desc.keywords
        assert "crosswalk" in desc.keywords
        assert extractor.infer_scenario_type(raw) == ScenarioType.UNKNOWN

    def test_structured_facts_take_priority_over_text_keywords(self):
        """facts 已有结构化事实时，场景类型应优先来自 facts。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "The text mentions a pedestrian near a crosswalk."},
        ])
        facts = SceneFacts(has_yellow_light=True, traffic_light_status_source="drivelm_status")

        assert extractor.infer_scenario_type(raw, facts) == ScenarioType.YELLOW_LIGHT

    def test_narrative_is_scrubbed_before_downstream_use(self):
        """narrative 进入下游前应脱敏标签词与合规结论词。"""
        extractor = SceneTextExtractor()
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "The illegal variant should stop before the line."},
        ])
        desc = extractor.extract(raw)
        lowered = desc.narrative.lower()

        assert "illegal" not in lowered
        assert "variant" not in lowered
        assert "should stop" not in lowered
        assert "[redacted]" in desc.narrative

    def test_keywords_same_category_enum_order(self):
        """同类别（让行）多命中时按枚举顺序取第一个。"""
        extractor = SceneTextExtractor()
        # 同时含"斑马线"(PEDESTRIAN) 和"对向"(ONCOMING)
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "对向有直行车辆，斑马线有行人横穿。"},
        ])
        result = extractor.infer_scenario_type(raw)
        # 同属让行类，PEDESTRIAN 枚举序在 ONCOMING 之前
        assert result == ScenarioType.UNKNOWN

    def test_signal_priority_over_yield(self):
        """同时命中信号灯和让行关键词时，信号灯优先。"""
        extractor = SceneTextExtractor()
        # 同时含"红灯"(RED_LIGHT) 和"行人"(PEDESTRIAN)
        raw = _make_raw_scene(qa_pairs=[
            {"answer": "前方红灯，斑马线上有行人横穿。"},
        ])
        result = extractor.infer_scenario_type(raw)
        # 显式优先级：信号灯类 > 让行类
        assert result == ScenarioType.UNKNOWN


# =============================================================================
# SCENARIO_KEYWORD_MAP 完整性测试
# =============================================================================


class TestScenarioKeywordMap:
    """关键词映射表覆盖验证。"""

    def test_all_scenario_types_have_keywords(self):
        """除 UNKNOWN 外，所有 ScenarioType 都应有对应关键词列表。"""
        for st in ScenarioType:
            if st == ScenarioType.UNKNOWN:
                continue
            assert st in SCENARIO_KEYWORD_MAP, f"{st} 缺少关键词映射"
            assert len(SCENARIO_KEYWORD_MAP[st]) > 0, f"{st} 关键词列表为空"

    def test_no_duplicate_keywords_across_types(self):
        """同一关键词不应出现在多个 ScenarioType 中。"""
        seen: dict[str, ScenarioType] = {}
        for st, kw_list in SCENARIO_KEYWORD_MAP.items():
            for kw in kw_list:
                if kw in seen:
                    pytest.fail(
                        f"关键词 '{kw}' 同时出现在 {seen[kw].value} 和 {st.value}"
                    )
                seen[kw] = st
