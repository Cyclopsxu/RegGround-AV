"""Layer 2 测试共享 fixture。"""

from pathlib import Path

import pytest

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery
from src.layer2.rule_graph import Layer2Settings, RuleGraph

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tiny_graph_path() -> Path:
    """测试用最小图谱 YAML 文件路径。"""
    return FIXTURE_DIR / "tiny_rule_graph.yaml"


@pytest.fixture
def tiny_settings(tiny_graph_path: Path) -> Layer2Settings:
    """指向测试用最小图谱的 Layer2Settings。"""
    return Layer2Settings(
        rule_graph_path=tiny_graph_path,
        enable_keyword_fallback=True,
    )


@pytest.fixture
def rule_graph(tiny_settings: Layer2Settings) -> RuleGraph:
    """已加载测试用最小图谱的 RuleGraph 实例。"""
    graph = RuleGraph(tiny_settings)
    graph.load()
    return graph


@pytest.fixture
def red_light_query() -> SceneQuery:
    """红灯场景查询。"""
    return SceneQuery(
        scene_id="scene-red-light-001",
        frame_token="frame-red",
        scenario_type=ScenarioType.RED_LIGHT,
        keywords=["red light", "stop line"],
        scene_facts=SceneFacts(has_red_light=True),
    )


@pytest.fixture
def pedestrian_query() -> SceneQuery:
    """行人场景查询。"""
    return SceneQuery(
        scene_id="scene-pedestrian-001",
        frame_token="frame-ped",
        scenario_type=ScenarioType.PEDESTRIAN,
        keywords=["pedestrian", "crosswalk"],
        scene_facts=SceneFacts(pedestrian_in_crosswalk=True),
    )


@pytest.fixture
def unknown_query() -> SceneQuery:
    """无命中场景查询。"""
    return SceneQuery(
        scene_id="scene-unknown-001",
        frame_token="frame-unknown",
        scenario_type=ScenarioType.UNKNOWN,
        keywords=["normal driving"],
        scene_facts=SceneFacts(),
    )
