"""Layer 2 集成测试。"""

import pytest

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery
from src.layer2.models import RetrievalMode, RuleStatus
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings


@pytest.fixture(scope="module")
def mvp_retriever() -> GraphRAGRetriever:
    """加载 MVP 图的 GraphRAGRetriever（模块级复用）。"""
    return GraphRAGRetriever(Layer2Settings())


def make_query(**overrides: object) -> SceneQuery:
    """创建测试用 SceneQuery。"""
    defaults: dict[str, object] = {
        "scene_id": "integration-scene",
        "frame_token": "integration-frame",
        "scenario_type": ScenarioType.UNKNOWN,
        "keywords": [],
        "scene_facts": SceneFacts(),
    }
    defaults.update(overrides)
    return SceneQuery(**defaults)  # type: ignore[arg-type]


class TestMvpConditions:
    """MVP condition 召回测试。"""

    @pytest.mark.parametrize(
        ("facts", "expected_rule_ids"),
        [
            (SceneFacts(has_red_light=True), {"R-SIG-01", "R-SIG-03"}),
            (SceneFacts(has_yellow_light=True), {"R-SIG-02", "R-SIG-03"}),
            (SceneFacts(pedestrian_in_crosswalk=True), {"R-YLD-01", "R-YLD-05"}),
            (SceneFacts(oncoming_vehicle_moving=True), {"R-YLD-02", "R-YLD-05"}),
            (SceneFacts(ego_on_minor_road=True), {"R-YLD-03"}),
        ],
    )
    def test_each_mvp_condition_returns_expected_rules(
        self,
        mvp_retriever: GraphRAGRetriever,
        facts: SceneFacts,
        expected_rule_ids: set[str],
    ) -> None:
        subgraph = mvp_retriever.retrieve(make_query(scene_facts=facts))
        assert {rule.node_id for rule in subgraph.rules} == expected_rule_ids
        assert subgraph.retrieval_mode == RetrievalMode.STRUCTURED


class TestKeywordFallback:
    """关键词兜底集成测试。"""

    def test_english_keyword_fallback(self, mvp_retriever: GraphRAGRetriever) -> None:
        subgraph = mvp_retriever.retrieve(make_query(keywords=["zebra"]))
        assert {rule.node_id for rule in subgraph.rules} == {"R-YLD-01", "R-YLD-05"}
        assert subgraph.retrieval_mode == RetrievalMode.KEYWORD_FALLBACK


class TestOverridePath:
    """override 多跳场景测试。"""

    def test_removed_emergency_fixture_does_not_override_red_light(
        self,
        mvp_retriever: GraphRAGRetriever,
    ) -> None:
        subgraph = mvp_retriever.retrieve(
            make_query(
                scene_facts=SceneFacts(
                    has_red_light=True,
                    emergency_vehicle_active=True,
                )
            )
        )
        by_id = {rule.node_id: rule for rule in subgraph.rules}
        assert by_id["R-SIG-01"].status == RuleStatus.ACTIVE
        assert by_id["R-SIG-01"].overridden_by == []
        assert {rule.node_id for rule in subgraph.active_rules} == {
            "R-SIG-01",
            "R-SIG-03",
        }
        assert subgraph.overrides_applied == []


class TestEmptyAndUnknownScenarios:
    """空场景与未知场景测试。"""

    def test_no_match_returns_empty_subgraph(
        self,
        mvp_retriever: GraphRAGRetriever,
    ) -> None:
        subgraph = mvp_retriever.retrieve(make_query(keywords=["normal driving"]))
        assert subgraph.rules == []
        assert subgraph.active_rules == []
        assert subgraph.hard_rules == []
        assert subgraph.soft_rules == []


class TestRetrievalMetadata:
    """检索元信息测试。"""

    def test_all_node_ids_exist_in_graph(self, mvp_retriever: GraphRAGRetriever) -> None:
        subgraph = mvp_retriever.retrieve(
            make_query(
                scene_facts=SceneFacts(
                    has_red_light=True,
                    pedestrian_in_crosswalk=True,
                )
            )
        )
        for rule in subgraph.rules:
            assert rule.node_id in mvp_retriever._rule_graph._g

    def test_query_duration_reasonable(self, mvp_retriever: GraphRAGRetriever) -> None:
        subgraph = mvp_retriever.retrieve(
            make_query(scene_facts=SceneFacts(has_red_light=True))
        )
        assert subgraph.query_duration_ms < 100
