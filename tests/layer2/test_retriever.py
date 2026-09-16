"""Layer 2 retriever.py 单元测试。"""

from datetime import datetime

import pytest

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery
from src.layer2.condition_matcher import ConditionMatcher
from src.layer2.conflict_resolver import ConflictResolver
from src.layer2.models import RetrievalMode, RuleStatus, RuleSubgraph
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings, RuleGraph


@pytest.fixture
def retriever(tiny_settings: Layer2Settings) -> GraphRAGRetriever:
    """创建使用测试用最小图谱的 GraphRAGRetriever。"""
    return GraphRAGRetriever(tiny_settings)


def make_query(**overrides: object) -> SceneQuery:
    """创建测试用 SceneQuery。"""
    defaults: dict[str, object] = {
        "scene_id": "test-scene",
        "frame_token": "frame-001",
        "scenario_type": ScenarioType.UNKNOWN,
        "keywords": [],
        "scene_facts": SceneFacts(),
    }
    defaults.update(overrides)
    return SceneQuery(**defaults)  # type: ignore[arg-type]


class TestRetrieve:
    """检索编排测试。"""

    def test_retrieve_red_light_returns_rule(
        self,
        retriever: GraphRAGRetriever,
        red_light_query: SceneQuery,
    ) -> None:
        subgraph = retriever.retrieve(red_light_query)
        assert isinstance(subgraph, RuleSubgraph)
        assert {rule.node_id for rule in subgraph.rules} == {"R-SIG-01"}

    def test_retrieve_pedestrian_returns_rule(
        self,
        retriever: GraphRAGRetriever,
        pedestrian_query: SceneQuery,
    ) -> None:
        subgraph = retriever.retrieve(pedestrian_query)
        assert {rule.node_id for rule in subgraph.rules} == {"R-YLD-01"}

    def test_retrieve_no_match_returns_empty_subgraph(
        self,
        retriever: GraphRAGRetriever,
        unknown_query: SceneQuery,
    ) -> None:
        subgraph = retriever.retrieve(unknown_query)
        assert subgraph.rules == []
        assert subgraph.active_rules == []
        assert subgraph.hard_rules == []
        assert subgraph.soft_rules == []

    def test_scene_metadata_propagated(self, retriever: GraphRAGRetriever) -> None:
        query = make_query(
            scene_id="my-scene-42",
            frame_token="my-frame-42",
            scene_facts=SceneFacts(has_red_light=True),
        )
        subgraph = retriever.retrieve(query)
        assert subgraph.scene_id == "my-scene-42"
        assert subgraph.frame_token == "my-frame-42"
        assert subgraph.graph_version == "test-v3.0"

    def test_retrieval_mode_structured(self, retriever: GraphRAGRetriever) -> None:
        query = make_query(scene_facts=SceneFacts(has_red_light=True))
        subgraph = retriever.retrieve(query)
        assert subgraph.retrieval_mode == RetrievalMode.STRUCTURED
        assert all(rule.retrieved_via == RetrievalMode.STRUCTURED for rule in subgraph.rules)

    def test_retrieval_mode_keyword_fallback(self, retriever: GraphRAGRetriever) -> None:
        query = make_query(keywords=["red"])
        subgraph = retriever.retrieve(query)
        assert subgraph.retrieval_mode == RetrievalMode.KEYWORD_FALLBACK
        assert all(
            rule.retrieved_via == RetrievalMode.KEYWORD_FALLBACK
            for rule in subgraph.rules
        )

    def test_structured_match_ignores_keyword_condition(self, retriever: GraphRAGRetriever) -> None:
        query = make_query(
            keywords=["pedestrian"],
            scene_facts=SceneFacts(has_red_light=True),
        )
        subgraph = retriever.retrieve(query)
        assert subgraph.retrieval_mode == RetrievalMode.STRUCTURED
        assert {rule.node_id for rule in subgraph.rules} == {"R-SIG-01"}

    def test_override_excludes_overridden_from_active_rules(
        self,
        retriever: GraphRAGRetriever,
    ) -> None:
        query = make_query(
            scene_facts=SceneFacts(has_red_light=True, emergency_vehicle_active=True),
        )
        subgraph = retriever.retrieve(query)
        by_id = {rule.node_id: rule for rule in subgraph.rules}
        assert by_id["R-SIG-01"].status == RuleStatus.OVERRIDDEN
        assert by_id["R-SIG-01"].overridden_by == ["R-YLD-04"]
        assert {rule.node_id for rule in subgraph.active_rules} == {"R-YLD-04"}
        assert {rule.node_id for rule in subgraph.hard_rules} == {"R-YLD-04"}
        assert len(subgraph.overrides_applied) == 1

    def test_matched_conditions_recorded(self, retriever: GraphRAGRetriever) -> None:
        query = make_query(scene_facts=SceneFacts(has_red_light=True))
        subgraph = retriever.retrieve(query)
        assert subgraph.matched_conditions == ["C-RED-PHASE"]

    def test_query_duration_ms_positive(self, retriever: GraphRAGRetriever) -> None:
        subgraph = retriever.retrieve(make_query())
        assert subgraph.query_duration_ms >= 0
        assert isinstance(subgraph.query_duration_ms, int)

    def test_retrieval_timestamp_utc(self, retriever: GraphRAGRetriever) -> None:
        subgraph = retriever.retrieve(make_query())
        assert isinstance(subgraph.retrieval_timestamp, datetime)


class TestClose:
    """close 方法测试。"""

    def test_close_does_not_raise(self, retriever: GraphRAGRetriever) -> None:
        retriever.close()


class TestDependencyInjection:
    """依赖注入测试。"""

    def test_inject_rule_graph_matcher_and_resolver(
        self,
        tiny_settings: Layer2Settings,
    ) -> None:
        graph = RuleGraph(tiny_settings)
        graph.load()
        matcher = ConditionMatcher(
            graph.all_condition_specs(),
            enable_keyword_fallback=True,
        )
        retriever = GraphRAGRetriever(
            tiny_settings,
            rule_graph=graph,
            matcher=matcher,
            conflict_resolver=ConflictResolver(),
        )
        subgraph = retriever.retrieve(make_query(scene_facts=SceneFacts(has_red_light=True)))
        assert {rule.node_id for rule in subgraph.rules} == {"R-SIG-01"}
