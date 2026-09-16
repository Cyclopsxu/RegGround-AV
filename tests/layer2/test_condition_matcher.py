"""Layer 2 condition_matcher.py 单元测试。"""

import pytest

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery
from src.layer2.condition_matcher import ConditionMatcher
from src.layer2.models import ConditionSpec, RetrievalMode
from src.layer2.rule_graph import RuleGraph


@pytest.fixture
def condition_specs(rule_graph: RuleGraph) -> dict[str, ConditionSpec]:
    """从已加载的 RuleGraph 获取 Condition 规格。"""
    return rule_graph.all_condition_specs()


@pytest.fixture
def matcher(condition_specs: dict[str, ConditionSpec]) -> ConditionMatcher:
    """创建默认的 ConditionMatcher（兜底开启）。"""
    return ConditionMatcher(condition_specs, enable_keyword_fallback=True)


@pytest.fixture
def matcher_no_fallback(condition_specs: dict[str, ConditionSpec]) -> ConditionMatcher:
    """创建关闭兜底的 ConditionMatcher。"""
    return ConditionMatcher(condition_specs, enable_keyword_fallback=False)


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


class TestStructuredMatch:
    """结构化事实匹配测试。"""

    def test_red_light_fact_matches_red_phase(self, matcher: ConditionMatcher) -> None:
        query = make_query(scene_facts=SceneFacts(has_red_light=True))
        condition_ids, mode = matcher.match(query)
        assert condition_ids == ["C-RED-PHASE"]
        assert mode == RetrievalMode.STRUCTURED

    def test_pedestrian_fact_matches_crosswalk(self, matcher: ConditionMatcher) -> None:
        query = make_query(scene_facts=SceneFacts(pedestrian_in_crosswalk=True))
        condition_ids, mode = matcher.match(query)
        assert condition_ids == ["C-PED-IN-CROSSWALK"]
        assert mode == RetrievalMode.STRUCTURED

    def test_facts_override_inconsistent_scenario(self, matcher: ConditionMatcher) -> None:
        query = make_query(
            scenario_type=ScenarioType.PEDESTRIAN,
            scene_facts=SceneFacts(has_red_light=True),
        )
        condition_ids, mode = matcher.match(query)
        assert condition_ids == ["C-RED-PHASE"]
        assert mode == RetrievalMode.STRUCTURED

    def test_scenario_type_does_not_replace_missing_facts(
        self, matcher: ConditionMatcher
    ) -> None:
        query = make_query(scenario_type=ScenarioType.RED_LIGHT)
        condition_ids, mode = matcher.match(query)
        assert condition_ids == []
        assert mode == RetrievalMode.STRUCTURED

    def test_no_match_returns_empty(self, matcher: ConditionMatcher) -> None:
        query = make_query(keywords=["normal driving"])
        condition_ids, mode = matcher.match(query)
        assert condition_ids == []
        assert mode == RetrievalMode.STRUCTURED


class TestKeywordFallback:
    """英文关键词兜底测试。"""

    def test_fallback_partial_match(self, matcher: ConditionMatcher) -> None:
        query = make_query(keywords=["red"])
        condition_ids, mode = matcher.match(query)
        assert condition_ids == ["C-RED-PHASE"]
        assert mode == RetrievalMode.KEYWORD_FALLBACK

    def test_fallback_disabled_no_hit(
        self, matcher_no_fallback: ConditionMatcher
    ) -> None:
        query = make_query(keywords=["red"])
        condition_ids, mode = matcher_no_fallback.match(query)
        assert condition_ids == []
        assert mode == RetrievalMode.STRUCTURED

    def test_keywords_are_ignored_when_facts_match(self, matcher: ConditionMatcher) -> None:
        query = make_query(
            keywords=["pedestrian"],
            scene_facts=SceneFacts(has_red_light=True),
        )
        condition_ids, mode = matcher.match(query)
        assert condition_ids == ["C-RED-PHASE"]
        assert mode == RetrievalMode.STRUCTURED

    def test_fallback_still_no_match(self, matcher: ConditionMatcher) -> None:
        query = make_query(keywords=["unrelated"])
        condition_ids, mode = matcher.match(query)
        assert condition_ids == []
        assert mode == RetrievalMode.STRUCTURED


class TestDeterminism:
    """引用透明性测试。"""

    def test_same_input_same_output(self, matcher: ConditionMatcher) -> None:
        query = make_query(
            keywords=["pedestrian"],
            scene_facts=SceneFacts(has_red_light=True),
        )
        assert matcher.match(query) == matcher.match(query)
