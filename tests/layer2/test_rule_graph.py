"""Layer 2 rule_graph.py 单元测试。"""

from pathlib import Path

import pytest

from src.layer2.exceptions import GraphDefinitionError, RuleGraphValidationError
from src.layer2.models import RetrievalMode, Severity
from src.layer2.rule_graph import Layer2Settings, RuleGraph


def write_graph(path: Path, body: str) -> Path:
    """写入测试 YAML 并返回路径。"""
    path.write_text(body, encoding="utf-8")
    return path


class TestRuleGraphLoad:
    """图加载测试。"""

    def test_load_success(self, rule_graph: RuleGraph) -> None:
        graph = rule_graph._g
        assert rule_graph.version == "test-v3.0"
        assert graph.number_of_nodes() == 14
        assert graph.number_of_edges() == 15

    def test_load_sets_loaded_flag(self, rule_graph: RuleGraph) -> None:
        assert rule_graph._loaded is True

    def test_load_missing_file(self, tmp_path: Path) -> None:
        settings = Layer2Settings(rule_graph_path=tmp_path / "nonexistent.yaml")
        graph = RuleGraph(settings)
        with pytest.raises(GraphDefinitionError, match="不存在"):
            graph.load()

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = write_graph(tmp_path / "empty.yaml", "")
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(GraphDefinitionError, match="为空"):
            graph.load()

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        path = write_graph(tmp_path / "invalid.yaml", ": bad yaml: :::")
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(GraphDefinitionError, match="YAML"):
            graph.load()

    def test_load_missing_graph_version(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "missing_version.yaml",
            """
conditions: []
rules: []
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="graph_version"):
            graph.load()

    def test_load_duplicate_id(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "dup.yaml",
            """
graph_version: test
conditions:
  - id: C-RED-PHASE
    description: red
  - id: C-RED-PHASE
    description: red again
rules: []
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="重复"):
            graph.load()

    def test_load_dangling_edge(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "dangling.yaml",
            """
graph_version: test
rules:
  - id: R-TEST
    code: test
    description: test
    severity: hard
    category: signal
    applies_when: [C-NOEXIST]
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="悬空边"):
            graph.load()

    def test_load_invalid_severity(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "bad_severity.yaml",
            """
graph_version: test
rules:
  - id: R-TEST
    code: test
    description: test
    severity: invalid
    category: signal
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="severity"):
            graph.load()

    def test_load_invalid_category(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "bad_category.yaml",
            """
graph_version: test
rules:
  - id: R-TEST
    code: test
    description: test
    severity: hard
    category: invalid
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="category"):
            graph.load()

    def test_load_invalid_override_rule(self, tmp_path: Path) -> None:
        path = write_graph(
            tmp_path / "bad_override.yaml",
            """
graph_version: test
conditions:
  - id: C-RED-PHASE
    description: red
rules:
  - id: R-TEST
    code: test
    description: test
    severity: hard
    category: signal
    applies_when: [C-RED-PHASE]
    overrides:
      - rule_id: R-NOEXIST
        condition: C-RED-PHASE
        description: bad
""",
        )
        graph = RuleGraph(Layer2Settings(rule_graph_path=path))
        with pytest.raises(RuleGraphValidationError, match="悬空边"):
            graph.load()


class TestRulesForConditions:
    """遍历召回测试。"""

    def test_match_red_phase_returns_r_sig_01(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE"])
        assert [rule.node_id for rule in rules] == ["R-SIG-01"]
        assert rules[0].severity == Severity.HARD
        assert rules[0].category == "signal"

    def test_match_multiple_conditions_aggregates_before_build(
        self, rule_graph: RuleGraph
    ) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE", "C-EMERGENCY-ACTIVE"])
        by_id = {rule.node_id: rule for rule in rules}
        assert set(by_id) == {"R-SIG-01", "R-YLD-04"}
        assert by_id["R-SIG-01"].matched_conditions == ["C-RED-PHASE"]
        assert by_id["R-YLD-04"].matched_conditions == ["C-EMERGENCY-ACTIVE"]

    def test_no_match_returns_empty(self, rule_graph: RuleGraph) -> None:
        assert rule_graph.rules_for_conditions(["C-NOEXIST"]) == []

    def test_empty_input_returns_empty(self, rule_graph: RuleGraph) -> None:
        assert rule_graph.rules_for_conditions([]) == []

    def test_unloaded_graph_returns_empty(self, tiny_settings: Layer2Settings) -> None:
        graph = RuleGraph(tiny_settings)
        assert graph.rules_for_conditions(["C-RED-PHASE"]) == []

    def test_retrieved_via_is_structured(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE"])
        assert rules[0].retrieved_via == RetrievalMode.STRUCTURED

    def test_actors_collected(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE"])
        assert {actor.type for actor in rules[0].actors} == {"vehicle"}

    def test_consequences_collected(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE"])
        assert rules[0].consequences[0].type == "signal_violation"


class TestOverridesFor:
    """override 查询测试。"""

    def test_returns_override_when_rules_and_condition_match(self, rule_graph: RuleGraph) -> None:
        overrides = rule_graph.overrides_for(
            ["R-SIG-01", "R-YLD-04"],
            ["C-RED-PHASE", "C-EMERGENCY-ACTIVE"],
        )
        assert len(overrides) == 1
        assert overrides[0].overriding_rule_id == "R-YLD-04"
        assert overrides[0].overridden_rule_id == "R-SIG-01"
        assert overrides[0].condition_id == "C-EMERGENCY-ACTIVE"

    def test_override_not_returned_when_condition_missing(self, rule_graph: RuleGraph) -> None:
        overrides = rule_graph.overrides_for(["R-SIG-01", "R-YLD-04"], ["C-RED-PHASE"])
        assert overrides == []

    def test_override_not_returned_when_rule_missing(self, rule_graph: RuleGraph) -> None:
        overrides = rule_graph.overrides_for(["R-YLD-04"], ["C-EMERGENCY-ACTIVE"])
        assert overrides == []


class TestAllConditionSpecs:
    """条件规格查询测试。"""

    def test_returns_condition_specs(self, rule_graph: RuleGraph) -> None:
        specs = rule_graph.all_condition_specs()
        assert specs["C-RED-PHASE"].keywords == ["red light", "traffic light", "stop line"]
        assert specs["C-RED-PHASE"].fact_keys == ["has_red_light"]

    def test_unloaded_graph_returns_empty(self, tiny_settings: Layer2Settings) -> None:
        graph = RuleGraph(tiny_settings)
        assert graph.all_condition_specs() == {}


class TestDeterministicOrdering:
    """确定性排序测试。"""

    def test_rules_for_conditions_deterministic_order(self, rule_graph: RuleGraph) -> None:
        conditions = ["C-EMERGENCY-ACTIVE", "C-RED-PHASE"]
        result1 = rule_graph.rules_for_conditions(conditions)
        result2 = rule_graph.rules_for_conditions(conditions)
        assert [rule.node_id for rule in result1] == [rule.node_id for rule in result2]

    def test_rules_for_conditions_sorted_by_node_id(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-EMERGENCY-ACTIVE", "C-RED-PHASE"])
        rule_ids = [rule.node_id for rule in rules]
        assert rule_ids == sorted(rule_ids)

    def test_duplicate_condition_ids_handled(self, rule_graph: RuleGraph) -> None:
        rules = rule_graph.rules_for_conditions(["C-RED-PHASE", "C-RED-PHASE"])
        assert [rule.node_id for rule in rules] == ["R-SIG-01"]


class TestLoadIdempotency:
    """load() 幂等性测试。"""

    def test_load_twice_no_error(self, tiny_settings: Layer2Settings) -> None:
        graph = RuleGraph(tiny_settings)
        graph.load()
        graph.load()
        assert graph._loaded is True
        assert graph._g.number_of_nodes() == 14
