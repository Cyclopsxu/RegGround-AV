"""法规图谱加载、校验与遍历查询。"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.layer2.exceptions import GraphDefinitionError, RuleGraphValidationError
from src.layer2.models import (
    ActorType,
    ConditionSpec,
    Consequence,
    RetrievalMode,
    RoadActor,
    RuleCategory,
    RuleNode,
    RuleOverride,
    Severity,
)

logger = logging.getLogger(__name__)


class Layer2Settings(BaseSettings):
    """Layer 2 运行期配置。"""

    rule_graph_path: Path = Path("data/layer2/rule_graph.yaml")
    enable_keyword_fallback: bool = True

    model_config = SettingsConfigDict(env_prefix="LAYER2_")


class RuleGraph:
    """NetworkX 法规图的加载、校验与遍历查询封装。"""

    def __init__(self, settings: Layer2Settings) -> None:
        self._settings = settings
        self._g: nx.DiGraph = nx.DiGraph()
        self._loaded = False
        self._version = ""

    @property
    def version(self) -> str:
        """当前图谱版本。"""
        return self._version

    def load(self) -> None:
        """读取 YAML 图定义、构建内存图、执行完整性校验。"""
        if self._loaded:
            logger.warning("图谱已加载，跳过重复 load() 调用")
            return

        graph_path = self._settings.rule_graph_path
        logger.info("开始加载法规图谱 path=%s", graph_path)

        raw = self._read_yaml(graph_path)
        self._validate_and_build(raw, graph_path)
        self._loaded = True

        logger.info(
            "法规图谱加载完成 path=%s version=%s nodes=%d edges=%d",
            graph_path,
            self._version,
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    def rules_for_conditions(self, condition_ids: list[str]) -> list[RuleNode]:
        """给定命中的 Condition id，遍历召回全部适用 TrafficRule。"""
        if not self._loaded or not condition_ids:
            return []

        cond_by_rule: dict[str, list[str]] = defaultdict(list)
        for condition_id in sorted(set(condition_ids)):
            if condition_id not in self._g:
                continue
            for rule_id, _, edge in self._g.in_edges(condition_id, data=True):
                if edge.get("rel") != "APPLIES_WHEN":
                    continue
                if self._g.nodes[rule_id].get("kind") != "TrafficRule":
                    continue
                cond_by_rule[rule_id].append(condition_id)

        return [
            self._build_rule_node(rule_id, sorted(conditions), RetrievalMode.STRUCTURED)
            for rule_id, conditions in sorted(cond_by_rule.items())
        ]

    def overrides_for(
        self,
        rule_ids: list[str],
        condition_ids: list[str],
    ) -> list[RuleOverride]:
        """返回当前召回规则集合中可触发的 override 定义。"""
        if not self._loaded or not rule_ids or not condition_ids:
            return []

        rule_id_set = set(rule_ids)
        condition_id_set = set(condition_ids)
        overrides: list[RuleOverride] = []

        for overriding_rule_id in sorted(rule_id_set):
            if overriding_rule_id not in self._g:
                continue
            for _, overridden_rule_id, edge in self._g.out_edges(overriding_rule_id, data=True):
                if edge.get("rel") != "OVERRIDES":
                    continue
                condition_id = str(edge.get("condition_id", ""))
                if overridden_rule_id not in rule_id_set or condition_id not in condition_id_set:
                    continue
                overrides.append(
                    RuleOverride(
                        overriding_rule_id=overriding_rule_id,
                        overridden_rule_id=overridden_rule_id,
                        condition_id=condition_id,
                        reason=str(edge.get("reason", "")),
                    )
                )

        return overrides

    def all_condition_specs(self) -> dict[str, ConditionSpec]:
        """返回 {condition_id: ConditionSpec}，供 ConditionMatcher 使用。"""
        if not self._loaded:
            return {}

        specs: dict[str, ConditionSpec] = {}
        for node_id, data in self._g.nodes(data=True):
            if data.get("kind") != "Condition":
                continue
            specs[node_id] = ConditionSpec(
                condition_id=node_id,
                description=str(data.get("description", "")),
                keywords=list(data.get("keywords", []) or []),
                fact_keys=list(data.get("fact_keys", []) or []),
            )
        return dict(sorted(specs.items()))

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """读取并解析 YAML 图定义文件。"""
        if not path.exists():
            raise GraphDefinitionError(f"图定义文件不存在: {path!s}")

        try:
            with path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError as e:
            raise GraphDefinitionError(f"图定义文件不可读 {path!s}: {e}") from e
        except yaml.YAMLError as e:
            raise GraphDefinitionError(f"YAML 解析失败 {path!s}: {e}") from e

        if raw is None:
            raise GraphDefinitionError(f"图定义文件为空: {path!s}")
        if not isinstance(raw, dict):
            raise GraphDefinitionError(f"图定义文件顶层必须为 mapping: {path!s}")
        return raw

    def _validate_and_build(self, raw: dict[str, Any], graph_path: Path) -> None:
        """校验 YAML 原始数据并构建 NetworkX 图。"""
        graph_version = raw.get("graph_version")
        if not isinstance(graph_version, str) or not graph_version:
            raise RuleGraphValidationError(f"缺少 graph_version，位于 {graph_path!s}")
        self._version = graph_version

        for node_list, kind in [
            ("conditions", "Condition"),
            ("behaviors", "Behavior"),
            ("actors", "RoadActor"),
            ("consequences", "Consequence"),
            ("rules", "TrafficRule"),
        ]:
            for item in raw.get(node_list, []) or []:
                if not isinstance(item, dict):
                    raise RuleGraphValidationError(
                        f"{kind} 节点必须为 mapping，位于 {graph_path!s}"
                    )
                node_id = item.get("id")
                if not node_id:
                    raise RuleGraphValidationError(
                        f"{kind} 节点缺少 id 字段，位于 {graph_path!s}"
                    )
                if node_id in self._g:
                    raise RuleGraphValidationError(
                        f"节点 id 重复: {node_id!r} 在 {graph_path!s}"
                    )
                attrs = {k: v for k, v in item.items() if k != "id"}
                attrs["kind"] = kind
                self._g.add_node(node_id, **attrs)

        self._validate_rules(graph_path)
        self._add_edges(raw)

    def _validate_rules(self, graph_path: Path) -> None:
        """校验所有 TrafficRule 节点的完整性。"""
        valid_severity = {item.value for item in Severity}
        valid_category = {item.value for item in RuleCategory}

        for node_id, data in self._g.nodes(data=True):
            if data.get("kind") != "TrafficRule":
                continue

            severity = data.get("severity")
            if severity not in valid_severity:
                raise RuleGraphValidationError(
                    f"非法 severity 值 {severity!r}，规则 {node_id!r}，"
                    f"允许值为 {valid_severity}，位于 {graph_path!s}"
                )

            category = data.get("category")
            if category not in valid_category:
                raise RuleGraphValidationError(
                    f"非法 category 值 {category!r}，规则 {node_id!r}，"
                    f"允许值为 {valid_category}，位于 {graph_path!s}"
                )

            for field, expected_kind in [
                ("applies_when", "Condition"),
                ("prohibits", "Behavior"),
                ("requires", "Behavior"),
                ("applies_to", "RoadActor"),
                ("consequences", "Consequence"),
                ("exception_when", "Condition"),
            ]:
                self._validate_refs(node_id, field, expected_kind, graph_path)

            for override in data.get("overrides", []) or []:
                if not isinstance(override, dict):
                    raise RuleGraphValidationError(
                        f"override 必须为 mapping，规则 {node_id!r}，位于 {graph_path!s}"
                    )
                overridden_rule_id = override.get("rule_id")
                condition_id = override.get("condition")
                self._validate_ref(
                    node_id,
                    "overrides.rule_id",
                    overridden_rule_id,
                    "TrafficRule",
                    graph_path,
                )
                self._validate_ref(
                    node_id,
                    "overrides.condition",
                    condition_id,
                    "Condition",
                    graph_path,
                )

    def _validate_refs(
        self,
        node_id: str,
        field: str,
        expected_kind: str,
        graph_path: Path,
    ) -> None:
        """校验规则字段引用的节点存在且类型正确。"""
        for ref_id in self._g.nodes[node_id].get(field, []) or []:
            self._validate_ref(node_id, field, ref_id, expected_kind, graph_path)

    def _validate_ref(
        self,
        node_id: str,
        field: str,
        ref_id: Any,
        expected_kind: str,
        graph_path: Path,
    ) -> None:
        """校验单个节点引用。"""
        if ref_id not in self._g:
            raise RuleGraphValidationError(
                f"悬空边: 规则 {node_id!r} {field.upper()} 引用了不存在的节点 "
                f"{ref_id!r}，位于 {graph_path!s}"
            )
        actual_kind = self._g.nodes[ref_id].get("kind")
        if actual_kind != expected_kind:
            raise RuleGraphValidationError(
                f"边类型不匹配: 规则 {node_id!r} {field.upper()} 引用了 "
                f"{ref_id!r}，期望 kind={expected_kind!r}，实际 kind={actual_kind!r}，"
                f"位于 {graph_path!s}"
            )

    def _add_edges(self, raw: dict[str, Any]) -> None:
        """根据 YAML 定义添加所有图边。"""
        edge_defs = [
            ("applies_when", "APPLIES_WHEN"),
            ("prohibits", "PROHIBITS"),
            ("requires", "REQUIRES"),
            ("applies_to", "APPLIES_TO"),
            ("consequences", "LEADS_TO_CONSEQUENCE"),
            ("exception_when", "EXCEPTION_WHEN"),
        ]

        for item in raw.get("rules", []) or []:
            rule_id = item["id"]
            for field, rel in edge_defs:
                for target_id in item.get(field, []) or []:
                    self._g.add_edge(rule_id, target_id, rel=rel)

            for override in item.get("overrides", []) or []:
                self._g.add_edge(
                    rule_id,
                    override["rule_id"],
                    rel="OVERRIDES",
                    condition_id=override["condition"],
                    reason=override.get("description", ""),
                )

        for item in raw.get("conditions", []) or []:
            condition_id = item["id"]
            for target_id in item.get("co_occurs", []) or []:
                self._g.add_edge(condition_id, target_id, rel="CO_OCCURS")

    def _build_rule_node(
        self,
        rule_id: str,
        matched_conditions: list[str],
        retrieval_mode: RetrievalMode,
    ) -> RuleNode:
        """从图节点数据组装 RuleNode。"""
        data = self._g.nodes[rule_id]

        actors: list[RoadActor] = []
        consequences: list[Consequence] = []
        for _, neighbor, edge in self._g.out_edges(rule_id, data=True):
            neighbor_data = self._g.nodes[neighbor]
            neighbor_kind = neighbor_data.get("kind")
            edge_rel = edge.get("rel")

            if edge_rel == "APPLIES_TO" and neighbor_kind == "RoadActor":
                actors.append(
                    RoadActor(
                        type=ActorType(neighbor_data.get("type", ActorType.VEHICLE.value)),
                        description=str(neighbor_data.get("description", "")),
                        priority_level=int(neighbor_data.get("priority_level", 3)),
                    )
                )
            elif edge_rel == "LEADS_TO_CONSEQUENCE" and neighbor_kind == "Consequence":
                consequences.append(
                    Consequence(
                        type=str(neighbor_data.get("type", "")),
                        description=str(neighbor_data.get("description", "")),
                        severity_score=int(neighbor_data.get("severity_score", 1)),
                    )
                )

        return RuleNode(
            node_id=rule_id,
            code=str(data.get("code", "")),
            description=str(data.get("description", "")),
            severity=Severity(data["severity"]),
            category=RuleCategory(data["category"]),
            actors=actors,
            consequences=consequences,
            matched_conditions=matched_conditions,
            retrieved_via=retrieval_mode,
        )
