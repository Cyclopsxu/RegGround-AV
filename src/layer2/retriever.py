"""Layer 2 对外 Facade —— GraphRAGRetriever。"""

import logging
import time
from datetime import UTC, datetime

from src.layer1.models import SceneQuery
from src.layer2.condition_matcher import ConditionMatcher
from src.layer2.conflict_resolver import ConflictResolver
from src.layer2.models import RetrievalMode, RuleNode, RuleSubgraph
from src.layer2.rule_graph import Layer2Settings, RuleGraph

logger = logging.getLogger(__name__)


class GraphRAGRetriever:
    """Layer 2 对外的核心入口，编排匹配、遍历与冲突消解。"""

    def __init__(
        self,
        settings: Layer2Settings,
        *,
        rule_graph: RuleGraph | None = None,
        matcher: ConditionMatcher | None = None,
        conflict_resolver: ConflictResolver | None = None,
    ) -> None:
        self._settings = settings
        self._rule_graph = rule_graph or RuleGraph(settings)
        self._rule_graph.load()

        condition_specs = self._rule_graph.all_condition_specs()
        self._matcher = matcher or ConditionMatcher(
            condition_specs,
            enable_keyword_fallback=settings.enable_keyword_fallback,
        )
        self._conflict_resolver = conflict_resolver or ConflictResolver()

        logger.info(
            "GraphRAGRetriever 初始化完成 conditions=%d graph_version=%s",
            len(condition_specs),
            self._rule_graph.version,
        )

    def retrieve(self, query: SceneQuery) -> RuleSubgraph:
        """将 SceneQuery 映射为可追溯的法规子图。"""
        start = time.perf_counter()

        condition_ids, mode = self._matcher.match(query)
        rules = self._rule_graph.rules_for_conditions(condition_ids)
        rules = self._with_retrieval_mode(rules, mode)
        overrides = self._rule_graph.overrides_for([r.node_id for r in rules], condition_ids)
        resolved_rules, applied = self._conflict_resolver.resolve(
            rules,
            condition_ids,
            overrides,
        )

        query_duration_ms = int((time.perf_counter() - start) * 1000)
        subgraph = RuleSubgraph(
            scene_id=query.scene_id,
            frame_token=query.frame_token,
            rules=resolved_rules,
            matched_conditions=condition_ids,
            retrieval_mode=mode,
            overrides_applied=applied,
            query_duration_ms=query_duration_ms,
            graph_version=self._rule_graph.version,
            retrieval_timestamp=datetime.now(UTC),
        )

        logger.info(
            "检索完成 scene_id=%s frame_token=%s retrieval_mode=%s "
            "matched_conditions=%s num_rules_returned=%d num_active_rules=%d "
            "overrides_applied=%d query_duration_ms=%d",
            query.scene_id,
            query.frame_token,
            mode.value,
            condition_ids,
            len(subgraph.rules),
            len(subgraph.active_rules),
            len(applied),
            query_duration_ms,
        )

        return subgraph

    def close(self) -> None:
        """释放资源。NetworkX 内存图无需额外关闭。"""
        logger.info("GraphRAGRetriever 关闭")

    @staticmethod
    def _with_retrieval_mode(rules: list[RuleNode], mode: RetrievalMode) -> list[RuleNode]:
        """按本次查询模式标记 RuleNode，避免运行期修改 frozen 模型。"""
        if mode == RetrievalMode.STRUCTURED:
            return rules
        return [rule.model_copy(update={"retrieved_via": mode}) for rule in rules]
