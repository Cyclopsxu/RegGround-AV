"""Layer 2：图谱检索层。"""

from src.layer2.models import (
    ConditionSpec,
    RetrievalMode,
    RuleCategory,
    RuleNode,
    RuleOverride,
    RuleStatus,
    RuleSubgraph,
    ScenarioType,
    SceneFacts,
    SceneQuery,
    Severity,
)
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings

__all__ = [
    "GraphRAGRetriever",
    "Layer2Settings",
    "ConditionSpec",
    "RetrievalMode",
    "RuleCategory",
    "RuleNode",
    "RuleOverride",
    "RuleStatus",
    "RuleSubgraph",
    "SceneFacts",
    "SceneQuery",
    "ScenarioType",
    "Severity",
    "retrieve",
]


def retrieve(
    query: SceneQuery,
    *,
    retriever: GraphRAGRetriever | None = None,
) -> RuleSubgraph:
    """便捷函数，将 SceneQuery 映射为法规子图。

    若未提供 retriever，将使用默认配置创建新实例并立即检索。
    生产环境推荐显式管理 GraphRAGRetriever 生命周期。

    Args:
        query: Layer 1 输出的 Layer 2 窄接口
        retriever: 可复用的 GraphRAGRetriever 实例（可选）

    Returns:
        包含匹配法规的 RuleSubgraph
    """
    if retriever is None:
        settings = Layer2Settings()
        retriever = GraphRAGRetriever(settings)
    return retriever.retrieve(query)
