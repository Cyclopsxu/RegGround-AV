"""Layer 2 citation validity 基准。

对全部代表场景跑 retrieve，断言返回的每个 rule_id 均存在于图中。
这是确定性图检索对 citation validity 的构造性保证。

用法: python -m pytest tests/layer2/benchmark_hallucination.py -v -s
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery
from src.layer2.models import RetrievalMode
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "scene_id": "bench-red-light",
        "scenario_type": ScenarioType.RED_LIGHT,
        "keywords": ["red light", "stop line"],
        "scene_facts": SceneFacts(has_red_light=True),
    },
    {
        "scene_id": "bench-yellow-light",
        "scenario_type": ScenarioType.YELLOW_LIGHT,
        "keywords": ["yellow light", "traffic light"],
        "scene_facts": SceneFacts(has_yellow_light=True),
    },
    {
        "scene_id": "bench-pedestrian",
        "scenario_type": ScenarioType.PEDESTRIAN,
        "keywords": ["pedestrian", "crosswalk"],
        "scene_facts": SceneFacts(pedestrian_in_crosswalk=True),
    },
    {
        "scene_id": "bench-oncoming",
        "scenario_type": ScenarioType.ONCOMING,
        "keywords": ["oncoming", "left turn"],
        "scene_facts": SceneFacts(oncoming_vehicle_moving=True),
    },
    {
        "scene_id": "bench-minor-road",
        "scenario_type": ScenarioType.MINOR_ROAD,
        "keywords": ["minor road", "main road"],
        "scene_facts": SceneFacts(ego_on_minor_road=True),
    },
    {
        "scene_id": "bench-emergency",
        "scenario_type": ScenarioType.EMERGENCY,
        "keywords": ["ambulance", "siren"],
        "scene_facts": SceneFacts(emergency_vehicle_active=True),
    },
    {
        "scene_id": "bench-unknown-no-match",
        "scenario_type": ScenarioType.UNKNOWN,
        "keywords": ["normal driving"],
        "scene_facts": SceneFacts(),
    },
    {
        "scene_id": "bench-keyword-fallback",
        "scenario_type": ScenarioType.UNKNOWN,
        "keywords": ["zebra"],
        "scene_facts": SceneFacts(),
    },
]


@pytest.fixture(scope="module")
def benchmark_retriever() -> GraphRAGRetriever:
    """加载 MVP 图的 retriever（模块级复用）。"""
    return GraphRAGRetriever(Layer2Settings())


def _build_queries() -> list[SceneQuery]:
    """从 BENCHMARK_QUERIES 构建 SceneQuery 列表。"""
    return [
        SceneQuery(
            scene_id=raw["scene_id"],
            frame_token=f"{raw['scene_id']}-frame",
            scenario_type=raw["scenario_type"],
            keywords=raw["keywords"],
            scene_facts=raw["scene_facts"],
        )
        for raw in BENCHMARK_QUERIES
    ]


def test_benchmark_citation_validity(benchmark_retriever: GraphRAGRetriever) -> None:
    """citation validity 基准：断言虚构 rule_id 为 0。"""
    queries = _build_queries()
    graph = benchmark_retriever._rule_graph._g

    stats: dict[str, Any] = {
        "total_queries": len(queries),
        "total_rules_returned": 0,
        "hits": 0,
        "misses": 0,
        "invalid_citations": 0,
        "keyword_fallbacks": 0,
        "citation_validity": 1.0,
        "hit_rate": 0.0,
        "fallback_rate": 0.0,
        "per_query": [],
    }

    for query in queries:
        subgraph = benchmark_retriever.retrieve(query)
        num_rules = len(subgraph.rules)
        is_hit = num_rules > 0
        is_fallback = subgraph.retrieval_mode == RetrievalMode.KEYWORD_FALLBACK

        if is_hit:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
        if is_fallback:
            stats["keyword_fallbacks"] += 1

        stats["total_rules_returned"] += num_rules
        for rule in subgraph.rules:
            assert rule.node_id in graph, (
                f"无效 citation rule_id={rule.node_id!r} scene={query.scene_id}"
            )

        stats["per_query"].append(
            {
                "scene_id": query.scene_id,
                "num_rules": num_rules,
                "is_hit": is_hit,
                "retrieval_mode": subgraph.retrieval_mode.value,
            }
        )

    total = stats["total_queries"]
    stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
    stats["fallback_rate"] = stats["keyword_fallbacks"] / total if total > 0 else 0.0
    stats["citation_validity"] = 1.0

    result_json = json.dumps(stats, ensure_ascii=False, indent=2)
    logger.info("citation validity 基准结果:\n%s", result_json)

    output_path = Path(__file__).resolve().parent / "benchmark_hallucination_result.json"
    output_path.write_text(result_json, encoding="utf-8")

    assert stats["invalid_citations"] == 0
    assert stats["citation_validity"] == 1.0


def test_benchmark_performance(benchmark_retriever: GraphRAGRetriever) -> None:
    """性能基准：验证 P95 ≤ 10 ms。"""
    durations: list[int] = []
    for query in _build_queries():
        for _ in range(5):
            subgraph = benchmark_retriever.retrieve(query)
            durations.append(subgraph.query_duration_ms)

    durations.sort()
    p95_idx = int(len(durations) * 0.95)
    p95 = durations[min(p95_idx, len(durations) - 1)]

    logger.info("性能基准: P95=%d ms", p95)
    assert p95 <= 10
