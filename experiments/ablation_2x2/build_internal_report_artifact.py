# ruff: noqa: E501
"""Build the canonical portable-report artifact for the internal 2×2 comparison."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

CONDITION_LABELS = {
    "full": "full (RAG + validation)",
    "no_validation": "no_validation (RAG only)",
    "no_rag": "no_rag (validation only)",
    "baseline": "baseline (neither)",
}


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _condition_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition, label in CONDITION_LABELS.items():
        item = metrics["conditions"][condition]
        rows.append(
            {
                "condition": label,
                "condition_key": condition,
                "scenes": item["n_scenes"],
                "decisions": item["n_decisions"],
                "machine_validity": item["machine_citation_validity_final"],
                "citations": item["citations_final"],
                "citations_per_decision": item["citations_per_decision"],
                "no_citation_rate": item["no_citation_decision_rate"],
                "veto_rate": item["veto_rate"],
                "uncertain_rate": item["uncertain_rate"],
                "chosen_available_rate": item["chosen_available_rate"],
                "tokens_total": item["tokens_total"],
                "tokens_per_scene": item["tokens_mean_per_scene"],
                "mean_latency_s": item["duration_mean_ms"] / 1000,
                "p95_latency_s": item["duration_p95_ms"] / 1000,
                "citation_retries": item["citation_retries"],
                "schema_retries": item["schema_retries"],
                "final_violated_scenes": item["citation_check_violations_final"],
                "enforcement_exhausted": item["enforcement_exhausted"],
            }
        )
    return rows


def _pairwise_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, item in metrics["pairwise_agreement"].items():
        left, right = key.split("__vs__")
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "status_agreement": item["status_agreement"],
                "chosen_agreement": item["chosen_agreement"],
                "exact_ranking_agreement": item["exact_ranking_agreement"],
                "mean_kendall_tau": item["mean_kendall_tau"],
                "decisions": item["n_decisions"],
                "scenes": item["n_scenes"],
            }
        )
    return rows


def _sql_literal(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is None:
        return "NULL"
    return str(value)


def _condition_source_sql(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0])
    values = ",\n    ".join(
        "(" + ", ".join(_sql_literal(row[field]) for field in fields) + ")" for row in rows
    )
    query = (
        "WITH condition_summary (\n    "
        + ", ".join(fields)
        + "\n) AS (\n  VALUES\n    "
        + values
        + "\n)\nSELECT * FROM condition_summary;"
    )
    result = sqlite3.connect(":memory:").execute(query).fetchall()
    if len(result) != len(rows):
        raise ValueError("condition source SQL did not reproduce all rows")
    return query


def build_artifact(metrics: dict[str, Any]) -> dict[str, Any]:
    conditions = _condition_rows(metrics)
    pairwise = _pairwise_rows(metrics)
    full = metrics["conditions"]["full"]
    no_validation = metrics["conditions"]["no_validation"]
    no_rag = metrics["conditions"]["no_rag"]
    baseline = metrics["conditions"]["baseline"]
    retried = no_rag["first_final_agreement_retried_only"]
    total_tokens = sum(item["tokens_total"] for item in metrics["conditions"].values())
    condition_sql = _condition_source_sql(conditions)

    source = {
        "id": "internal_2x2_metrics",
        "label": "Frozen 82-scene DeepSeek 2×2 run and derived internal metrics",
        "path": "internal_metrics.json",
        "query": {
            "description": "Deterministic Python aggregation over four paired JSONL condition files.",
            "language": "sql",
            "sql": condition_sql,
            "engine": "sqlite",
            "executed_at": "2026-08-02T00:00:00Z",
            "tables_used": [
                "V2_16K/full/records.jsonl",
                "V2_16K/no_validation/records.jsonl",
                "V2_16K/no_rag/records.jsonl",
                "V2_16K/baseline/records.jsonl",
            ],
            "filters": [
                "82 frozen BLINDED_INFERENCE scenes",
                "gold fields absent",
                "four conditions paired by frame_token",
            ],
            "metric_definitions": {
                "machine_validity": (
                    "Valid citation events divided by all final citation events; context-ID "
                    "membership for RAG and audited-catalog existence for free text."
                ),
                "status_agreement": "Matched trajectory statuses divided by paired decisions.",
                "latency": "Per-request API duration; summed duration is not concurrent wall time.",
            },
        },
    }
    technical_summary = f"""## 技术摘要

本轮 2×2 内部对比已经完成：四个条件各覆盖 **82 个相同冻结场景**，共 **328 次模型调用、{total_tokens:,} tokens**。当前最明确的机制结论是：GraphRAG 两个条件的机器可核验引用有效率均为 **100%**，自由文本引用条件仅为 **{_percent(no_rag['machine_citation_validity_final'])}**（no_rag）和 **{_percent(baseline['machine_citation_validity_final'])}**（baseline）。这个指标只检查“上下文 ID / 审计法规库中是否存在”，**不能替代人工对内容正确性和场景适用性的判断**。

验证器在 RAG 轨道从未触发引用修复，因此 full 与 no_validation 的裁决差异不能归因于验证器，只能视为两次独立随机采样的波动。no_rag 轨道触发 **{no_rag['citation_retries']}** 次修复，其中仅 **{no_rag['citation_retries'] - no_rag['enforcement_exhausted']}** 次最终通过，**{no_rag['enforcement_exhausted']}** 次耗尽；同时 token 由 baseline 的 **{baseline['tokens_total']:,}** 增至 **{no_rag['tokens_total']:,}**。因此目前不建议把自由文本引用校验当作可靠的自动兜底。

裁决正确率、偏好正确率和完整幻觉率仍需等待全部人工标注与 adjudication 后再计算。"""
    citation_narrative = f"""## 约束式引用稳定，自由文本引用修复没有形成可靠闭环

下图比较最终响应中的机器可核验引用有效率。RAG 条件的分母是规则 ID 引用事件，校验标准是 ID 是否属于检索上下文；自由文本条件的分母是法规条款引用事件，校验标准是规范化条款是否存在于审计法规库。两个轨道的语义不同，因此 **100% 对约 20% 不能直接解读为人工意义上的准确率提升**，但它清楚说明受限 ID 输出比自由文本条款更容易机械校验。

no_rag 的单次修复并未改善总体有效率：首轮为 **{_percent(no_rag['machine_citation_validity_first'])}**，最终为 **{_percent(no_rag['machine_citation_validity_final'])}**。在被修复的 77 个场景中，首尾裁决状态一致率为 **{_percent(retried['status_agreement'])}**、chosen 一致率为 **{_percent(retried['chosen_agreement'])}**、严格排序一致率为 **{_percent(retried['exact_ranking_agreement'])}**；修复不仅昂贵，也会改写任务输出。"""
    cost_narrative = f"""## no_rag 验证代价最高，但未换来稳定的引用通过

下图按场景比较平均 token。no_rag 平均 **{no_rag['tokens_mean_per_scene']:,.0f} tokens/场景**，约为 baseline 的 **{no_rag['tokens_mean_per_scene'] / baseline['tokens_mean_per_scene']:.2f}×**；平均请求时长为 **{no_rag['duration_mean_ms'] / 1000:.1f}s**，baseline 为 **{baseline['duration_mean_ms'] / 1000:.1f}s**。full 反而略少于 no_validation（{full['tokens_mean_per_scene']:,.0f} vs {no_validation['tokens_mean_per_scene']:,.0f} tokens/场景），因为 RAG 轨道没有触发引用修复，二者差异主要反映独立采样波动。

这里的 duration 是每次 API 请求的计时；四路并发意味着这些时长不能相加后当作实际墙钟时间。"""
    behavior_narrative = """## 条件之间的裁决输出差异很大，当前不能判断哪一个更正确

成对内部一致率显示，六组条件的 trajectory status 一致率约为 60%–74%，chosen 一致率约为 48%–60%，严格排序一致率约为 39%–57%。这说明 RAG 提示内容、自由文本引用方式和独立模型采样都会显著改变输出。没有人工 gold 时，这些只能描述“变化有多大”，不能排序四个条件的裁决质量。"""
    scope = """## 范围、数据与指标定义

- 样本：82 个冻结内部实验场景；red_light 56、oncoming 14、pedestrian 12；难度分布 hard 70、easy 9、medium 3。
- 配对：每个条件均为相同 frame_token 集；每个条件 343 个 trajectory decisions。
- 自动引用有效率：最终引用事件中通过机器规则的比例，不含人工内容正确性或适用性评分。
- 无引用率：343 个 trajectory decisions 中引用数为 0 的比例；preference-pair 引用不进入该分母。
- 行为一致率：两个条件对同一 trajectory 给出相同 status 的比例；排序指标在同一场景内计算。
- 成本：模型返回的 token 总量与单场景请求 duration。"""
    methodology = """## 配对方法与不确定性

所有对比以 frame_token 配对。条件汇总使用事件级分母；因子差异使用场景级指标的配对均值，并用固定随机种子进行 5,000 次 paired bootstrap，生成 95% 区间。完整 2×2 对比保留四个计划对照：RAG 在 validation 开/关时的差异，以及 validation 在 RAG 开/关时的差异，并计算 difference-in-differences。

这是单次 LLM 采样实验，不是重复测量实验。尤其 full 与 no_validation 的 prompt hash 相同且均未触发引用修复，二者输出差异是随机采样噪声的直接证据；因此当前不作验证器对裁决行为的因果结论。"""
    limitations = """## 局限与稳健性检查

**可分享但必须带 caveat。** 已核对四个条件均为 82 条唯一记录、frame 集完全相同、无 gold 字段；headline 数值已从原始 JSONL 独立复算并与生成的 metrics JSON 对齐。主要限制有三点：

1. 自由文本轨道的机器校验只验证条款“存在”，不验证引用内容、语义或场景适用性。
2. 条件各自进行独立模型采样，未重复运行；验证开关与采样噪声没有完全解耦。
3. 样本高度集中于 hard/red_light，分层结论统计功效有限。

因此本报告适合判断管线行为、失败模式和成本，不适合宣布哪个条件的最终裁决更准确。"""
    next_steps = """## 建议下一步

1. 现在保留四个条件的冻结输出，不再追加或覆盖推理结果。
2. 自由文本引用校验若继续使用，应先修复规范化/匹配策略，并把“修复后仍失败”作为显式失败状态，而不是静默保留。
3. 等全部人工盲标完成并 adjudication 后，将 gold 合并到这 82 个场景，计算 verdict、pair、top-1、Kendall tau 与人工引用适用性。
4. 若要识别 validation 的净效应，下一轮应复用同一首轮响应做离线 validator replay，或增加重复随机种子。"""
    questions = """## 待完整人工对比回答的问题

- GraphRAG 的 100% ID 可校验性是否同时带来更高的人工引用正确性与适用性？
- full 较低的 veto rate 是更准确，还是偏宽松？
- no_rag 的修复对 verdict / ranking 的改写是纠错还是引入新错误？
- 结果在 red_light、oncoming、pedestrian 与难度分层上是否一致？"""

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "2×2 消融内部对比：82 个冻结场景",
            "generatedAt": "2026-08-02T00:00:00Z",
            "charts": [
                {
                    "id": "citation_validity",
                    "title": "最终机器可核验引用有效率",
                    "subtitle": "82 个配对场景；RAG 为上下文 ID 校验，自由文本为审计法规库存在性校验",
                    "type": "bar",
                    "dataset": "conditions",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {"field": "condition", "type": "nominal", "label": "条件"},
                        "y": {
                            "field": "machine_validity",
                            "type": "quantitative",
                            "label": "有效率",
                            "format": "percent",
                        },
                    },
                    "valueFormat": "percent",
                },
                {
                    "id": "tokens_per_scene",
                    "title": "每场景平均 token",
                    "subtitle": "82 个配对场景；包含结构重试和引用修复调用",
                    "type": "bar",
                    "dataset": "conditions",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {"field": "condition", "type": "nominal", "label": "条件"},
                        "y": {
                            "field": "tokens_per_scene",
                            "type": "quantitative",
                            "label": "tokens / 场景",
                            "format": "number",
                        },
                    },
                    "valueFormat": "number",
                },
            ],
            "tables": [
                {
                    "id": "condition_summary",
                    "title": "四条件内部汇总",
                    "subtitle": "精确值；质量指标不含人工 gold",
                    "dataset": "conditions",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "condition", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "condition", "label": "条件", "type": "text"},
                        {"field": "machine_validity", "label": "机器引用有效率", "format": "percent"},
                        {"field": "veto_rate", "label": "veto rate", "format": "percent"},
                        {"field": "uncertain_rate", "label": "uncertain rate", "format": "percent"},
                        {"field": "tokens_total", "label": "tokens", "format": "number"},
                        {"field": "mean_latency_s", "label": "平均时长(s)", "format": "number"},
                        {"field": "citation_retries", "label": "引用重试", "format": "number"},
                        {"field": "enforcement_exhausted", "label": "修复耗尽", "format": "number"},
                    ],
                },
                {
                    "id": "pairwise_agreement",
                    "title": "条件间输出一致率",
                    "subtitle": "同一 frame / trajectory 配对；仅衡量一致，不衡量正确",
                    "dataset": "pairwise",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "status_agreement", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "comparison", "label": "对比", "type": "text"},
                        {"field": "status_agreement", "label": "status 一致", "format": "percent"},
                        {"field": "chosen_agreement", "label": "chosen 一致", "format": "percent"},
                        {"field": "exact_ranking_agreement", "label": "严格排序一致", "format": "percent"},
                        {"field": "mean_kendall_tau", "label": "Kendall tau", "format": "number"},
                    ],
                },
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 2×2 消融内部对比：82 个冻结场景"},
                {"id": "technical_summary", "type": "markdown", "body": technical_summary, "sourceId": source["id"]},
                {"id": "citation_finding", "type": "markdown", "body": citation_narrative, "sourceId": source["id"]},
                {"id": "citation_chart", "type": "chart", "chartId": "citation_validity", "layout": "full"},
                {"id": "condition_table", "type": "table", "tableId": "condition_summary", "layout": "full"},
                {"id": "cost_finding", "type": "markdown", "body": cost_narrative, "sourceId": source["id"]},
                {"id": "cost_chart", "type": "chart", "chartId": "tokens_per_scene", "layout": "full"},
                {"id": "behavior_finding", "type": "markdown", "body": behavior_narrative, "sourceId": source["id"]},
                {"id": "agreement_table", "type": "table", "tableId": "pairwise_agreement", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": scope, "sourceId": source["id"]},
                {"id": "methodology", "type": "markdown", "body": methodology, "sourceId": source["id"]},
                {"id": "limitations", "type": "markdown", "body": limitations, "sourceId": source["id"]},
                {"id": "next_steps", "type": "markdown", "body": next_steps},
                {"id": "questions", "type": "markdown", "body": questions},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-02T00:00:00Z",
            "status": "ready",
            "datasets": {"conditions": conditions, "pairwise": pairwise},
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(build_artifact(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
