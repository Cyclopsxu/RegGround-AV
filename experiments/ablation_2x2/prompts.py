"""Prompt variants with only the two documented ablation factors changed."""

from __future__ import annotations

import json
from typing import Any

from experiments.ablation_2x2.conditions import AblationCondition
from src.layer3.leakage_guard import LeakageGuard

SYSTEM_PROMPT = (
    "你是中国道路交通法规审计员。逐条判断候选轨迹，输出严格 JSON；不得补充输入中不存在的场景事实。"
)

_NO_RAG_INSTRUCTION = (
    "依据你所知的中国道路交通安全法及相关法规判定，并必须为每个判定引用"
    "具体法律条款（法律名称 + 条号），写入输出的 cited_provisions 字段。"
)


def build_prompt(condition: AblationCondition, scenario: dict[str, Any]) -> tuple[str, str]:
    digest = str(scenario["digest_v_a"])
    candidates = list(scenario["candidates"])
    sections = [f"## 场景要件（digest v_a）\n{digest}"]
    if condition.graph_rag:
        rule_lines = [
            f"[{rule['rule_id']}] {rule['code']} {rule['description']}"
            for rule in scenario.get("retrieved_rules", [])
        ]
        sections.append("## 可引用规则（GraphRAG 检索结果）\n" + "\n".join(rule_lines))
    if scenario.get("scene_description"):
        sections.append(f"## 场景描述\n{scenario['scene_description']}")
    sections.append(
        "## 候选特征\n"
        + "\n".join(f"[{item['trajectory_id']}] {item['features']}" for item in candidates)
    )
    citation_field = "cited_rule_ids" if condition.graph_rag else "cited_provisions"
    instruction = "引用只能使用上方给出的 rule_id。" if condition.graph_rag else _NO_RAG_INSTRUCTION
    schema = {
        "verdicts": [
            {
                "trajectory_id": "traj_a",
                "status": "cleared|vetoed|uncertain",
                "reason": "...",
                citation_field: [],
            }
        ],
        "preference_ranking": ["traj_a"],
        "chosen_trajectory_id": "traj_a or null",
        "preference_pairs": [
            {
                "preferred_id": "traj_a",
                "dispreferred_id": "traj_b",
                "reason": "...",
                citation_field: [],
            }
        ],
    }
    sections.append(
        f"## 判定要求\n{instruction}\n输出结构：{json.dumps(schema, ensure_ascii=False)}"
    )
    user_prompt = "\n\n".join(sections)
    LeakageGuard.assert_prompt_safe(SYSTEM_PROMPT, user_prompt)
    return SYSTEM_PROMPT, user_prompt


__all__ = ["SYSTEM_PROMPT", "build_prompt"]
