"""自然语言审计摘要生成器。"""

from __future__ import annotations

import logging

from src.layer1.models import JudgeInput
from src.layer2.models import RuleSubgraph
from src.layer3.leakage_guard import LeakageGuard
from src.layer3.llm_client import LLMClient
from src.layer3.models import (
    Layer3Settings,
    PairwisePreference,
    ReasoningStep,
    TrajectoryVerdict,
    VetoStatus,
)
from src.layer3.prompt_loader import load_system_prompt

logger = logging.getLogger(__name__)


class LabelGenerator:
    """LLM 摘要生成 + 模板降级。"""

    def __init__(
        self,
        settings: Layer3Settings,
        llm_client: LLMClient,
    ) -> None:
        self._settings = settings
        self._llm = llm_client

    def generate(
        self,
        judge_input: JudgeInput,
        subgraph: RuleSubgraph,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
        reasoning_chain: list[ReasoningStep],
    ) -> str:
        return self._generate_with_llm(
            judge_input,
            subgraph,
            verdicts,
            preference_pairs,
            reasoning_chain,
        )

    def fallback_summary(
        self,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
    ) -> str:
        return self._fallback_summary(verdicts, preference_pairs)

    def _generate_with_llm(
        self,
        judge_input: JudgeInput,
        subgraph: RuleSubgraph,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
        reasoning_chain: list[ReasoningStep],
    ) -> str:
        system_prompt = load_system_prompt("label_generator")
        user_prompt = (
            f"## 场景信息\n"
            f"scene_id: {judge_input.scene_id}\n"
            f"frame_token: {judge_input.frame_token}\n"
            f"场景描述: {judge_input.narrative}\n\n"
            f"## 法规参考\n{self._build_rules_text(subgraph)}\n\n"
            f"## 裁决结果\n{self._build_verdict_text(verdicts)}\n\n"
            f"## 偏好对\n{self._build_pair_text(preference_pairs)}\n\n"
            f"## 推理摘要\n{self._build_reasoning_text(reasoning_chain)}\n\n"
            "请按指定四段结构生成自然语言审计摘要。"
        )
        LeakageGuard.assert_runtime_text_safe(
            judge_input.narrative,
            *(feature.natural_language_summary for feature in judge_input.trajectory_features),
        )
        text, _tokens = self._llm.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self._settings.label_temperature,
        )
        return text.strip()

    @staticmethod
    def _build_rules_text(subgraph: RuleSubgraph) -> str:
        lines: list[str] = []
        for rule in subgraph.active_rules:
            code = f" ({rule.code})" if rule.code and rule.code != rule.node_id else ""
            lines.append(f"[{rule.node_id}]{code} {rule.description}")
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def _build_verdict_text(verdicts: list[TrajectoryVerdict]) -> str:
        lines: list[str] = []
        for verdict in verdicts:
            refs = ", ".join(f"[{rid}]" for rid in verdict.veto.violated_rule_ids)
            refs_text = f" refs={refs}" if refs else ""
            rank_text = f" rank={verdict.rank}" if verdict.rank is not None else ""
            lines.append(
                f"{verdict.trajectory_id}: status={verdict.veto.status.value}"
                f"{rank_text}{refs_text}; reason={verdict.veto.reason}"
            )
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def _build_pair_text(preference_pairs: list[PairwisePreference]) -> str:
        if not preference_pairs:
            return "（无）"
        lines = []
        for pair in preference_pairs:
            refs = ", ".join(f"[{rid}]" for rid in pair.referenced_rule_ids)
            refs_text = f" refs={refs}" if refs else ""
            lines.append(
                f"{pair.preferred_id} > {pair.dispreferred_id}; "
                f"basis={pair.basis}; confidence={pair.confidence:.2f}; "
                f"reason={pair.reasoning}{refs_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_reasoning_text(reasoning_chain: list[ReasoningStep]) -> str:
        if not reasoning_chain:
            return "（无）"
        return "\n".join(f"- {step.stage.value}: {step.content}" for step in reasoning_chain)

    @staticmethod
    def _fallback_summary(
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
    ) -> str:
        cleared = [v for v in verdicts if v.veto.status == VetoStatus.CLEARED]
        vetoed = [v for v in verdicts if v.veto.status == VetoStatus.VETOED]
        uncertain = [v for v in verdicts if v.veto.status == VetoStatus.UNCERTAIN]

        parts: list[str] = []
        ranked_cleared = sorted(
            cleared,
            key=lambda v: v.rank if v.rank is not None else 10_000,
        )

        if ranked_cleared:
            parts.append(f"结论：chosen 轨迹为 {ranked_cleared[0].trajectory_id}。")
        else:
            parts.append("结论：没有可选择的 CLEARED 轨迹，chosen 为空。")

        if vetoed:
            veto_lines = ["否决说明："]
            for verdict in vetoed:
                refs = " ".join(f"[{rid}]" for rid in verdict.veto.violated_rule_ids)
                veto_lines.append(
                    f"- {verdict.trajectory_id}: {verdict.veto.reason} {refs}".rstrip()
                )
            parts.append("\n".join(veto_lines))
        else:
            parts.append("否决说明：无 VETOED 轨迹。")

        if uncertain:
            parts.append(
                "不确定说明："
                + "、".join(v.trajectory_id for v in uncertain)
                + " 未形成闭环裁定，不作为 chosen。"
            )
        else:
            parts.append("不确定说明：无 UNCERTAIN 轨迹。")

        pairwise = [p for p in preference_pairs if p.basis == "pairwise_judgment"]
        if pairwise:
            lines = ["偏好说明："]
            for pair in pairwise:
                refs = " ".join(f"[{rid}]" for rid in pair.referenced_rule_ids)
                lines.append(
                    f"- {pair.preferred_id} 优于 {pair.dispreferred_id}: "
                    f"{pair.reasoning} {refs}".rstrip()
                )
            parts.append("\n".join(lines))
        elif ranked_cleared:
            parts.append(
                "偏好说明：CLEARED 轨迹按 pairwise 可用结果或输入顺序排列。"
            )
        else:
            parts.append("偏好说明：无 CLEARED 内部偏好可报告。")

        return "\n\n".join(parts)
