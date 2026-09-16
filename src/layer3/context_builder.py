"""评判上下文构建器。"""

from __future__ import annotations

import logging

from src.layer1.models import JudgeInput, SceneFactsDigest, TrajectoryFeatures
from src.layer2.models import RuleNode, RuleSubgraph, Severity
from src.layer3.exceptions import ContextTooLargeError
from src.layer3.leakage_guard import LeakageGuard
from src.layer3.models import JudgeContext, Layer3Settings

logger = logging.getLogger(__name__)

_CHAR_PER_TOKEN_ESTIMATE = 2.5


class ContextBuilder:
    """将 JudgeInput 与 RuleSubgraph 拼装为 JudgeContext。"""

    def __init__(self, settings: Layer3Settings) -> None:
        self._settings = settings

    def build(self, judge_input: JudgeInput, subgraph: RuleSubgraph) -> JudgeContext:
        active_rules = list(subgraph.active_rules)
        hard_rules = [
            r for r in active_rules if r.severity == Severity.HARD
        ][: self._settings.max_rules_in_context]
        soft_rules = [
            r for r in active_rules if r.severity == Severity.SOFT
        ][: self._settings.max_rules_in_context]

        hard_rules = self._sort_rules(hard_rules)
        soft_rules = self._sort_rules(soft_rules)
        features = judge_input.trajectory_features[: self._settings.max_trajectories_per_judge]
        trajectory_ids = self._trajectory_ids(len(features))

        scene_narrative = judge_input.narrative or f"场景 {judge_input.scene_id}"
        scene_facts_text = self.render_scene_facts_digest(judge_input.scene_facts_digest)
        hard_rules_text = self._format_rules(hard_rules, "硬性规则")
        soft_rules_text = self._format_rules(soft_rules, "软性规则")
        trajectories_text = self._format_trajectories(features, trajectory_ids)

        hard_rule_ids = [r.node_id for r in hard_rules]
        soft_rule_ids = [r.node_id for r in soft_rules]
        hard_rule_categories = [r.category.value for r in hard_rules]
        soft_rule_categories = [r.category.value for r in soft_rules]

        estimated_tokens = self._estimate_tokens(
            scene_narrative,
            scene_facts_text,
            hard_rules_text,
            soft_rules_text,
            trajectories_text,
        )
        if estimated_tokens > self._settings.max_prompt_tokens:
            logger.warning(
                "Context estimated %d tokens exceeds budget, dropping soft rules",
                estimated_tokens,
            )
            soft_rules_text = ""
            soft_rule_ids = []
            soft_rule_categories = []
            estimated_tokens = self._estimate_tokens(
                scene_narrative,
                scene_facts_text,
                hard_rules_text,
                trajectories_text,
            )
            if estimated_tokens > self._settings.max_prompt_tokens:
                raise ContextTooLargeError(
                    f"Context too large after truncation: ~{estimated_tokens} tokens"
                )

        context = JudgeContext(
            scene_id=judge_input.scene_id,
            frame_token=judge_input.frame_token,
            scene_narrative=scene_narrative,
            scene_facts_digest=judge_input.scene_facts_digest,
            scene_facts_text=scene_facts_text,
            hard_rules_text=hard_rules_text,
            soft_rules_text=soft_rules_text,
            trajectories_text=trajectories_text,
            hard_rule_ids=hard_rule_ids,
            soft_rule_ids=soft_rule_ids,
            hard_rule_categories=hard_rule_categories,
            soft_rule_categories=soft_rule_categories,
            trajectory_ids=trajectory_ids,
            total_prompt_tokens=estimated_tokens,
        )

        self._assert_context_prompt_safe(context)
        return context

    @staticmethod
    def render_scene_facts_digest(digest: SceneFactsDigest) -> str:
        """以中性措辞渲染 Layer 3 可见的法规触发事实。"""
        phase = {
            "red": "红灯",
            "green": "绿灯",
            "unknown": "未知",
        }[digest.signal_phase]
        source = "DriveLM 标注" if digest.phase_source == "drivelm_status" else "无"
        distance = (
            "无本向停止线"
            if digest.governing_stop_line_signed_m is None
            else f"{digest.governing_stop_line_signed_m:.1f} m（负值表示起点已越过）"
        )
        moving = (
            "未知" if digest.pedestrian_moving is None
            else ("是" if digest.pedestrian_moving else "否")
        )
        intersection = (
            "未知" if digest.location_is_intersection is None
            else ("是" if digest.location_is_intersection else "否")
        )
        return "\n".join([
            "[场景要件]",
            f"当前方向信号相位: {phase}（来源: {source}）",
            "信号灯相位为 keyframe 时刻（t=0）快照，其后相位未知",
            f"自车转向意图: {digest.ego_turn_intent}",
            f"本向停止线有符号距离: {distance}",
            "前进路径相交横道内有行人: "
            f"{'是' if digest.pedestrian_in_forward_crosswalk else '否'}",
            f"该行人正在移动: {moving}",
            f"存在移动中的对向车辆: {'是' if digest.oncoming_vehicle_moving else '否'}",
            f"位置为路口: {intersection}",
        ]) + "\n"

    @staticmethod
    def _sort_rules(rules: list[RuleNode]) -> list[RuleNode]:
        def priority(rule: RuleNode) -> tuple[int, str]:
            max_consequence = max(
                (c.severity_score for c in rule.consequences),
                default=0,
            )
            return (-max_consequence, rule.node_id)

        return sorted(rules, key=priority)

    @staticmethod
    def _format_rules(rules: list[RuleNode], label: str) -> str:
        if not rules:
            return f"[{label}]\n（无）\n"

        lines = [f"[{label}]"]
        for rule in rules:
            actors = "、".join(a.type.value for a in rule.actors) if rule.actors else "通用"
            consequences = (
                "；".join(
                    f"{c.type}({c.severity_score}/10)" for c in rule.consequences
                )
                if rule.consequences
                else "无具体后果描述"
            )
            code = f" {rule.code}" if rule.code and rule.code != rule.node_id else ""
            lines.append(
                f"[{rule.node_id}]{code} {rule.description}"
                f"（适用于: {actors}，后果: {consequences}）"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_trajectories(
        features: list[TrajectoryFeatures],
        trajectory_ids: list[str],
    ) -> str:
        if not features:
            return "[候选轨迹特征]\n（无）\n"

        lines = ["[候选轨迹特征]"]
        for traj_id, feature in zip(trajectory_ids, features, strict=False):
            summary = feature.natural_language_summary or "无可用轨迹特征摘要"
            parts = [
                f"[{traj_id}] {summary}",
                (
                    f"速度: max={feature.max_speed_mps:.2f} m/s, "
                    f"min={feature.min_speed_mps:.2f} m/s, "
                    f"mean={feature.mean_speed_mps:.2f} m/s"
                ),
                (
                    f"空间: distance={feature.total_distance_m:.2f} m, "
                    f"max_lateral_offset={feature.max_lateral_offset_m:.2f} m, "
                    f"point_count={feature.n_waypoints}"
                ),
            ]
            if feature.conflict_zone_behavior:
                parts.append(f"冲突区行为: {feature.conflict_zone_behavior}")
            lines.append("；".join(parts))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _trajectory_ids(count: int) -> list[str]:
        return [f"traj_{ContextBuilder._alpha_suffix(i)}" for i in range(count)]

    @staticmethod
    def _alpha_suffix(index: int) -> str:
        chars: list[str] = []
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            chars.append(chr(ord("a") + remainder))
        return "".join(reversed(chars))

    @staticmethod
    def _estimate_tokens(*parts: str) -> int:
        return int(sum(len(part) for part in parts) / _CHAR_PER_TOKEN_ESTIMATE)

    @staticmethod
    def _assert_context_prompt_safe(context: JudgeContext) -> None:
        LeakageGuard.assert_runtime_text_safe(
            context.scene_narrative,
            context.scene_facts_text,
            context.trajectories_text,
        )
