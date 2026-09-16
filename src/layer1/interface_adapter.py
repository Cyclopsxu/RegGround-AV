"""Layer 1 到下游层的窄接口适配。"""

from __future__ import annotations

from src.layer1.leakage_guard import LeakageGuard
from src.layer1.models import JudgeInput, SceneContext, SceneFactsDigest, SceneQuery


class InterfaceAdapter:
    """从 SceneContext 生成 SceneQuery 与 JudgeInput。"""

    def to_scene_query(self, context: SceneContext) -> SceneQuery:
        """生成 Layer 2 入参，不包含候选轨迹坐标。"""
        return SceneQuery(
            scene_id=context.scene_id,
            frame_token=context.frame_token,
            scenario_type=context.scenario_type,
            keywords=list(context.description.keywords),
            scene_facts=context.scene_facts,
        )

    def to_judge_input(self, context: SceneContext) -> JudgeInput:
        """生成 Layer 3 入参，并扫描最终 prompt 可见片段。"""
        LeakageGuard.assert_safe_prompt_fragment(context.description.narrative)
        for features in context.trajectory_features:
            LeakageGuard.assert_safe_prompt_fragment(features.natural_language_summary)
            LeakageGuard.assert_safe_prompt_fragment(features.conflict_zone_behavior)

        facts = context.scene_facts
        digest = SceneFactsDigest(
            signal_phase="red" if facts.has_red_light else "unknown",
            phase_source=facts.traffic_light_status_source,
            ego_turn_intent=facts.ego_turn_intent,
            governing_stop_line_signed_m=facts.governing_stop_line_signed_m,
            pedestrian_in_forward_crosswalk=facts.pedestrian_in_forward_crosswalk,
            pedestrian_moving=facts.pedestrian_moving,
            oncoming_vehicle_moving=facts.oncoming_vehicle_moving,
            location_is_intersection=facts.location_is_intersection,
        )
        return JudgeInput(
            scene_id=context.scene_id,
            frame_token=context.frame_token,
            narrative=context.description.narrative,
            scene_facts_digest=digest,
            trajectory_features=list(context.trajectory_features),
        )
