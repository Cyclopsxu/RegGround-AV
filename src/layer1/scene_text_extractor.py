"""场景文本提取器。

从 DriveLM QA JSON（封装在 RawScene.raw_json 中）提取场景语义描述，
并通过关键词/状态匹配推断场景的法规触发类别（ScenarioType）。
"""

from __future__ import annotations

from typing import Any

from src.layer1.exceptions import SceneParseError
from src.layer1.leakage_guard import LeakageGuard
from src.layer1.models import (
    EgoState,
    LocationType,
    RawScene,
    ScenarioType,
    SceneDescription,
    SceneFacts,
)

# =============================================================================
# 关键词映射表（§5.2.1）
# =============================================================================

SCENARIO_KEYWORD_MAP: dict[ScenarioType, list[str]] = {
    ScenarioType.RED_LIGHT: [
        "red light", "traffic light", "stop line", "red signal", "红灯",
    ],
    ScenarioType.YELLOW_LIGHT: [
        "yellow light", "amber light", "yellow signal", "黄灯",
    ],
    ScenarioType.PEDESTRIAN: [
        "pedestrian", "crosswalk", "crossing", "zebra crossing", "行人",
    ],
    ScenarioType.ONCOMING: [
        "oncoming", "opposite direction", "left turn", "yield", "对向",
    ],
    ScenarioType.MINOR_ROAD: [
        "minor road", "side road", "merge", "main road", "支路",
    ],
    ScenarioType.EMERGENCY: [
        "ambulance", "police car", "fire truck", "emergency vehicle", "siren", "救护车",
    ],
}

# 位置关键词
LOCATION_KEYWORDS: dict[LocationType, list[str]] = {
    LocationType.INTERSECTION: [
        "intersection", "crossroad", "junction", "交叉口", "路口", "十字路口", "丁字路口",
    ],
    LocationType.URBAN_ROAD:   ["urban road", "road segment", "城市道路", "道路", "路段"],
}


# =============================================================================
# SceneTextExtractor
# =============================================================================


class SceneTextExtractor:
    """从 RawScene 的 DriveLM QA 数据中提取场景语义文本与场景类型。

    纯规则推断，无 LLM 调用。
    信号灯类场景依赖 DriveLM key_object_infos 中的红绿灯 status 字段；
    当 status 不可得时返回 UNKNOWN，不抛异常。
    """

    def __init__(self, max_narrative_chars: int = 600) -> None:
        """初始化提取器。

        Args:
            max_narrative_chars: narrative 最大字符数，超出截断。
        """
        self.max_narrative_chars = max_narrative_chars

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def extract(
        self,
        raw_scene: RawScene,
        facts: SceneFacts | None = None,
    ) -> SceneDescription:
        """从 RawScene 提取完整场景语义描述。

        依次提取 narrative、keywords、location_type、scenario_type，
        组装为 SceneDescription 返回。

        Args:
            raw_scene: 含 DriveLM QA 数据的原始场景对象。

        Returns:
            填充完整的 SceneDescription。

        Raises:
            SceneParseError: raw_json 结构不可解析。
        """
        raw = raw_scene.raw_json
        if not raw:
            raise SceneParseError(f"raw_json 为空，无法提取场景描述: {raw_scene.scene_id}")

        narrative = self._build_narrative(raw, facts)
        keywords = self._extract_keywords(raw)
        location_type = self._infer_location_type(narrative, keywords)
        return SceneDescription(
            narrative=narrative,
            keywords=keywords,
            location_type=location_type,
        )

    def extract_ego_state(self, raw_scene: RawScene) -> EgoState:
        """从 RawScene 提取自车状态快照。

        优先从 ego_poses 帧间位移估算速度；若不可得则尝试从 QA 文本解析。

        Args:
            raw_scene: 原始场景对象。

        Returns:
            EgoState 对象（speed_mps 可能为 None）。
        """
        speed: float | None = None

        # 1. 尝试从 ego_pose 帧间位移计算速度
        poses = raw_scene.ego_poses
        ki = raw_scene.keyframe_index
        if len(poses) > 1 and ki < len(poses) - 1:
            try:
                p0 = poses[ki]
                p1 = poses[ki + 1]
                t0 = p0.get("translation", [0, 0, 0])
                t1 = p1.get("translation", [0, 0, 0])
                dt_us = (p1.get("timestamp", 0) - p0.get("timestamp", 0))
                if dt_us > 0:
                    dx = t1[0] - t0[0]
                    dy = t1[1] - t0[1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    speed = dist / (dt_us / 1_000_000)
            except (IndexError, KeyError, TypeError, ZeroDivisionError):
                pass

        # 2. 若不可得，尝试从 QA 文本解析
        if speed is None:
            speed = self._parse_speed_from_text(raw_scene.raw_json)

        return EgoState(speed_mps=speed)

    def infer_scenario_type(
        self,
        raw_scene: RawScene,
        facts: SceneFacts | None = None,
    ) -> ScenarioType:
        """从结构化事实、narrative、keywords 及 key_object_infos 推断场景类别。

        匹配规则：
        1. 若 SceneFacts 已有结构化事实，优先使用事实。
        2. 信号灯类（RED_LIGHT / YELLOW_LIGHT）：检查 key_object_infos 中
           红绿灯关键对象的 status 字段。若 status 明确标注 "red" / "yellow"，
           直接返回对应类型。若 status 不可得，则由关键词兜底匹配；
           若关键词也匹配不到，返回 UNKNOWN。
        3. 让行类（PEDESTRIAN / ONCOMING / MINOR_ROAD / EMERGENCY）：
           纯关键词匹配。

        优先级策略（显式，不依赖枚举定义顺序）：
        - 信号灯类 > 让行类：真实场景中若同时触发信号灯与让行约束，
          信号灯是主导法规触发条件，Layer 2 以信号灯为优先入口。
        - 同类多命中时按 ScenarioType 枚举定义顺序取值。

        Args:
            raw_scene: 原始场景对象。

        Returns:
            匹配的 ScenarioType；无匹配返回 UNKNOWN。
        """
        if facts is not None:
            if facts.has_red_light:
                return ScenarioType.RED_LIGHT
            if facts.has_yellow_light:
                return ScenarioType.YELLOW_LIGHT
            if facts.pedestrian_in_forward_crosswalk:
                return ScenarioType.PEDESTRIAN
            if facts.oncoming_vehicle_moving and facts.ego_turn_intent == "left":
                return ScenarioType.ONCOMING
            if facts.ego_on_minor_road is True:
                return ScenarioType.MINOR_ROAD
            if facts.emergency_vehicle_active:
                return ScenarioType.EMERGENCY
            # 关键词只能作为检索提示，不能生成法规场景标签。
            return ScenarioType.UNKNOWN

        raw = raw_scene.raw_json
        if not raw:
            return ScenarioType.UNKNOWN

        # --- 信号灯类：优先检查 key_object_infos 中的红绿灯 status ---
        key_objs = raw.get("key_object_infos", {})
        if isinstance(key_objs, dict):
            for obj_info in key_objs.values():
                if not isinstance(obj_info, dict):
                    continue
                category = obj_info.get("category", "").lower()
                # 红绿灯相关对象
                if category in ("traffic_light", "traffic light", "红绿灯", "信号灯"):
                    status = str(obj_info.get("Status", obj_info.get("status", ""))).lower()
                    if status in ("red", "红灯"):
                        return ScenarioType.RED_LIGHT
                    if status in ("yellow", "黄灯"):
                        return ScenarioType.YELLOW_LIGHT
                    # status 存在但非红/黄，继续检查其他红绿灯对象
                    continue

        return ScenarioType.UNKNOWN

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------

    # 信号灯类与让行类的显式分组（用于关键词匹配优先级，不依赖枚举定义顺序）
    _SIGNAL_TYPES: frozenset[ScenarioType] = frozenset({
        ScenarioType.RED_LIGHT, ScenarioType.YELLOW_LIGHT,
    })
    _YIELD_TYPES: frozenset[ScenarioType] = frozenset({
        ScenarioType.PEDESTRIAN, ScenarioType.ONCOMING,
        ScenarioType.MINOR_ROAD, ScenarioType.EMERGENCY,
    })

    @classmethod
    def _match_by_keywords(cls, combined_text: str) -> ScenarioType:
        """从文本中收集所有命中类型，按显式优先级返回。

        优先级策略：
        - 信号灯类（RED_LIGHT / YELLOW_LIGHT）> 让行类（PEDESTRIAN / …）。
          真实场景中若同时触发信号灯与让行约束，信号灯是主导法规触发条件，
          Layer 2 以信号灯为优先入口；让行类可由 Layer 2/3 在子条件中细化。
        - 同类多命中时按 ScenarioType 枚举定义顺序取值。

        Args:
            combined_text: narrative + keywords 拼接文本。

        Returns:
            匹配的 ScenarioType；无匹配返回 UNKNOWN。
        """
        # 第一遍：收集所有命中类型（每类只记录首次命中）
        hits: list[ScenarioType] = []
        for scenario_type in ScenarioType:
            if scenario_type == ScenarioType.UNKNOWN:
                continue
            kw_list = SCENARIO_KEYWORD_MAP.get(scenario_type, [])
            for kw in kw_list:
                if kw.lower() in combined_text:
                    hits.append(scenario_type)
                    break  # 每个类型只记录一次，继续检查下一类型

        # 第二遍：按显式优先级分组返回
        signal_hits = [h for h in hits if h in cls._SIGNAL_TYPES]
        if signal_hits:
            return signal_hits[0]  # 多个信号灯命中取枚举序靠前者

        yield_hits = [h for h in hits if h in cls._YIELD_TYPES]
        if yield_hits:
            return yield_hits[0]   # 多个让行命中取枚举序靠前者

        return ScenarioType.UNKNOWN

    def _build_narrative(
        self,
        raw: dict[str, Any],
        facts: SceneFacts | None = None,
    ) -> str:
        """仅使用感知描述与结构化事实构建场景叙述。"""
        parts: list[str] = []
        explicit_description = raw.get(
            "drivelm_scene_description",
            raw.get("scene_description", ""),
        )
        if isinstance(explicit_description, str) and explicit_description.strip():
            parts.append(explicit_description.strip())

        # planning / behavior 类答案含规划结论，整类排除。
        qa_pairs = raw.get("qa_pairs", [])
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                if not isinstance(qa, dict):
                    continue
                category = str(qa.get("category", "perception")).lower()
                answer = qa.get("answer", "")
                cleaned_answer = " ".join(str(answer).split())
                if (
                    category == "perception"
                    and cleaned_answer.lower().rstrip(".") not in {"", "yes", "no"}
                ):
                    parts.append(cleaned_answer)

        perception = raw.get("perception", "")
        if isinstance(perception, str) and perception.strip():
            parts.append(perception.strip())

        if facts is not None:
            if facts.location_is_intersection:
                parts.append("The ego vehicle is approaching an intersection.")
            if facts.oncoming_vehicle_moving:
                parts.append("A moving vehicle is approaching from the opposite direction.")
            if facts.pedestrian_in_crosswalk:
                parts.append("A pedestrian is currently in the crosswalk.")
            if facts.has_red_light:
                parts.append("The traffic light for the ego direction is red.")
            if facts.has_yellow_light:
                parts.append("The traffic light for the ego direction is yellow.")
            if facts.traffic_light_status_source == "none":
                parts.append("No traffic light state information is available.")

        narrative = LeakageGuard.scrub_text(self._join_with_limit(parts))
        LeakageGuard.assert_safe_prompt_fragment(narrative)
        return narrative

    def _join_with_limit(self, parts: list[str]) -> str:
        """按完整片段拼接，避免在句子中间硬截断。"""
        accepted: list[str] = []
        used = 0
        for part in parts:
            cleaned = " ".join(part.split())
            extra = len(cleaned) + (1 if accepted else 0)
            if used + extra > self.max_narrative_chars:
                continue
            accepted.append(cleaned)
            used += extra
        return " ".join(accepted)

    def _extract_keywords(self, raw: dict[str, Any]) -> list[str]:
        """从 narrative 与 QA 中提取关键词。

        基于 SCENARIO_KEYWORD_MAP 中所有关键词做命中检测。

        Args:
            raw: raw_json 字典。

        Returns:
            去重后的关键词列表。
        """
        text_parts: list[str] = []
        qa_pairs = raw.get("qa_pairs", [])
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                if isinstance(qa, dict):
                    text_parts.extend(str(qa.get(key, "")) for key in ("question", "answer"))
        text_parts.extend(str(raw.get(key, "")) for key in ("perception", "prediction"))
        narrative = " ".join(text_parts).lower()
        found: list[str] = []
        seen: set[str] = set()

        # 从所有场景类型关键词中检查命中
        for kw_list in SCENARIO_KEYWORD_MAP.values():
            for kw in kw_list:
                key = kw.lower()
                if key not in seen and key in narrative:
                    found.append(kw)
                    seen.add(key)

        # 同时检查 raw 中直接的 keywords 字段
        explicit_kws = raw.get("keywords", [])
        if isinstance(explicit_kws, list):
            for kw in explicit_kws:
                if isinstance(kw, str) and kw.lower() not in seen:
                    found.append(kw)
                    seen.add(kw.lower())

        return found

    @staticmethod
    def _infer_location_type(narrative: str, keywords: list[str]) -> LocationType:
        """从 narrative 和 keywords 推断位置类型。

        Args:
            narrative: 叙述文本。
            keywords: 关键词列表。

        Returns:
            推断的 LocationType。
        """
        combined = (narrative + " " + " ".join(keywords)).lower()
        for loc_type, kw_list in LOCATION_KEYWORDS.items():
            for kw in kw_list:
                if kw.lower() in combined:
                    return loc_type
        return LocationType.UNKNOWN

    @staticmethod
    def _parse_speed_from_text(raw: dict[str, Any]) -> float | None:
        """从 QA 文本中尝试解析自车速度数值。

        简单正则匹配 "XX km/h" 或 "XX m/s" 模式。

        Args:
            raw: raw_json 字典。

        Returns:
            解析出的速度（m/s），解析失败返回 None。
        """
        import re

        # 拼接所有文本
        text_parts: list[str] = []
        qa_pairs = raw.get("qa_pairs", [])
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                if isinstance(qa, dict):
                    for field in ("question", "answer"):
                        val = qa.get(field, "")
                        if isinstance(val, str):
                            text_parts.append(val)

        # 也检查 ego_speed 字段
        ego_speed = raw.get("ego_speed")
        # bool 是 int 的子类，需显式排除避免 True→1.0 / False→0.0
        if isinstance(ego_speed, (int, float)) and not isinstance(ego_speed, bool):
            return float(ego_speed)

        text = " ".join(text_parts)
        if not text:
            return None

        # 匹配 "XX km/h"
        match = re.search(r"(\d+(?:\.\d+)?)\s*km/h", text)
        if match:
            return float(match.group(1)) / 3.6

        # 匹配 "XX m/s"
        match = re.search(r"(\d+(?:\.\d+)?)\s*m/s", text)
        if match:
            return float(match.group(1))

        return None
