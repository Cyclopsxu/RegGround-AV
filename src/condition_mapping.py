"""跨层共享的事实与法规条件映射。"""

from typing import Protocol


class SceneFactsLike(Protocol):
    """共享映射需要的最小事实接口。"""

    has_red_light: bool
    has_yellow_light: bool
    pedestrian_in_crosswalk: bool
    oncoming_vehicle_moving: bool
    ego_on_minor_road: bool | None
    emergency_vehicle_active: bool

FACT_TO_CONDITION: dict[str, str] = {
    "has_red_light": "C-RED-PHASE",
    "has_yellow_light": "C-YELLOW-PHASE",
    "pedestrian_in_crosswalk": "C-PED-IN-CROSSWALK",
    "oncoming_vehicle_moving": "C-ONCOMING-STRAIGHT",
    "ego_on_minor_road": "C-ON-MINOR-ROAD",
}

SCENARIO_TO_FACT: dict[str, str] = {
    "red_light": "has_red_light",
    "yellow_light": "has_yellow_light",
    "pedestrian": "pedestrian_in_crosswalk",
    "oncoming": "oncoming_vehicle_moving",
    "minor_road": "ego_on_minor_road",
}


def would_retrieve_hard_rule(facts: SceneFactsLike, scenario_type: str) -> bool:
    """判断场景类型是否有同源结构化事实支撑。"""
    fact_key = SCENARIO_TO_FACT.get(str(scenario_type))
    return fact_key is not None and getattr(facts, fact_key, False) is True
