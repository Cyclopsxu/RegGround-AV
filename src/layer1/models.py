"""Layer 1 数据模型定义。

本模块定义 Layer 1（数据解析层）全部公共数据模型、枚举与内部中间类型。
所有模型使用 Pydantic v2，strict=True，extra="forbid"（RawScene 除外）。
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# 枚举定义（§4.1）
# =============================================================================


class TrajectorySource(StrEnum):
    """轨迹来源。"""
    PREDICTION = "prediction"
    PLANNING = "planning"       # 由 ego_pose 重建的真实轨迹归此类
    SYNTHETIC = "synthetic"     # TrajectoryAugmentor 合成


class QACategory(StrEnum):
    """DriveLM QA 类别。"""
    PERCEPTION = "perception"
    PREDICTION = "prediction"
    PLANNING = "planning"
    BEHAVIOR = "behavior"


class LocationType(StrEnum):
    """场景位置类型。"""
    INTERSECTION = "intersection"
    URBAN_ROAD = "urban_road"
    UNKNOWN = "unknown"


class ScenarioType(StrEnum):
    """v3.1 场景法规触发类别，对齐 Layer 2 图谱条件。

    用途：
    (1) 指导 augmentor 注入正确违规；
    (2) 供 Layer 2 条件匹配先验。

    不得用于向 LLM 透露合规性结论。
    """
    # 信号灯类（对应 Layer 2 的 C-RED-PHASE / C-YELLOW-PHASE）
    RED_LIGHT = "red_light"
    YELLOW_LIGHT = "yellow_light"
    # 让行类（对应 C-PED-IN-CROSSWALK / C-ONCOMING-STRAIGHT /
    #          C-ON-MINOR-ROAD / C-EMERGENCY-ACTIVE）
    PEDESTRIAN = "pedestrian"
    ONCOMING = "oncoming"
    MINOR_ROAD = "minor_road"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"   # 无法推断（含红绿灯相位不可得），augmentor 用通用扰动


class TrajectoryVariantType(StrEnum):
    """[严格隔离] 合成轨迹变体标签。

    仅在 augmentor 内部与 benchmark 脚本使用，
    绝不传入 Layer 3 任何 prompt 构建函数。
    """
    GROUND_TRUTH = "ground_truth"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ILLEGAL = "illegal"
    SUBOPTIMAL = "suboptimal"
    HARD_CASE = "hard_case"


ExcludeReason = Literal[
    "not_evaluable_short_window",
    "injection_degenerate",
    "no_violation_injectable",
    "predicate_not_decidable",
    "gt_precheck_not_evaluable",
]


# =============================================================================
# 基础数据模型
# =============================================================================


class Waypoint(BaseModel):
    """轨迹航点（§4.2）。

    坐标系：以关键帧自车位姿为原点，
    x = 纵向（车头方向，前进为正），y = 横向（左为正）。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    x: float = Field(ge=-250.0, le=250.0, description="纵向位移（前进为正），范围 -250~+250 m")
    y: float = Field(ge=-250.0, le=250.0, description="横向位移（左为正），范围 -250~+250 m")
    t: float = Field(default=0.0, ge=0.0, description="相对关键帧的秒数，>= 0")


class Trajectory(BaseModel):
    """轨迹对象（§4.3）。

    不携带 TrajectoryVariantType。变体标签只存在于 AugmentedTrajectory 中。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    traj_id: str = Field(description="轨迹 ID；进入 SceneContext 后必须为 traj_a/traj_b/...")
    source: TrajectorySource
    waypoints: list[Waypoint] = Field(min_length=2, max_length=200,
                                       description="航点序列，按时间递增")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("waypoints")
    @classmethod
    def _validate_waypoints(cls, v: list[Waypoint]) -> list[Waypoint]:
        """校验 waypoints 长度在 [2, 200] 范围内。"""
        if not (2 <= len(v) <= 200):
            raise ValueError(f"waypoints length must be in [2, 200], got {len(v)}")
        return v


class PathWaypoint(BaseModel):
    """Layer 1 内部长路径素材点，不受 6 秒候选坐标范围约束。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    x: float
    y: float
    t: float = Field(ge=0.0)


class PathTrajectory(BaseModel):
    """仅供候选合成使用的真实空间路径，不进入 SceneContext。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    traj_id: str
    source: TrajectorySource
    waypoints: list[PathWaypoint] = Field(min_length=2, max_length=200)


class QAPair(BaseModel):
    """DriveLM 问答对（§4.6）。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    category: QACategory
    question: str
    answer: str


class EgoState(BaseModel):
    """自车状态快照（§4.7）。

    speed_mps 可由 ego_pose 相邻帧位移 / 时间差计算得到。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    speed_mps: float | None = Field(default=None, ge=0.0,
                                     description="自车速度（m/s），可由 ego_pose 帧间位移计算")


class SceneDescription(BaseModel):
    """场景语义描述（§4.8）。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    narrative: str = Field(default="", description="场景叙述文本，由 QA 答案拼接")
    keywords: list[str] = Field(default_factory=list, description="场景关键词列表")
    location_type: LocationType = Field(default=LocationType.UNKNOWN,
                                         description="场景位置类型")


# =============================================================================
# 轨迹特征与合成轨迹（Layer 1 内部 + benchmark）
# =============================================================================


class TrajectoryFeatures(BaseModel):
    """轨迹数值语义特征（§4.4）。

    由 TrajectoryAnalyzer 从坐标序列计算，封装速度/空间/关键区域数值特征 +
    自然语言描述。传给 Layer 3，替代原始坐标。

    natural_language_summary 严禁含合规性结论（违规/违反/合规/非法/应该/必须）。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    # 速度特征
    max_speed_mps: float = Field(ge=0.0, description="最大速度（m/s）")
    min_speed_mps: float = Field(ge=0.0, description="最小速度（m/s）")
    mean_speed_mps: float = Field(ge=0.0, description="平均速度（m/s）")

    # 空间特征
    total_distance_m: float = Field(ge=0.0, description="轨迹总行程（m）")
    max_lateral_offset_m: float = Field(ge=0.0, description="最大横向偏移（m）")
    n_waypoints: int = Field(ge=2, description="航点数量")

    # 冲突区几何与停车特征
    entered_conflict_zone: bool = False
    min_distance_to_conflict_m: float | None = Field(default=None, ge=0.0)
    full_stop: bool = False
    stop_duration_s: float = Field(default=0.0, ge=0.0)
    min_speed_in_zone_mps: float | None = Field(default=None, ge=0.0)
    stopped_before_zone: bool | None = False

    # v2：按适用规则分列的冲突区事实与动态智能体交互事实。
    conflict_zone_metrics: list[ConflictZoneMetrics] = Field(default_factory=list)
    agent_interactions: list[AgentInteraction] = Field(default_factory=list)

    # 关键区域行为描述
    conflict_zone_behavior: str = Field(
        default="",
        description="冲突区域附近的行为描述（信号灯场景对应停止线，让行场景对应冲突点）",
    )

    # 自然语言摘要（传入 Layer 3 LLM prompt）
    natural_language_summary: str = Field(
        default="",
        description="轨迹数值特征的纯自然语言描述，严禁含合规性结论词",
    )


class AugmentedTrajectory(BaseModel):
    """合成轨迹包装（§4.5）。

    包装合成轨迹与 benchmark 元数据，仅在 Layer 1 内部与 benchmark 脚本流转，
    不进入 SceneContext。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    trajectory: Trajectory
    features: TrajectoryFeatures | None = Field(default=None,
                                                 description="轨迹语义特征（可为空）")
    variant_type: TrajectoryVariantType
    expected_verdict: Literal["cleared", "vetoed", "exclude"] = Field(
        description="由最终几何特征推导的 benchmark 判定"
    )
    expected_verdict_reason: str = Field(default="", description="谓词推导或退化原因")
    exclude_reason: ExcludeReason | None = None
    predicate_version: str = Field(default="", description="打标谓词版本")
    feature_text_hash: str = Field(default="", description="LLM 可见舍入特征的哈希")
    is_degenerate: bool = Field(default=False, description="是否与已有候选特征碰撞")
    gt_precheck_status: Literal[
        "passed", "failed", "not_evaluable", "not_gt"
    ] = "not_gt"
    generation_note: str = Field(default="", description="合成说明（如扰动参数）")


# =============================================================================
# Layer 1 最终输出与内部中间类型
# =============================================================================


class ConflictZone(BaseModel):
    """由 Map Expansion 提供的停止线、横道等冲突区几何。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["stop_line", "ped_crossing", "yield_line", "unknown"]
    polygon_xy: list[tuple[float, float]] = Field(default_factory=list)
    distance_from_ego_m: float | None = Field(default=None, ge=0.0)
    source_layer: str


class ConflictZoneMetrics(BaseModel):
    """一条候选轨迹与某类适用冲突区的中性几何事实。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["stop_line", "ped_crossing"]
    governing: bool
    min_distance_m: float = Field(ge=0.0)
    entered: bool
    ended_inside: bool = False
    ended_inside_other_zone: bool = False
    entry_time_s: float | None = Field(default=None, ge=0.0)
    penetration_depth_m: float | None = Field(default=None, ge=0.0)
    min_speed_in_zone_mps: float | None = Field(default=None, ge=0.0)
    stopped_before_zone: bool | None = None


class AgentInteraction(BaseModel):
    """候选轨迹与一个相关交通参与者的时空关系。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    agent_kind: Literal["pedestrian", "oncoming_vehicle"]
    agent_track_id: str
    ego_zone_arrival_s: float | None = Field(default=None, ge=0.0)
    agent_zone_window_s: tuple[float, float] | None = None
    min_time_gap_s: float | None = None
    min_spatiotemporal_gap_m: float | None = Field(default=None, ge=0.0)
    crossing_order: Literal[
        "ego_first", "agent_first", "temporal_overlap", "no_shared_zone"
    ]


class SceneFacts(BaseModel):
    """结构化场景事实，供 Layer 2 条件匹配使用。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    has_red_light: bool = False
    has_yellow_light: bool = False
    traffic_light_status_source: Literal["drivelm_status", "none"] = "none"
    pedestrian_in_crosswalk: bool = False
    oncoming_vehicle_moving: bool = False
    emergency_vehicle_active: bool = False
    ego_on_minor_road: bool | None = None
    location_is_intersection: bool | None = None
    ego_displacement_m: float | None = Field(default=None, ge=0.0)
    conflict_zones: list[ConflictZone] = Field(default_factory=list)
    ego_turn_intent: Literal["left", "right", "straight", "unknown"] = "unknown"
    governing_stop_line_signed_m: float | None = None
    pedestrian_in_forward_crosswalk: bool = False
    pedestrian_moving: bool | None = None
    window_insufficient: bool = False


class SceneFactsDigest(BaseModel):
    """Layer 3 安全场景要件投影；不含坐标、标签或结论。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    signal_phase: Literal["red", "green", "unknown"]
    phase_source: Literal["drivelm_status", "none"]
    ego_turn_intent: Literal["left", "right", "straight", "unknown"]
    governing_stop_line_signed_m: float | None
    pedestrian_in_forward_crosswalk: bool
    pedestrian_moving: bool | None
    oncoming_vehicle_moving: bool
    location_is_intersection: bool | None


class CandidateBenchmarkLabel(BaseModel):
    """匿名候选轨迹到内部 benchmark 标签的映射。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    anonymous_id: str
    original_internal_id: str
    variant_type: TrajectoryVariantType
    expected_verdict: Literal["cleared", "vetoed", "exclude"]
    expected_verdict_reason: str = ""
    exclude_reason: ExcludeReason | None = None
    predicate_version: str = ""
    feature_text_hash: str = ""
    is_degenerate: bool = False
    difficulty: Literal["easy", "medium", "hard"]
    gt_precheck_status: Literal[
        "passed", "failed", "not_evaluable", "not_gt"
    ] = "not_gt"


class BenchmarkLabels(BaseModel):
    """只允许 benchmark/evaluation 读取的标签映射。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    scene_id: str
    frame_token: str
    data_version: str = "2026-07-11-v2"
    labels: list[CandidateBenchmarkLabel]


class SceneContext(BaseModel):
    """Layer 1 内部统一上下文。

    v3.1 规定 SceneContext 不直接传给 Layer 2/3；下游只能通过
    InterfaceAdapter 获得 SceneQuery 或 JudgeInput。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    scene_id: str
    frame_token: str = ""
    location: Literal["boston-seaport"] = "boston-seaport"
    description: SceneDescription = Field(default_factory=SceneDescription)
    ego_state: EgoState = Field(default_factory=EgoState, description="自车状态快照")
    scenario_type: ScenarioType = ScenarioType.UNKNOWN
    scene_facts: SceneFacts = Field(default_factory=SceneFacts)
    candidate_trajectories: list[Trajectory] = Field(
        default_factory=list,
        description="候选轨迹列表，进入 SceneContext 前必须匿名化并打乱顺序",
    )
    trajectory_features: list[TrajectoryFeatures] = Field(
        default_factory=list,
        description="与 candidate_trajectories 一一对应的语义特征",
    )

    @model_validator(mode="after")
    def _validate_candidates(self) -> Self:
        if any(not _is_anonymous_traj_id(traj.traj_id) for traj in self.candidate_trajectories):
            raise ValueError(
                "SceneContext candidate traj_id must be anonymized as traj_a/traj_b/..."
            )
        if self.trajectory_features and (
            len(self.trajectory_features) != len(self.candidate_trajectories)
        ):
            raise ValueError("trajectory_features must align with candidate_trajectories")
        return self


class SceneQuery(BaseModel):
    """Layer 2 窄接口：不暴露轨迹坐标或 benchmark 标签。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    scene_id: str
    frame_token: str
    scenario_type: ScenarioType
    keywords: list[str] = Field(default_factory=list)
    scene_facts: SceneFacts = Field(default_factory=SceneFacts)


class JudgeInput(BaseModel):
    """Layer 3 窄接口：暴露安全场景要件与轨迹事实摘要。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    scene_id: str
    frame_token: str
    narrative: str
    scene_facts_digest: SceneFactsDigest
    trajectory_features: list[TrajectoryFeatures] = Field(default_factory=list)


class ParsedSceneBundle(BaseModel):
    """Layer 1 对编排层交付的完整包。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    context: SceneContext
    scene_query: SceneQuery
    judge_input: JudgeInput
    benchmark_labels: BenchmarkLabels


class RawScene(BaseModel):
    """解析前的原始数据（§4.10），内部中间类型，不对外暴露。

    dataset_loader 负责一切 nuScenes I/O，把 ego_pose 序列预读进 RawScene，
    从而保证 TrajectoryExtractor 是纯函数、可独立单测。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_id: str
    frame_token: str
    source_file: Path
    raw_json: dict[str, Any] = Field(
        default_factory=dict,
        description="当前帧的 DriveLM QA + key_object_infos",
    )
    location: Literal["boston-seaport"] = "boston-seaport"
    ego_poses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本场景 ego_pose 序列（全局），"
                    "每项含 translation(x,y,z)、rotation(四元数)、timestamp",
    )
    path_ego_poses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="候选合成专用的长窗口 ego_pose；为空时回退到 ego_poses",
    )
    map_records: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Map Expansion 查询结果，按 layer name 分组",
    )
    annotations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="当前 keyframe 的 nuScenes 3D 标注及属性摘要",
    )
    keyframe_index: int = Field(
        default=0,
        ge=0,
        description="当前关键帧在 ego_poses 中的下标",
    )


def _is_anonymous_traj_id(traj_id: str) -> bool:
    suffix = traj_id.removeprefix("traj_")
    return suffix != traj_id and suffix.isalpha() and suffix.islower()
