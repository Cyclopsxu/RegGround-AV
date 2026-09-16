# Layer 1 数据解析层 · 功能与接口设计文档

**RegGround-AV · 法规接地的可解释轨迹偏好打标系统**

*Software Design Specification — Data Parsing Layer*

| 项目 | 内容 |
|---|---|
| 文档版本 | v3.2（修订） |
| 前序版本 | v3.0（2026-06）、v2.0（2026-04-30）、v1.0（2026-04-17） |
| 文档状态 | v3.2 Gate 3 候选版（复冒烟通过前不冻结） |
| 日期 | 2026-07-17 |
| 修订依据 | `docs/RegGround-AV_问题分析与修订清单.md` A1-A6、B7-B12、C13、F28-F29 |

---

## 1. 修订目标

Layer 1 v3.2 保留 v3.1 的接口边界，并修正候选合成的物理合理性：候选只改变速度/时机，不改变人类真实空间路径。

v3.1 的设计目标是把这些风险转化为明确接口约束：

- 数据范围锁定为 Boston-only，处理单元锁定为 keyframe。
- 使用 nuScenes Map Expansion 提供停止线、人行横道与路口几何。
- 使用 LIDAR_TOP `sample_data` 链分别读取 6 秒轨迹窗口与 12 秒路径素材窗口。
- 所有合成变体沿同一条弧长参数化真实路径重采样，只修改速度剖面。
- 对速度施加曲率横向加速度上限，并在出口校验 `drivable_area`。
- 在进入 `SceneContext` 前匿名化候选轨迹 ID 并随机打乱顺序。
- 对 narrative 与最终 prompt 输入做合规性结论词审计。
- 向下游输出最小可见接口：Layer 2 使用 `SceneQuery`，Layer 3 使用 `JudgeInput`。

---

## 2. 层级职责与边界

### 2.1 职责声明

Layer 1 负责把 nuScenes、DriveLM-nuScenes 与 Map Expansion 数据融合为可审计的场景表达，并生成匿名化候选轨迹及其结构化特征。本层完成从原始数据到下游最小接口的转换，不承担法规判定或偏好排序。

### 2.2 In Scope

- 过滤 nuScenes scene，仅保留 `log.location == "boston-seaport"`。
- 以 DriveLM keyframe，即 nuScenes `sample_token`，作为处理与评估基本单元。
- 从 DriveLM QA 与 key object status 提取场景叙事、关键词、信号灯相位。
- 从 nuScenes 3D 标注和 Map Expansion 提取结构化事实 `SceneFacts`。
- 通过 LIDAR_TOP `sample_data` 链读取 6 秒 GT 与默认 12 秒路径素材。
- 从 Map Expansion 的 `stop_line` 与 `ped_crossing` 计算冲突区几何。
- 基于真实轨迹合成候选轨迹，并在进入下游前匿名化 ID 和随机打乱顺序。
- 生成 Layer 2 的 `SceneQuery` 与 Layer 3 的 `JudgeInput`。
- 输出 benchmark 侧标签映射，但禁止其进入 Layer 2/3 调用路径。

### 2.3 Out of Scope

- LLM 调用、法规图谱检索、偏好排序和最终裁判。
- 红绿灯视觉感知。信号灯相位只消费 DriveLM `key_object_infos.*.Status`。
- 交通法规内容校验。
- 数据集下载。
- 新加坡场景、左行道路语义、BDD-X 等其他数据源。

### 2.4 关键假设声明

本项目使用中国道路交通法规解释 Boston 场景，是一个实验简化假设。Boston 与中国道路同属右行语义，因此“左转让对向直行”“右侧通行”等几何关系可保持一致。论文和实验报告必须显式声明该假设，并说明新加坡左行场景已全部剔除。

---

## 3. 模块架构

```
src/layer1/
    __init__.py
    models.py
    exceptions.py
    dataset_loader.py
    scene_text_extractor.py
    scene_fact_extractor.py
    trajectory_extractor.py
    trajectory_augmentor.py
    trajectory_analyzer.py
    candidate_anonymizer.py
    leakage_guard.py
    interface_adapter.py
    availability_spike.py
```

| 模块 | 职责 |
|---|---|
| `dataset_loader.py` | 读 DriveLM QA、nuScenes 表、Map Expansion；构建 Boston scene 白名单 |
| `scene_text_extractor.py` | 提取 narrative、keywords、traffic light status，并执行文本脱敏 |
| `scene_fact_extractor.py` | 用 3D 标注和地图几何生成 `SceneFacts` |
| `trajectory_extractor.py` | 从密集 ego pose 重建真实轨迹 |
| `trajectory_augmentor.py` | 生成 GT、conservative、aggressive、suboptimal、illegal、hard variants |
| `trajectory_analyzer.py` | 计算 `TrajectoryFeatures` 与自然语言摘要 |
| `candidate_anonymizer.py` | 将候选轨迹改写为 `traj_a`、`traj_b` 等匿名 ID 并打乱顺序 |
| `leakage_guard.py` | 扫描 narrative、feature summary 与 prompt 片段，防止标签或结论词泄露 |
| `interface_adapter.py` | 从 `ParsedSceneBundle` 生成 `SceneQuery` 与 `JudgeInput` |
| `availability_spike.py` | W1 数据可得性统计脚本 |

---

## 4. 数据依赖

### 4.1 Boston-only 场景筛选

筛选链路：

```
scene.json -> log_token -> log.json.location
```

`DriveLMDatasetLoader.__init__` 必须在加载 DriveLM keyframes 前构建 Boston scene token 白名单，并将 DriveLM 顶层 scene token 与白名单做交集过滤。

保留条件：

```python
log.location == "boston-seaport"
```

处理和评估基本单元为 keyframe，即 DriveLM `key_frames` 下的 `sample_token`。`scene_id` 仅用于分组归属。

### 4.2 密集自车轨迹重建

轨迹重建使用 LIDAR_TOP 通道的 `sample_data` 链，而不是 2Hz keyframe 链。

| nuScenes 表 | 必用字段 | 用途 |
|---|---|---|
| `sample.json` | `token`, `scene_token`, `timestamp` | 定位 DriveLM keyframe |
| `sample_data.json` | `sample_token`, `ego_pose_token`, `calibrated_sensor_token`, `timestamp`, `next` | 沿 LIDAR_TOP 全帧链读取未来窗口 |
| `calibrated_sensor.json` | `sensor_token` | 连接 sensor |
| `sensor.json` | `channel` | 过滤 `LIDAR_TOP` |
| `ego_pose.json` | `translation`, `rotation`, `timestamp` | 重建局部坐标轨迹与速度 |

默认窗口：

- `future_horizon_s = 6.0`
- `path_horizon_s = 12.0`
- `target_channel = "LIDAR_TOP"`
- 6 秒 GT 期望 waypoint 数约 120；12 秒路径读取后均匀降采样到最多 200 点。
- 路径源耗尽时不外推，候选到达路径末端即提前截断。

### 4.3 地图冲突区几何

新增依赖：nuScenes Map Expansion v1.3。Boston-only 下只需 `maps/expansion/boston-seaport.json`。

| 地图层 | 用途 |
|---|---|
| `stop_line` | 信号灯停止线、停车让行线，计算 `min_distance_to_conflict_m` |
| `ped_crossing` | 行人让行冲突区，判断行人是否在横道内 |
| `traffic_light` | 辅助确认信号灯存在，不提供相位 |
| `road_segment` | 判断是否在路口，辅助 `LocationType.INTERSECTION` |
| `lane` | 辅助支路/主路与车道归属判断 |
| `drivable_area` | 候选出口逐 waypoint 可行驶区域包含校验 |

### 4.4 让行结构化事实

让行类场景不得只靠文本关键词。主路径使用 nuScenes 3D 标注与地图几何：

| 条件 | 主路径 |
|---|---|
| 行人在横道内 | `human.pedestrian.*` annotation 中心落入最近 `ped_crossing` 多边形，且 attribute 包含 `pedestrian.moving` |
| 对向直行车 | `vehicle.*` 与 ego 朝向夹角大于 150 度，attribute 包含 `vehicle.moving` |
| 执行任务特种车辆 | `vehicle.emergency.*` 或已配置 emergency 类别，且 moving；若 Boston 样本不足，W1 后可裁剪该细分 |
| 支路汇入主路 | ego 所属 lane/road_segment 与主路 lane 拓扑关系匹配；不足时只作为候选事实，不强行触发 |

---

## 5. 数据模型

### 5.1 枚举

```python
class ScenarioType(str, Enum):
    RED_LIGHT = "red_light"
    YELLOW_LIGHT = "yellow_light"
    PEDESTRIAN = "pedestrian"
    ONCOMING = "oncoming"
    MINOR_ROAD = "minor_road"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"

class TrajectoryVariantType(str, Enum):
    GROUND_TRUTH = "ground_truth"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ILLEGAL = "illegal"
    SUBOPTIMAL = "suboptimal"
    HARD_CASE = "hard_case"
```

`TrajectoryVariantType` 只允许存在于 benchmark 侧模型中，严禁出现在 `SceneContext`、`SceneQuery`、`JudgeInput` 或任何 LLM prompt 中。

### 5.2 Waypoint 与 Trajectory

```python
class Waypoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    t: float

class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    traj_id: str                 # 进入 SceneContext 后必须是 traj_a/traj_b/...
    source: TrajectorySource
    waypoints: list[Waypoint]    # 长度 2-200
    confidence: float | None = None
```

### 5.3 SceneFacts

```python
class ConflictZone(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["stop_line", "ped_crossing", "yield_line", "unknown"]
    polygon_xy: list[tuple[float, float]]
    distance_from_ego_m: float | None
    source_layer: str

class SceneFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    has_red_light: bool = False
    has_yellow_light: bool = False
    traffic_light_status_source: Literal["drivelm_status", "none"] = "none"
    pedestrian_in_crosswalk: bool = False
    oncoming_vehicle_moving: bool = False
    emergency_vehicle_active: bool = False
    ego_on_minor_road: bool | None = None
    location_is_intersection: bool | None = None
    conflict_zones: list[ConflictZone] = Field(default_factory=list)
```

### 5.4 SceneContext

`SceneContext` 是 Layer 1 内部统一上下文，允许包含候选轨迹坐标，但不得直接传给 Layer 2/3。

```python
class SceneContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    frame_token: str
    location: Literal["boston-seaport"]
    description: SceneDescription
    ego_state: EgoState
    scenario_type: ScenarioType
    scene_facts: SceneFacts
    candidate_trajectories: list[Trajectory]       # 匿名 ID，顺序已打乱
    trajectory_features: list[TrajectoryFeatures]  # 与 candidate_trajectories 一一对应
```

### 5.5 BenchmarkLabels

```python
class CandidateBenchmarkLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    anonymous_id: str
    original_internal_id: str
    variant_type: TrajectoryVariantType
    expected_verdict: Literal["cleared", "vetoed", "exclude"]
    difficulty: Literal["easy", "medium", "hard"]
    gt_precheck_status: Literal["passed", "failed", "not_gt"]

class BenchmarkLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    labels: list[CandidateBenchmarkLabel]
```

该模型只由评估脚本读取，不允许传入 `SceneQuery`、`JudgeInput` 或 Layer 3。

### 5.6 下游窄接口

Layer 2 入参：

```python
class SceneQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    frame_token: str
    scenario_type: ScenarioType
    keywords: list[str]
    scene_facts: SceneFacts
```

Layer 3 入参：

```python
class JudgeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    frame_token: str
    narrative: str
    trajectory_features: list[TrajectoryFeatures]
```

`JudgeInput` 不包含 waypoints、variant labels、expected verdict 或 benchmark mapping，使 LLM 物理上无法读取坐标和标签。

### 5.7 ParsedSceneBundle

```python
class ParsedSceneBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: SceneContext
    scene_query: SceneQuery
    judge_input: JudgeInput
    benchmark_labels: BenchmarkLabels
```

---

## 6. 核心接口规范

### 6.1 DriveLMDatasetLoader

```python
class DriveLMDatasetLoader:
    def __init__(
        self,
        nuscenes_root: Path,
        drivelm_qa_path: Path,
        *,
        map_name: Literal["boston-seaport"] = "boston-seaport",
        nuscenes_version: str = "v1.0-trainval",
        future_horizon_s: float = 6.0,
        path_horizon_s: float = 12.0,
    ) -> None: ...

    def list_keyframes(self) -> list[tuple[str, str]]: ...

    def load_keyframe(self, scene_token: str, frame_token: str) -> RawScene: ...

    def iter_keyframes(self) -> Iterator[RawScene]: ...
```

加载器初始化时必须：

1. 构建 Boston scene token 白名单。
2. 加载 DriveLM QA，并只保留白名单内 keyframes。
3. 初始化 `NuScenesMap(map_name="boston-seaport")`。
4. 预建 LIDAR_TOP sample_data 索引，支持从 keyframe 向后读取密集 ego pose。

### 6.2 SceneTextExtractor

```python
class SceneTextExtractor:
    def extract(self, raw_scene: RawScene) -> SceneDescription: ...
    def infer_scenario_type(self, raw_scene: RawScene, facts: SceneFacts) -> ScenarioType: ...
```

关键词表必须包含英文，中文只作为补充：

```python
SCENARIO_KEYWORD_MAP = {
    ScenarioType.RED_LIGHT: ["red light", "traffic light", "stop line", "red signal", "红灯"],
    ScenarioType.YELLOW_LIGHT: ["yellow light", "amber light", "yellow signal", "黄灯"],
    ScenarioType.PEDESTRIAN: ["pedestrian", "crosswalk", "crossing", "zebra crossing", "行人"],
    ScenarioType.ONCOMING: ["oncoming", "opposite direction", "left turn", "yield", "对向"],
    ScenarioType.MINOR_ROAD: ["minor road", "side road", "merge", "main road", "支路"],
    ScenarioType.EMERGENCY: ["ambulance", "police car", "fire truck", "emergency vehicle", "siren", "救护车"],
}
```

`narrative` 采用白名单构建：只读取 DriveLM 场景描述、perception 类 QA 和 `SceneFacts` 的中性转述；planning、behavior、prediction 类 QA 不进入 narrative。输出不超过 600 字符，按完整片段取舍，并在进入下游前经过 `LeakageGuard` 检查。违禁词至少包含：

- 标签词：`illegal`, `ground truth`, `vetoed`, `cleared`, `variant`
- 合规结论词：`violate`, `violation`, `compliant`, `non-compliant`, `should stop`, `must yield`
- 中文补充：`违规`, `违反`, `合规`, `非法`, `应该`, `必须`

### 6.3 SceneFactExtractor

```python
class SceneFactExtractor:
    def extract(self, raw_scene: RawScene) -> SceneFacts: ...
```

事实提取顺序：

1. 从 Map Expansion 查询 ego 50m 半径内 `stop_line`、`ped_crossing`、`traffic_light`。
2. 从 DriveLM `key_object_infos.*.Status` 读取信号灯相位。
3. 从 `sample_annotation`、`category`、`attribute` 读取行人、车辆、特种车辆。
4. 使用地图多边形包含关系和朝向夹角生成结构化布尔事实。
5. 文本关键词仅作为事实不可得时的兜底证据，不覆盖结构化主路径。

### 6.4 TrajectoryExtractor

```python
class TrajectoryExtractor:
    def __init__(
        self,
        future_horizon_s: float = 6.0,
        path_horizon_s: float = 12.0,
        min_waypoints: int = 2,
    ) -> None: ...
    def extract(self, raw_scene: RawScene) -> Trajectory: ...
    def extract_path(self, raw_scene: RawScene) -> PathTrajectory: ...
    def local_drivable_polygons(self, raw_scene: RawScene) -> list[list[tuple[float, float]]]: ...
```

重建逻辑：

- 以 keyframe ego pose 为局部坐标原点。
- 局部 x 为车头方向，局部 y 为左侧方向。
- `t` 为相对 keyframe 的秒数。
- `extract()` 只返回 6 秒 GT；`extract_path()` 返回默认 12 秒真实路径素材。
- 路径不足 12 秒时接受数据源提前耗尽，不做直线或曲线外推。
- 任一输出少于 `min_waypoints` 时抛 `TrajectoryExtractionError`。

### 6.5 TrajectoryAugmentor

```python
class TrajectoryAugmentor:
    def __init__(
        self,
        seed: int = 42,
        *,
        a_lat_max_mps2: float = 3.5,
        pedestrian_illegal_gap_target_s: float = -0.3,
    ) -> None: ...
    def augment(
        self,
        ground_truth: Trajectory,
        scenario_type: ScenarioType,
        scene_facts: SceneFacts,
        *,
        scene_id: str,
        frame_token: str,
        path_trajectory: PathTrajectory | Trajectory | None = None,
        drivable_area_polygons: list[list[tuple[float, float]]] | None = None,
    ) -> list[AugmentedTrajectory]: ...
```

约束：

- GT 路径先按弧长 `s` 参数化，所有合成变体只改变 `v(s)` 并在同一路径重采样；禁止平移、缩放或扰动 x/y 几何。
- conservative 通常使用 0.60x 速度；红灯且可定位本向停止线时，在同一路径上于停止线前 1.0 m 停车并保持至窗口结束。suboptimal 保持轻微波动速度剖面；aggressive 使用 1.30x 速度；hard_case 使用中段 creeping 速度剖面。
- red/yellow/pedestrian/oncoming 等 illegal 语义通过移除低速段或提高同一路径速度实现，不得构造新几何。
- pedestrian illegal 的目标时间间隙为 `-0.3s`;未达标时只增强同路径速度剖面,物理约束下仍无法达到则记 `no_violation_injectable`。
- 任一点速度满足 `v(s) <= sqrt(a_lat_max / kappa(s))`，默认 `a_lat_max = 3.5 m/s²`。
- 每条候选出口必须逐 waypoint 落在 Map Expansion `drivable_area` 内；有地图覆盖但出界时抛 `TrajectoryAugmentationError`，该场景不得静默进入 benchmark。
- 长路径素材耗尽时候选提前截断，绝不外推路径。
- hard 变体用于 E21/E22 难度梯度，如 creeping、临界停车、停后提前起步。
- 违规变体必须由同源结构化 fact 支撑；无支撑、与既有候选特征退化或达不到交互目标时标记 `no_violation_injectable`，不得缩放 x/y 制造差异。
- GT 轨迹进入 benchmark 前必须做快速筛查。明确违规的 GT 轨迹从 benchmark 排除，或标记 `expected_verdict="vetoed"` 并记录原因。

### 6.6 CandidateAnonymizer

```python
class CandidateAnonymizer:
    def __init__(self, seed: int = 42) -> None: ...

    def anonymize(
        self,
        candidates: list[AugmentedTrajectory],
    ) -> tuple[list[Trajectory], BenchmarkLabels]: ...
```

行为：

- 候选顺序使用 `Random(f"{seed}:{frame_token}")` 逐 keyframe 随机打乱。
- 下游可见 ID 必须为 `traj_a`、`traj_b`、`traj_c` 等。
- `BenchmarkLabels` 保存匿名 ID 到内部 variant 的映射，但不进入 `SceneContext`。
- 对 Layer 1 产出的 narrative 与 feature summary 做运行时扫描，确保 `illegal`、`ground_truth`、`vetoed` 等词不存在。

### 6.7 InterfaceAdapter

```python
class InterfaceAdapter:
    def to_scene_query(self, context: SceneContext) -> SceneQuery: ...
    def to_judge_input(self, context: SceneContext) -> JudgeInput: ...
```

`to_judge_input()` 必须重新运行 `LeakageGuard.assert_safe_prompt_fragment()`，扫描 narrative 和每条 `TrajectoryFeatures.natural_language_summary`。

### 6.8 Facade

```python
def parse_scene(raw_scene: RawScene, *, seed: int = 42) -> ParsedSceneBundle: ...

def parse_dataset(
    nuscenes_root: Path,
    *,
    drivelm_qa_path: Path,
    future_horizon_s: float = 6.0,
    path_horizon_s: float = 12.0,
    on_scene_error: Callable[[Exception, RawScene], None] | None = None,
) -> Iterator[ParsedSceneBundle]: ...
```

---

## 7. W1 数据可得性 Spike

必须在正式开发前运行：

```python
def collect_availability_stats(
    nuscenes_root: Path,
    drivelm_qa_path: Path,
) -> AvailabilityStats: ...
```

统计项：

| 指标 | 决策用途 |
|---|---|
| Boston ∩ DriveLM keyframe 总数 | 样本规模 |
| 含有效 traffic-element Status 的 keyframe 数与比例 | 决定是否保留信号灯场景 |
| red / yellow / green 数量 | 决定红灯、黄灯细分可行性 |
| 行人在 ped_crossing 内的 keyframe 数 | 决定 PEDESTRIAN 细分样本量 |
| `vehicle.emergency.*` keyframe 数 | 决定 EMERGENCY 是否保留 |
| 有 stop_line / ped_crossing 地图几何的比例 | 验证冲突区几何可得性 |

若某细分有效样本少于 20，默认从 MVP 范围中删除，避免论文实验建立在极小样本上。

---

## 8. 非功能需求

| 指标 | 目标 |
|---|---|
| 单 keyframe 解析 P95 | ≤ 400 ms |
| 密集轨迹重建 P95 | ≤ 80 ms |
| 地图与结构化事实提取 P95 | ≤ 150 ms |
| 候选生成与匿名化 P95 | ≤ 80 ms |
| 泄露扫描 | 100% 覆盖 Layer 1 narrative 与 feature summary |

由于窗口从 3 秒稀疏 keyframes 改为 6 秒密集 ego pose，v3.1 性能预算从 v3.0 的 250ms 放宽到 400ms。

---

## 9. 测试要求

| 测试文件 | 必须覆盖 |
|---|---|
| `test_dataset_loader.py` | Boston-only 过滤；6 秒 GT 与 12 秒路径双窗口；`drivable_area` 读取 |
| `test_scene_fact_extractor.py` | stop_line / ped_crossing 查询；行人在横道内；对向车夹角；emergency 稀疏场景 |
| `test_scene_text_extractor.py` | 英文关键词命中；Status 红/黄/绿解析；narrative 脱敏 |
| `test_trajectory_extractor.py` | 6/12 秒双窗口；路径耗尽截断；局部坐标与 drivable 多边形变换 |
| `test_trajectory_augmentor.py` | 全变体路径重合；横向加速度上限；出界拒绝；路径耗尽截断 |
| `test_candidate_anonymizer.py` | ID 为 `traj_*`；逐 keyframe 可复现随机；mapping 只在 benchmark labels 中 |
| `test_leakage_guard.py` | Layer 1 narrative 与 feature summary 扫描禁词 |
| `test_interface_adapter.py` | `SceneQuery` 无轨迹坐标；`JudgeInput` 无 waypoints 与 labels |
| `test_integration.py` | `parse_scene(RawScene(...))` 端到端输出 `ParsedSceneBundle` |

隔离专项测试示例：

```python
def test_no_variant_token_in_full_prompt_fragment(raw_scene):
    bundle = parse_scene(raw_scene)
    prompt_fragment = render_test_prompt(bundle.judge_input)
    banned = ["illegal", "ground_truth", "vetoed", "variant", "违规", "合规"]
    assert all(token not in prompt_fragment.lower() for token in banned)
```

---

## 10. 对接说明

Layer 1 对编排层交付 `ParsedSceneBundle`：

```python
for bundle in parse_dataset(nuscenes_root, drivelm_qa_path=qa_path):
    subgraph = retriever.retrieve(bundle.scene_query)
    label = judge.judge(bundle.judge_input, subgraph)
    evaluate(label, bundle.benchmark_labels)
```

只有评估函数可以读取 `benchmark_labels`。Layer 2 与 Layer 3 只接收窄接口。

---

## 11. 验收 Checklist

- [ ] Boston-only 过滤在 loader 初始化阶段完成。
- [ ] keyframe 是处理和评估基本单元。
- [ ] Map Expansion `boston-seaport.json` 是显式依赖。
- [ ] 轨迹窗口为 6 秒，路径素材窗口默认为 12 秒，路径耗尽不外推。
- [ ] `min_distance_to_conflict_m` 有停止线/横道几何来源。
- [ ] 英文关键词表覆盖 DriveLM QA。
- [ ] 所有合成变体与 GT 空间路径重合，只改变速度/时机。
- [ ] 合成候选满足曲率横向加速度上限。
- [ ] 有地图覆盖的候选通过 `drivable_area` 出口校验；正式运行报告通过数/校验数，只有实测全通过时才写“100%”。
- [ ] 进入下游的候选 ID 全部匿名化，且顺序逐 keyframe 可复现。
- [ ] narrative 与 feature summary 通过运行时泄露扫描。
- [ ] Layer 2/3 只接收 `SceneQuery` / `JudgeInput`。
- [ ] W1 availability spike 可输出样本统计报告。

### 11.1 论文与答辩冻结口径

- 方法叙事：候选多样性刻意集中在法规规制的速度与时机维度；空间几何固定为人类真实驾驶路径，作为控制变量隔离法规行为差异。
- Limitation：不同转弯线形（例如转弯半径大小）的几何候选合成留作后续工作。
- W6 联调：留意收集一个 GT 被 vetoed、保守变体胜出的完整案例，保存轨迹图、特征、规则命中与最终排序，作为答辩材料。

---

*文档结束，Layer 1 v3.2 修订*
