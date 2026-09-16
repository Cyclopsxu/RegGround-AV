# Layer 2 图谱检索层 · 功能与接口设计文档

**RegGround-AV · 法规接地的可解释轨迹偏好打标系统**

*Software Design Specification — Graph Retrieval Layer*

| 项目 | 内容 |
|---|---|
| 文档版本 | v3.0 |
| 前序版本 | v2.1（2026-06）、v2.0（2026-06）、v1.0（2026-04-17） |
| 文档状态 | 接口修订版，待开发实施 |
| 日期 | 2026-06 |
| 修订依据 | `docs/RegGround-AV_问题分析与修订清单.md` C13-C15、D16、F28 |

---

## 1. 修订目标

Layer 2 v2.1 已经从 Neo4j + LLM Cypher 改为 NetworkX 确定性检索，但仍有三类问题：

- 条件匹配仍依赖 DriveLM QA 关键词，面对英文文本、弱描述和场景歧义时不稳。
- 图谱只有 Condition 到 Rule 的一跳遍历，难以说明 Graph 相比 dict 的价值。
- 输入直接依赖完整 `SceneContext`，导致 Layer 2 能看见大量不需要的信息。

v3.0 修订为：

- 对外只接收 Layer 1 的 `SceneQuery` 窄接口。
- Condition 匹配主路径使用 `SceneFacts` 结构化事实，关键词仅作兜底。
- 图 schema 增加 `OVERRIDES` / `EXCEPTION_WHEN` 关系，支持冲突规则消解。
- RuleNode 构造不再 mutate frozen 模型。
- 将“零幻觉”表述限定为 citation validity 的构造性保障，不再声称 citation accuracy 已天然解决。

---

## 2. 层级职责与边界

### 2.1 职责声明

Layer 2 的唯一职责是把 `SceneQuery` 映射为可追溯的 `RuleSubgraph`。本层不判断轨迹是否违规，也不参与偏好排序。它只决定“在这个场景事实下，哪些法规可以被 Layer 3 引用”。

### 2.2 In Scope

- 加载并校验本地 YAML 法规图。
- 使用结构化事实命中 Condition。
- 在结构化事实不足时使用英文关键词兜底。
- 沿图遍历召回 TrafficRule、RoadActor、Consequence。
- 处理 `OVERRIDES` / `EXCEPTION_WHEN` 规则冲突。
- 输出有界、可反查的 rule_id 集合。
- 提供 citation validity 的构造性保障。

### 2.3 Out of Scope

- 读取完整 `SceneContext`、候选轨迹坐标或 benchmark labels。
- 判断某条候选轨迹是否违反规则。
- 调用 LLM、向量检索、数据库或网络服务。
- 校验中国法规在 Boston 场景中的法律效力。该简化假设由系统与论文统一声明。

---

## 3. 模块架构

```
src/layer2/
    __init__.py
    models.py
    exceptions.py
    rule_graph.py
    condition_matcher.py
    conflict_resolver.py
    retriever.py

data/layer2/
    rule_graph.yaml
```

| 模块 | 职责 |
|---|---|
| `models.py` | `SceneQuery` 依赖类型、`RuleNode`、`RuleSubgraph`、override 数据模型 |
| `rule_graph.py` | 加载 YAML，构建 NetworkX 图，提供遍历查询 |
| `condition_matcher.py` | `SceneFacts` / keywords 到 Condition 的确定性匹配 |
| `conflict_resolver.py` | 处理 `OVERRIDES` / `EXCEPTION_WHEN`，标记被覆盖规则 |
| `retriever.py` | Facade，编排匹配、遍历、冲突消解与 telemetry |

---

## 4. 对外接口与数据模型

### 4.1 SceneQuery

Layer 2 不再接收完整 `SceneContext`，只接收：

```python
class SceneQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    frame_token: str
    scenario_type: ScenarioType
    keywords: list[str]
    scene_facts: SceneFacts
```

约束：

- 不含候选轨迹坐标。
- 不含 `trajectory_features`。
- 不含 `variant_type`、`expected_verdict` 或匿名映射。
- `keywords` 已由 Layer 1 做英文标准化和泄露清洗。

### 4.2 枚举

```python
class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"

class RuleCategory(str, Enum):
    SIGNAL = "signal"
    YIELD = "yield"

class RetrievalMode(str, Enum):
    STRUCTURED = "structured"
    KEYWORD_FALLBACK = "keyword_fallback"
    MIXED = "mixed"

class RuleStatus(str, Enum):
    ACTIVE = "active"
    OVERRIDDEN = "overridden"
```

### 4.3 RuleOverride

```python
class RuleOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    overriding_rule_id: str
    overridden_rule_id: str
    condition_id: str
    reason: str
```

### 4.4 RuleNode

```python
class RuleNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    code: str
    description: str
    severity: Severity
    category: RuleCategory
    actors: list[RoadActor] = Field(default_factory=list)
    consequences: list[Consequence] = Field(default_factory=list)
    matched_conditions: list[str] = Field(default_factory=list)
    retrieved_via: RetrievalMode
    status: RuleStatus = RuleStatus.ACTIVE
    overridden_by: list[str] = Field(default_factory=list)
```

`RuleNode` 为 frozen 模型，`matched_conditions` 必须在构造前聚合完成，不允许先建对象再 append。

### 4.5 RuleSubgraph

```python
class RuleSubgraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    rules: list[RuleNode]
    matched_conditions: list[str]
    retrieval_mode: RetrievalMode
    overrides_applied: list[RuleOverride] = Field(default_factory=list)
    query_duration_ms: int
    graph_version: str
    retrieval_timestamp: datetime

    @computed_field
    @property
    def active_rules(self) -> list[RuleNode]:
        return [r for r in self.rules if r.status == RuleStatus.ACTIVE]

    @computed_field
    @property
    def hard_rules(self) -> list[RuleNode]:
        return [r for r in self.active_rules if r.severity == Severity.HARD]

    @computed_field
    @property
    def soft_rules(self) -> list[RuleNode]:
        return [r for r in self.active_rules if r.severity == Severity.SOFT]
```

Layer 3 默认只使用 `active_rules`。`OVERRIDDEN` 规则保留在 `rules` 中用于审计，但不参与一票否决。

---

## 5. 图谱 Schema

### 5.1 节点类型

| 节点类型 | 关键属性 | 说明 |
|---|---|---|
| `TrafficRule` | `id`, `code`, `description`, `severity`, `category` | 法规条款 |
| `Condition` | `id`, `description`, `keywords`, `fact_keys` | 触发条件 |
| `Behavior` | `id`, `description` | 受规制行为 |
| `RoadActor` | `id`, `type`, `description`, `priority_level` | 道路参与者 |
| `Consequence` | `id`, `type`, `description`, `severity_score` | 违规后果 |

### 5.2 关系类型

| 关系 | 起始 -> 终止 | 语义 |
|---|---|---|
| `APPLIES_WHEN` | TrafficRule -> Condition | 条件成立时规则适用 |
| `PROHIBITS` | TrafficRule -> Behavior | 禁止行为 |
| `REQUIRES` | TrafficRule -> Behavior | 要求行为 |
| `APPLIES_TO` | TrafficRule -> RoadActor | 适用对象 |
| `LEADS_TO_CONSEQUENCE` | TrafficRule -> Consequence | 后果 |
| `CO_OCCURS` | Condition -> Condition | 条件共现，当前仅审计不自动扩召 |
| `OVERRIDES` | TrafficRule -> TrafficRule | 前者在给定条件下覆盖后者 |
| `EXCEPTION_WHEN` | TrafficRule -> Condition | 条件成立时该规则被例外处理 |

### 5.3 YAML 示例

```yaml
graph_version: "2026-06-v3.0"

conditions:
  - id: C-RED-PHASE
    description: 当前方向为红灯相位
    keywords: ["red light", "traffic light", "red signal", "stop line"]
    fact_keys: ["has_red_light"]

  - id: C-YELLOW-PHASE
    description: 当前方向为黄灯相位
    keywords: ["yellow light", "amber light", "yellow signal"]
    fact_keys: ["has_yellow_light"]

  - id: C-PED-IN-CROSSWALK
    description: 人行横道内有正在通过的行人
    keywords: ["pedestrian", "crosswalk", "crossing", "zebra crossing"]
    fact_keys: ["pedestrian_in_crosswalk"]

  - id: C-ONCOMING-STRAIGHT
    description: 本车转向路径与对向移动车辆冲突
    keywords: ["oncoming", "opposite direction", "left turn", "yield"]
    fact_keys: ["oncoming_vehicle_moving"]

  - id: C-ON-MINOR-ROAD
    description: 本车由支路或让行路段汇入主路
    keywords: ["minor road", "side road", "merge", "main road"]
    fact_keys: ["ego_on_minor_road"]

  - id: C-EMERGENCY-ACTIVE
    description: 执行任务的特种车辆正在通行
    keywords: ["ambulance", "police car", "fire truck", "emergency vehicle", "siren"]
    fact_keys: ["emergency_vehicle_active"]

rules:
  - id: R-SIG-01
    code: "RoadTrafficSafetyLaw-38-red"
    description: 红灯相位下禁止越过停止线继续行驶
    severity: hard
    category: signal
    applies_when: [C-RED-PHASE]
    prohibits: [B-CROSS-STOPLINE]
    applies_to: [A-VEHICLE]
    consequences: [K-SIGNAL]

  - id: R-YLD-04
    code: "RoadTrafficSafetyLaw-53-emergency"
    description: 遇执行任务的特种车辆应当让行
    severity: hard
    category: yield
    applies_when: [C-EMERGENCY-ACTIVE]
    requires: [B-YIELD-EMERGENCY]
    applies_to: [A-VEHICLE, A-EMERGENCY]
    consequences: [K-YIELD]
    overrides:
      - rule_id: R-SIG-01
        condition: C-EMERGENCY-ACTIVE
        description: 执行任务特种车辆让行优先级高于原地等待红灯的普通约束
```

### 5.4 MVP 法规集合

| Rule ID | 类别 | severity | 条件 |
|---|---|---|---|
| R-SIG-01 | signal | hard | C-RED-PHASE |
| R-SIG-02 | signal | hard | C-YELLOW-PHASE |
| R-SIG-03 | signal | soft | C-RED-PHASE / C-YELLOW-PHASE |
| R-YLD-01 | yield | hard | C-PED-IN-CROSSWALK |
| R-YLD-02 | yield | hard | C-ONCOMING-STRAIGHT |
| R-YLD-03 | yield | hard | C-ON-MINOR-ROAD |
| R-YLD-04 | yield | hard | C-EMERGENCY-ACTIVE |
| R-YLD-05 | yield | soft | C-PED-IN-CROSSWALK / C-ONCOMING-STRAIGHT |

若 W1 spike 显示 Boston-only 下 emergency keyframe 数少于 20，则删除 `C-EMERGENCY-ACTIVE` 与 `R-YLD-04`，并同步删除 override 示例或仅保留为扩展测试 fixture。

---

## 6. 核心接口规范

### 6.1 ConditionMatcher

```python
class ConditionMatcher:
    def __init__(
        self,
        condition_specs: dict[str, ConditionSpec],
        *,
        enable_keyword_fallback: bool = True,
    ) -> None: ...

    def match(self, query: SceneQuery) -> tuple[list[str], RetrievalMode]: ...
```

主路径：

```python
FACT_TO_CONDITION = {
    "has_red_light": "C-RED-PHASE",
    "has_yellow_light": "C-YELLOW-PHASE",
    "pedestrian_in_crosswalk": "C-PED-IN-CROSSWALK",
    "oncoming_vehicle_moving": "C-ONCOMING-STRAIGHT",
    "ego_on_minor_road": "C-ON-MINOR-ROAD",
    "emergency_vehicle_active": "C-EMERGENCY-ACTIVE",
}
```

匹配规则：

1. 先读取 `query.scene_facts` 中为 True 的事实，命中对应 Condition。
2. `scenario_type` 不参与条件补充，只用于编排层准入与一致性诊断。
3. 若结构化事实为空，且启用 fallback，则对英文关键词做包含匹配。
4. 关键词 fallback 命中时 `retrieval_mode=KEYWORD_FALLBACK`。
5. facts 命中后立即返回，关键词不参与；运行期不产出 `MIXED`（枚举仅为兼容保留）。

示例：

```python
def match(self, query: SceneQuery) -> tuple[list[str], RetrievalMode]:
    structured = self._conditions_from_facts(query.scene_facts)
    keyword = []
    if not structured and self.enable_keyword_fallback:
        keyword = self._conditions_from_keywords(query.keywords)
    mode = self._mode(structured, keyword)
    return sorted(set(structured + keyword)), mode
```

### 6.2 RuleGraph

```python
class RuleGraph:
    def load(self) -> None: ...
    def rules_for_conditions(self, condition_ids: list[str]) -> list[RuleNode]: ...
    def overrides_for(self, rule_ids: list[str], condition_ids: list[str]) -> list[RuleOverride]: ...
    def all_condition_specs(self) -> dict[str, ConditionSpec]: ...
```

`rules_for_conditions()` 必须先聚合后构造：

```python
def rules_for_conditions(self, condition_ids: list[str]) -> list[RuleNode]:
    cond_by_rule: dict[str, list[str]] = defaultdict(list)
    for cid in set(condition_ids):
        for rid, _, edge in self._g.in_edges(cid, data=True):
            if edge.get("rel") == "APPLIES_WHEN" and self._g.nodes[rid].get("kind") == "TrafficRule":
                cond_by_rule[rid].append(cid)

    return [
        self._build_rule_node(rid, sorted(conds), RetrievalMode.STRUCTURED)
        for rid, conds in sorted(cond_by_rule.items())
    ]
```

禁止：

```python
node.matched_conditions.append(cid)
```

因为 `RuleNode` 是 frozen 模型。

### 6.3 ConflictResolver

```python
class ConflictResolver:
    def resolve(
        self,
        rules: list[RuleNode],
        matched_conditions: list[str],
        overrides: list[RuleOverride],
    ) -> tuple[list[RuleNode], list[RuleOverride]]: ...
```

行为：

- 当 override 的 `condition_id` 在 `matched_conditions` 中，且覆盖方与被覆盖方均在召回规则中时生效。
- 被覆盖规则保留在 `RuleSubgraph.rules`，但 `status=OVERRIDDEN`，`overridden_by=[...]`。
- Layer 3 的 hard filter 只消费 `active_rules`。

典型冲突：

| 同时命中 | 处理 |
|---|---|
| C-RED-PHASE + C-EMERGENCY-ACTIVE | `R-YLD-04 OVERRIDES R-SIG-01` |

### 6.4 GraphRAGRetriever

```python
class GraphRAGRetriever:
    def __init__(
        self,
        settings: Layer2Settings,
        *,
        rule_graph: RuleGraph | None = None,
        matcher: ConditionMatcher | None = None,
        conflict_resolver: ConflictResolver | None = None,
    ) -> None: ...

    def retrieve(self, query: SceneQuery) -> RuleSubgraph: ...
```

编排：

```python
def retrieve(self, query: SceneQuery) -> RuleSubgraph:
    start = time.perf_counter()
    condition_ids, mode = self.matcher.match(query)
    rules = self.rule_graph.rules_for_conditions(condition_ids)
    overrides = self.rule_graph.overrides_for([r.node_id for r in rules], condition_ids)
    resolved_rules, applied = self.conflict_resolver.resolve(rules, condition_ids, overrides)

    return RuleSubgraph(
        scene_id=query.scene_id,
        frame_token=query.frame_token,
        rules=resolved_rules,
        matched_conditions=condition_ids,
        retrieval_mode=mode,
        overrides_applied=applied,
        query_duration_ms=int((time.perf_counter() - start) * 1000),
        graph_version=self.rule_graph.version,
        retrieval_timestamp=datetime.now(UTC),
    )
```

运行期无命中必须返回空 `RuleSubgraph`，不得抛异常。

### 6.5 Facade

```python
def retrieve(
    query: SceneQuery,
    *,
    retriever: GraphRAGRetriever | None = None,
) -> RuleSubgraph: ...
```

---

## 7. 错误处理

```python
class Layer2Error(Exception): ...
class GraphDefinitionError(Layer2Error): ...
class RuleGraphValidationError(Layer2Error): ...
```

只有启动/加载阶段允许抛错：

| 异常 | 触发 |
|---|---|
| `GraphDefinitionError` | YAML 不存在、不可读、格式错误 |
| `RuleGraphValidationError` | id 重复、悬空引用、非法 relation、override 指向不存在规则 |

`retrieve()` 阶段不得抛业务异常。

---

## 8. 日志与 Telemetry

每次 retrieve 记录：

| 字段 | 用途 |
|---|---|
| `scene_id`, `frame_token` | 调用链关联 |
| `retrieval_mode` | 统计 structured / keyword_fallback；mixed 仅兼容历史记录 |
| `matched_conditions` | 审计 |
| `num_rules_returned` | 召回规模 |
| `num_active_rules` | Layer 3 实际可用规则 |
| `overrides_applied` | 图冲突消解是否生效 |
| `query_duration_ms` | 性能 |

实验报告中必须区分：

- citation validity：返回 rule_id 是否存在于图中。Layer 2 构造性保证 100%。
- citation accuracy：返回 rule_id 是否适用于场景。需要人工子集或 benchmark 半自动校验，Layer 2 不声称天然 100%。

---

## 9. 非功能需求

| 指标 | 目标 |
|---|---|
| `retrieve()` P95 | ≤ 10 ms |
| 图加载 P95 | ≤ 500 ms |
| 关键词 fallback 占比 | ≤ 10%，若超出说明结构化事实不足 |
| citation validity | 100% |
| override 测试覆盖 | 100% 分支覆盖 |

`retrieve()` 对相同 `SceneQuery` 必须完全确定性，无随机数、无网络调用。

---

## 10. 测试要求

| 测试文件 | 必须覆盖 |
|---|---|
| `test_condition_matcher.py` | 结构化事实优先；英文关键词 fallback；scenario_type 与 facts 不一致时 facts 优先 |
| `test_rule_graph.py` | YAML 校验；英文 keywords；override 边加载；先聚合后构造 RuleNode |
| `test_conflict_resolver.py` | 红灯 + emergency override；未命中 condition 时 override 不生效 |
| `test_retriever.py` | `SceneQuery` 入参；空命中返回空 subgraph；active_rules 排除 overridden |
| `test_integration.py` | 每个 MVP condition 召回预期 rule；override 场景走多跳 |
| `benchmark_citation_validity.py` | 全量输出 rule_id 均存在于图中 |

新增引用准确性评估脚本：

```python
def evaluate_citation_accuracy(
    labels: list[ManualCitationLabel],
    outputs: list[RuleSubgraph],
) -> CitationAccuracyReport: ...
```

该脚本服务论文实验，不属于 Layer 2 runtime。

---

## 11. 对接说明

编排层调用：

```python
for bundle in parse_dataset(nuscenes_root, drivelm_qa_path=qa_path):
    subgraph = retriever.retrieve(bundle.scene_query)
    label = judge.judge(bundle.judge_input, subgraph)
```

Layer 3 只读取：

- `subgraph.hard_rules`
- `subgraph.soft_rules`
- `subgraph.active_rules`
- `subgraph.matched_conditions`
- `subgraph.overrides_applied`

Layer 3 不回读 `SceneQuery`，避免重复场景匹配。

---

## 12. 验收 Checklist

- [ ] `retrieve()` 入参为 `SceneQuery`，不再依赖完整 `SceneContext`。
- [ ] Condition 主路径由 `SceneFacts` 驱动。
- [ ] YAML keywords 改为英文为主。
- [ ] 关键词 fallback 可关闭，且触发率被 telemetry 记录。
- [ ] `OVERRIDES` / `EXCEPTION_WHEN` schema 与加载校验完成。
- [ ] 红灯 + emergency 冲突场景可触发 override。
- [ ] `RuleNode` frozen 模型不被运行期 mutate。
- [ ] `RuleSubgraph.active_rules` 排除 overridden rules。
- [ ] 文档和实验术语区分 citation validity 与 citation accuracy。
- [ ] Layer 2 无 LLM、无网络、无数据库凭据。

---

*文档结束，Layer 2 v3.0*
