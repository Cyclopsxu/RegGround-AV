# Layer 3 逻辑裁判层 · 功能与接口设计文档

**RegGround-AV · 法规接地的可解释轨迹偏好打标系统**

*Software Design Specification — LLM-as-Judge & Preference Labeling Layer*

| 项目 | 内容 |
|---|---|
| 文档版本 | v3.0 |
| 前序版本 | v2.0（2026-06）、v1.0（2026-04-17） |
| 文档状态 | 接口修订版，待开发实施 |
| 日期 | 2026-06 |
| 修订依据 | `docs/RegGround-AV_问题分析与修订清单.md` D16-D20、E21-E27、F28-F29 |

---

## 1. 修订目标

Layer 3 v2.0 已将系统定位为可审计偏好打标器，但仍存在若干关键缺口：

- 只校验 cited rule 是否存在，不校验引用是否恰当。
- `UNCERTAIN` 状态没有闭环语义。
- pointwise 1-5 分打分与“排序稳定”要求冲突。
- Anthropic API 不支持 seed，确定性声明过强。
- Layer 3 接收完整 `SceneContext`，对“不读坐标/标签”的隔离仍靠纪律。
- 实验设计缺少人工标注、难度梯度、2x2 消融和下游偏好一致性验证。

v3.0 修订为：

- 对外只接收 `JudgeInput` 与 `RuleSubgraph`。
- Hard Filter 输出 `VETOED / CLEARED / UNCERTAIN` 的完整闭环语义。
- Preference Ranker 改为 pairwise 比较，直接生成偏好对与排序。
- 继续做 citation validity 校验，并新增 citation accuracy 评估接口。
- 报告 `decided_by=rule_engine/llm` 占比，避免循环性评估。
- 将人工标注子集、难度分级和 2x2 消融列为验收要求。

---

## 2. 层级职责与边界

### 2.1 职责声明

Layer 3 消费 Layer 1 的 `JudgeInput` 与 Layer 2 的 `RuleSubgraph`，产出 `AuditLabel`。它负责把候选轨迹特征、可引用法规集合和 LLM 判断组合成可审计偏好标签。

### 2.2 In Scope

- Prompt 构建与泄露扫描。
- Hard Filter：规则引擎快速裁定 + LLM 兜底。
- Pairwise Preference Ranker：对通过硬过滤的轨迹两两比较并推导排序。
- 结构化输出校验、rule_id 回引校验、重试和降级。
- `AuditLabel` 组装，包括 `chosen_trajectory_id`、`preference_ranking`、`preference_pairs`、`legal_basis`。
- telemetry：token、耗时、decided_by 占比、citation validity、pairwise 一致性。
- 实验支撑：人工标注、引用恰当性、2x2 消融、难度分级指标。

### 2.3 Out of Scope

- 读取候选轨迹原始 waypoints。
- 读取 `variant_type`、`expected_verdict` 或 benchmark mapping。
- 法规图谱检索和规则内容维护。
- 完整 DPO/RLAIF 训练。v3.0 只提供偏好一致性 proxy 指标。

---

## 3. 模块架构

```
src/layer3/
    __init__.py
    models.py
    exceptions.py
    settings.py
    llm_client.py
    context_builder.py
    leakage_guard.py
    hard_filter.py
    rule_engine.py
    preference_ranker.py
    pairwise_aggregator.py
    citation_validator.py
    label_generator.py
    judge.py
    experiments/
        citation_accuracy.py
        ablation_2x2.py
        human_annotation_eval.py
```

| 模块 | 职责 |
|---|---|
| `context_builder.py` | `JudgeInput` + `RuleSubgraph` -> prompt context |
| `leakage_guard.py` | 完整 system/user prompt 禁词扫描 |
| `rule_engine.py` | 明显违规快速裁定；困难案例返回 UNCERTAIN |
| `hard_filter.py` | 规则引擎 + LLM 兜底，输出 VetoResult |
| `preference_ranker.py` | pairwise LLM 比较 |
| `pairwise_aggregator.py` | Bradley-Terry 或多数胜场生成 ranking |
| `citation_validator.py` | citation validity runtime 校验，accuracy 实验评估 |
| `label_generator.py` | 生成自然语言摘要 |
| `judge.py` | Facade，编排四阶段和降级 |

---

## 4. 对外接口与数据模型

### 4.1 JudgeInput

Layer 3 不再接收完整 `SceneContext`。

```python
class JudgeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    frame_token: str
    narrative: str
    trajectory_features: list[TrajectoryFeatures]
```

约束：

- 不含 `candidate_trajectories`。
- 不含 waypoints。
- 不含 benchmark labels。
- `trajectory_features[*].traj_id` 必须是匿名 ID，如 `traj_a`。
- `narrative` 与 feature summary 已由 Layer 1 做第一道泄露清洗；Layer 3 仍需在最终 prompt 上二次扫描。

### 4.2 枚举

```python
class VetoStatus(str, Enum):
    VETOED = "vetoed"
    CLEARED = "cleared"
    UNCERTAIN = "uncertain"

class JudgeStage(str, Enum):
    CONTEXT_BUILDING = "context_building"
    HARD_FILTER = "hard_filter"
    PAIRWISE_RANKING = "pairwise_ranking"
    LABEL_GENERATION = "label_generation"

class ReportStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"

class DecisionSource(str, Enum):
    RULE_ENGINE = "rule_engine"
    LLM = "llm"
    FALLBACK = "fallback"
```

### 4.3 JudgeContext

```python
class JudgeContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    scene_narrative: str
    hard_rules_text: str
    soft_rules_text: str
    trajectories_text: str
    hard_rule_ids: list[str]
    soft_rule_ids: list[str]
    trajectory_ids: list[str]
    total_prompt_tokens: int
```

`trajectories_text` 只能由 `TrajectoryFeatures` 生成，禁止包含坐标数组。

### 4.4 VetoResult

```python
class VetoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    status: VetoStatus
    violated_rule_ids: list[str] = Field(default_factory=list)
    reason: str
    decided_by: DecisionSource

    @model_validator(mode="after")
    def _consistency(self):
        if self.status == VetoStatus.VETOED and not self.violated_rule_ids:
            raise ValueError("vetoed trajectory must cite at least one hard rule")
        if self.status in {VetoStatus.CLEARED, VetoStatus.UNCERTAIN} and self.violated_rule_ids:
            raise ValueError("cleared/uncertain trajectory must not cite violated rules")
        return self
```

### 4.5 PairwisePreference

```python
class PairwisePreference(BaseModel):
    model_config = ConfigDict(frozen=True)

    preferred_id: str
    dispreferred_id: str
    basis: Literal["compliance", "pairwise_judgment"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    referenced_rule_ids: list[str] = Field(default_factory=list)
    decided_by: DecisionSource
```

`basis="compliance"` 表示 CLEARED 优于 VETOED，由规则自动生成，不需要 LLM pairwise 调用。

### 4.6 TrajectoryVerdict

```python
class TrajectoryVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    veto: VetoResult
    rank: int | None = None

    @model_validator(mode="after")
    def _consistency(self):
        if self.veto.status == VetoStatus.VETOED and self.rank is not None:
            raise ValueError("vetoed trajectory must not have rank among cleared items")
        if self.veto.status == VetoStatus.UNCERTAIN and self.rank is not None:
            raise ValueError("uncertain trajectory must not have pairwise rank")
        return self
```

UNCERTAIN 语义：

- 无 score、无 rank。
- 出现在 `preference_ranking` 中，排序位于 CLEARED 之后、VETOED 之前。
- 不参与 `preference_pairs`。
- 不可作为 `chosen_trajectory_id`。
- 降级路径 `_uncertain_for_all` 生成的标签 `chosen_trajectory_id=None`。

### 4.7 AuditLabel

```python
class AuditLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene_id: str
    frame_token: str
    status: ReportStatus
    chosen_trajectory_id: str | None
    preference_ranking: list[str]
    preference_pairs: list[PairwisePreference]
    verdicts: list[TrajectoryVerdict]
    natural_language_summary: str
    legal_basis: list[str]
    reasoning_chain: list[ReasoningStep]
    stage_diagnostics: list[StageDiagnostics]
    citation_validity: float
    total_duration_ms: int
    generated_at: datetime
```

排序约定：

```text
CLEARED（pairwise ranking 顺序） > UNCERTAIN（原输入匿名顺序） > VETOED（原输入匿名顺序）
```

`chosen_trajectory_id` 只从 CLEARED 中选。若没有 CLEARED，则为 `None`。

---

## 5. 配置

```python
class Layer3Settings(BaseSettings):
    llm_provider: Literal["openai", "anthropic", "openai_compatible"] = "openai_compatible"
    llm_model: str = "deepseek-v4-pro"
    llm_model_version: str = "DeepSeek-V4-Pro"
    llm_thinking_mode: Literal["enabled", "disabled"] = "enabled"
    llm_api_key: SecretStr
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 1
    llm_retry_backoff_base_seconds: float = 3.0
    llm_retry_backoff_multiplier: float = 4.0
    llm_structured_max_tokens: int = 8192
    llm_text_max_tokens: int = 4096

    hard_filter_temperature: float = 0.0
    pairwise_ranker_temperature: float = 0.0
    label_temperature: float = 0.2

    max_trajectories_per_judge: int = 10
    max_rules_in_context: int = 20
    max_pairwise_comparisons: int = 45
    enable_cot: bool = True
    archive_llm_io: bool = False

    model_config = SettingsConfigDict(env_prefix="LAYER3_")
```

不提供通用 `seed` 配置。若某 provider 支持 seed，可在 provider-specific 子配置中启用，但不得作为跨 provider 可复现性要求。

DeepSeek OpenAI-compatible 非流式响应的排队 keep-alive 空行由 SDK 处理。`deepseek-v4-pro` 默认 thinking，推理 token 与最终答案共享 `max_tokens`；结构化输出预算不得沿用过小的 2000-token 上限，解析失败日志须记录 `finish_reason` 和 reasoning 字符数，但不得记录 reasoning 正文。

若保留 pointwise ranker 作为消融模式，`score_weights` 必须有 validator：

```python
@field_validator("score_weights")
@classmethod
def _validate_weights(cls, v):
    expected = {d.value for d in ScoreDimension}
    if set(v.keys()) != expected:
        raise ValueError("score_weights keys must match ScoreDimension")
    if abs(sum(v.values()) - 1.0) > 1e-6:
        raise ValueError("score_weights must sum to 1.0")
    return v
```

---

## 6. 核心接口规范

### 6.1 ContextBuilder

```python
class ContextBuilder:
    def build(self, judge_input: JudgeInput, subgraph: RuleSubgraph) -> JudgeContext: ...
```

处理逻辑：

1. 只使用 `subgraph.active_rules`。
2. hard/soft rules 按 severity 与 consequence 截断。
3. 轨迹文本只来自 `trajectory_features.natural_language_summary` 和结构化特征字段。
4. 场景要件在相位行后明确渲染“信号灯相位为 keyframe 时刻（t=0）快照，其后相位未知”。
5. 运行时只扫描 Layer 1 产出的 narrative、scene facts 文本与每条 feature summary。

prompt 模板通过静态单测检查；法规文本不参与运行时结论词扫描，避免“必须/禁止”等法规措辞误报。

### 6.2 HardFilter

```python
class HardFilter:
    def filter(
        self,
        context: JudgeContext,
        features: list[TrajectoryFeatures],
    ) -> tuple[list[VetoResult], list[ReasoningStep]]: ...
```

双阶段：

- 规则引擎处理明显违规与明显通过。
- 对临界停车、缓慢滚动、停后提前起步、让行后期提前起步等困难案例返回 `UNCERTAIN`，交给 LLM 兜底。

`RuleEngine` 输出：

```python
class RuleEngineDecision(BaseModel):
    trajectory_id: str
    status: VetoStatus
    violated_rule_ids: list[str]
    reason: str
    is_decisive: bool
```

设计要求：

- Easy 违规可由规则引擎裁定。
- Medium/Hard 变体应尽量进入 LLM 兜底，避免生成逻辑和检测逻辑完全同源。
- telemetry 必须记录 `decided_by` 占比。

### 6.3 CitationValidator

```python
class CitationValidator:
    def validate_validity(self, cited_ids: Iterable[str], allowed_ids: set[str]) -> None: ...
    def extract_summary_citations(self, text: str) -> list[str]: ...
```

runtime 只保证 citation validity：

- `VetoResult.violated_rule_ids` 必须属于 `context.hard_rule_ids`。
- `PairwisePreference.referenced_rule_ids` 必须属于 active rule ids。
- `natural_language_summary` 中 `[R-...]` 必须属于 `legal_basis`。

citation accuracy 由实验模块评估，不在 runtime 自动断言。

### 6.4 PreferenceRanker

```python
class PreferenceRanker:
    def rank(
        self,
        context: JudgeContext,
        cleared_features: list[TrajectoryFeatures],
    ) -> tuple[list[str], list[PairwisePreference], list[ReasoningStep]]: ...
```

pairwise 流程：

1. 对 CLEARED 轨迹生成两两组合。
2. 每次 LLM 只比较 A 与 B 哪条更优。
3. 输出 `PairwisePreference`，引用 soft rules 或防御性理由。
4. 使用 Bradley-Terry 或多数胜场聚合为 ranking。
5. 若 CLEARED 数量为 0 或 1，不调用 LLM，直接返回空 pairs 或单元素 ranking。

Pairwise prompt 必须要求模型回答：

```json
{
  "preferred_id": "traj_a",
  "dispreferred_id": "traj_b",
  "confidence": 0.0,
  "reasoning": "...",
  "referenced_rule_ids": ["R-YLD-05"]
}
```

`preferred_id` 和 `dispreferred_id` 必须来自本次比较的两个 ID，否则抛 `StructuredOutputValidationError` 并重试。

### 6.5 LabelGenerator

```python
class LabelGenerator:
    def generate(
        self,
        judge_input: JudgeInput,
        subgraph: RuleSubgraph,
        verdicts: list[TrajectoryVerdict],
        preference_pairs: list[PairwisePreference],
        reasoning_chain: list[ReasoningStep],
    ) -> str: ...
```

摘要结构：

1. 结论：说明 chosen 轨迹，若无 chosen 则说明原因。
2. 否决说明：逐条说明 VETOED 轨迹与引用 rule_id。
3. 不确定说明：列出 UNCERTAIN 轨迹，不把它们包装成合规。
4. 偏好说明：解释 CLEARED 内部的 pairwise 排序依据。

### 6.6 AuditJudge

```python
class AuditJudge:
    def judge(self, judge_input: JudgeInput, subgraph: RuleSubgraph) -> AuditLabel: ...
```

核心编排：

```python
def judge(self, judge_input: JudgeInput, subgraph: RuleSubgraph) -> AuditLabel:
    ctx = self.context_builder.build(judge_input, subgraph)
    veto_results, hard_reasoning = self.hard_filter.filter(ctx, judge_input.trajectory_features)

    cleared = [f for f in judge_input.trajectory_features
               if veto_results_by_id[f.traj_id].status == VetoStatus.CLEARED]

    ranking_cleared, pairwise_pairs, rank_reasoning = self.preference_ranker.rank(ctx, cleared)
    compliance_pairs = self._build_compliance_pairs(veto_results)
    ranking = self._merge_ranking(ranking_cleared, veto_results, ctx.trajectory_ids)
    verdicts = self._build_verdicts(veto_results, ranking_cleared)
    summary = self.label_generator.generate(judge_input, subgraph, verdicts,
                                            compliance_pairs + pairwise_pairs,
                                            hard_reasoning + rank_reasoning)

    return AuditLabel(
        scene_id=judge_input.scene_id,
        frame_token=judge_input.frame_token,
        status=status,
        chosen_trajectory_id=ranking_cleared[0] if ranking_cleared else None,
        preference_ranking=ranking,
        preference_pairs=compliance_pairs + pairwise_pairs,
        verdicts=verdicts,
        natural_language_summary=summary,
        legal_basis=self._collect_legal_basis(...),
        citation_validity=self._compute_citation_validity(...),
        ...
    )
```

`judge()` 不应向上抛业务异常。无法完成 LLM 阶段时降级为 `PARTIAL` 或 `FAILED` 标签。

---

## 7. 错误处理与降级

| 触发 | 行为 | status |
|---|---|---|
| Context 超长且截断后仍失败 | 返回 stub label | FAILED |
| Hard Filter LLM 超时 | 未决轨迹标 UNCERTAIN | PARTIAL |
| Pairwise Ranker 超时 | 只保留 compliance pairs，CLEARED 按输入顺序 | PARTIAL |
| Label Generator 超时 | 使用模板摘要 | PARTIAL |
| Citation validity 失败 | 重试；连续失败后该阶段降级 | PARTIAL |

`HallucinationError` 重命名语义：

```python
class CitationValidityError(Layer3Error): ...
class CitationAccuracyEvaluationError(Layer3Error): ...
```

runtime 的错误名应避免暗示“所有幻觉都能被拦截”。

---

## 8. 日志与 Telemetry

每次 judge 输出：

| 字段 | 用途 |
|---|---|
| `scene_id`, `frame_token` | 调用链 |
| `report_status` | 完整/降级/失败 |
| `vetoed_count`, `cleared_count`, `uncertain_count` | 裁定分布 |
| `decided_by_rule_engine_count`, `decided_by_llm_count` | 避免循环性评估 |
| `pairwise_comparisons_count` | 成本 |
| `citation_validity` | 运行期引用存在性 |
| `citation_validity_retries` | 防越界引用重试 |
| `total_llm_tokens`, `total_duration_ms` | 成本与性能 |
| `chosen_position` | position bias sanity check |

实验脚本额外统计：

- `chosen_position_distribution`
- `rule_engine_vs_llm_accuracy`
- easy/medium/hard 分级准确率
- citation accuracy
- 与人工偏好标注的 Kendall's tau
- 与人工违规标注的 Cohen's kappa

---

## 9. 非功能需求

| 指标 | 目标 |
|---|---|
| 单次 judge P95 | ≤ 7000 ms |
| 端到端 P95 | ≤ 10000 ms |
| citation validity | 100% |
| top-1 稳定性 | 同输入连续 5 次 ≥ 95%，temperature=0 时目标 100% |
| full-rank Kendall's tau 稳定性 | ≥ 0.8 |
| prompt 泄露扫描 | 100% 覆盖完整 system + user prompt |

不再要求“DimensionScore ±1 但偏序不变”。v3.0 的主排序依据是 pairwise 结果。

---

## 10. 实验与评估要求

### 10.1 难度梯度

Layer 1 benchmark labels 必须提供 `difficulty`：

| 难度 | 示例 | 预期 |
|---|---|---|
| Easy | 完全未减速越线、明显不让行 | 规则引擎可判定 |
| Medium | 缓慢滚动通过、临界减速 | 部分进入 LLM |
| Hard | 停后提前起步、行人离开前 0.5 秒启动 | 主要由 LLM 语义判断 |

报告每个难度上的准确率、citation validity、citation accuracy。

### 10.2 人工标注子集

必须标注 50-100 个 keyframes。每个样本标注：

- 哪些轨迹违规。
- 哪条轨迹最优。
- pairwise 偏好关系。
- 每条违规应引用的规则类别或 rule_id。

指标：

- 系统违规判定 vs 人工：Cohen's kappa。
- 系统排序 vs 人工：Kendall's tau。
- cited rule vs 人工标注：citation accuracy。

### 10.3 幻觉判定标准

实验文档需定义：

- citation validity：引用 rule_id 是否存在于可引用集合。
- citation accuracy：引用内容是否与违规类型匹配。
- hallucination rate = invalid citations + inaccurate citations / total citations。

GraphRAG 系统与无 RAG baseline 使用同一标准。

### 10.4 2x2 消融

| 条件 | GraphRAG | 回引校验 | 目的 |
|---|---|---|---|
| Full | 是 | 是 | 完整系统 |
| Ablation A | 是 | 否 | 只看接地 |
| Ablation B | 否 | 是 | 只看回校验 |
| Baseline | 否 | 否 | 纯 LLM |

报告 citation validity、citation accuracy、偏好一致性、token 和耗时。

### 10.5 下游有效性 proxy

不要求完整训练，但必须报告：

- 系统 preference pairs 与人工 preference pairs 的一致率。
- chosen 轨迹与人工 top-1 的一致率。
- Kendall's tau。

论文中将完整 DPO/RLAIF 训练列为后续工作。

---

## 11. 测试要求

| 测试文件 | 必须覆盖 |
|---|---|
| `test_context_builder.py` | 入参为 `JudgeInput`；无 waypoints；Layer 1 运行时文本泄露扫描 |
| `test_hard_filter.py` | Easy 规则判定；Hard 返回 UNCERTAIN；引用越界重试 |
| `test_preference_ranker.py` | pairwise 输出 ID 约束；0/1 条 CLEARED 不调用 LLM；聚合排序 |
| `test_pairwise_aggregator.py` | 多数胜场；Bradley-Terry；平局 tie-break |
| `test_judge.py` | UNCERTAIN 排序居中；chosen 只来自 CLEARED；pairs 不包含 UNCERTAIN |
| `test_label_generator.py` | UNCERTAIN 不被描述为合规；summary 引用均在 legal_basis |
| `test_citation_validator.py` | validity 越界拦截；summary citation 提取 |
| `test_position_bias.py` | 候选顺序不同但 top-1 不固定偏第一位 |
| `test_integration.py` | Layer 1 bundle -> Layer 2 subgraph -> Layer 3 label |

隔离测试示例：

```python
def test_prompt_contains_no_variant_tokens(judge_input, subgraph):
    ctx = ContextBuilder(settings).build(judge_input, subgraph)
    full_prompt = render_prompt(ctx)
    banned = ["illegal", "ground_truth", "vetoed", "variant", "违规", "合规"]
    assert all(token not in full_prompt.lower() for token in banned)
```

UNCERTAIN 测试示例：

```python
def test_uncertain_semantics(label):
    uncertain = [v for v in label.verdicts if v.veto.status == VetoStatus.UNCERTAIN]
    assert all(v.rank is None for v in uncertain)
    assert all(p.preferred_id not in {v.trajectory_id for v in uncertain}
               and p.dispreferred_id not in {v.trajectory_id for v in uncertain}
               for p in label.preference_pairs)
```

---

## 12. 对接说明

```python
for bundle in parse_dataset(nuscenes_root, drivelm_qa_path=qa_path):
    subgraph = retriever.retrieve(bundle.scene_query)
    label = judge.judge(bundle.judge_input, subgraph)
    evaluate(label, bundle.benchmark_labels)
```

Layer 3 只接收 `bundle.judge_input`，不接收 `bundle.context` 或 `bundle.benchmark_labels`。

---

## 13. 验收 Checklist

- [ ] `judge()` 入参为 `JudgeInput` + `RuleSubgraph`。
- [ ] Context Builder 不含 waypoints、variant labels、expected verdict。
- [ ] Layer 1 narrative 与 feature summary 扫描可拦截标签词和合规结论词。
- [ ] `UNCERTAIN` 无 score/rank，不参与 preference_pairs，排序居中。
- [ ] `chosen_trajectory_id` 只从 CLEARED 中产生。
- [ ] Preference Ranker 使用 pairwise 比较而非默认 pointwise 打分。
- [ ] 不声明 Anthropic seed 可复现，稳定性用多次运行指标表达。
- [ ] `decided_by` 占比写入 telemetry。
- [ ] citation validity runtime 100% 校验。
- [ ] citation accuracy 有人工子集评估脚本。
- [ ] benchmark 按 Easy/Medium/Hard 分级报告。
- [ ] 50-100 个 keyframes 人工标注为必做项。
- [ ] 2x2 消融实验脚本和指标输出齐备。
- [ ] 偏好一致性 proxy 指标可从人工标注子集计算。

---

*文档结束，Layer 3 v3.0*
