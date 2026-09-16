# RegGround-AV · 设计审查报告

**法规接地的可解释轨迹偏好打标系统 — 问题清单 & 数据依赖说明**

| 文档属性 | 内容 |
|---|---|
| 审查对象 | 系统架构 v2.0 / SRS v2.0 / Layer1 v3.0 / Layer2 v2.1 / Layer3 v2.0 |
| 场景约束 | Boston-only（剔除全部新加坡场景） |
| 文档日期 | 2026-06 |

---

## 目录

1. [问题清单（按数据流逻辑排序）](#一问题清单按数据流逻辑排序)
   - A. 数据源与数据可得性
   - B. Layer 1：解析与合成
   - C. Layer 2：图谱与检索
   - D. Layer 3：判定与打标
   - E. 评估与实验设计
   - F. 跨层接口与工程
2. [数据依赖精确清单（Boston-only）](#二数据依赖精确清单boston-only)

---

## 一、问题清单（按数据流逻辑排序）

> 严重等级说明：🔴 必须修复（影响系统正确性或论文成立性）　🟡 建议修复（影响学术严谨度）　🟢 加分项（性价比高的改进）

---

### A. 数据源与数据可得性

#### A1. 法规地域错配 / 左右行混杂 🔴

**问题**：nuScenes 同时包含波士顿（右行）和新加坡（左行）场景，用中国道交法（右行语义）判定左行数据，"左转让对向直行"等规则方向完全相反，让行类判定结论全部错误。

**已决策**：只保留 Boston 场景，剔除全部新加坡场景（`log.location == "boston-seaport"`）。

**残留事项**：论文中仍需一句话声明"中国法规 + 美国道路"是简化假设（右行语义一致，故几何判定成立）。过滤逻辑应在 `DriveLMDatasetLoader.__init__` 中构建 Boston scene_token 白名单，DriveLM 顶层 key 直接与白名单交集过滤。

---

#### A2. 信号灯相位覆盖率未知，核实排期太晚 🔴

**问题**：DriveLM 的 `key_object_infos.Status` 是相位的唯一来源，但覆盖稀疏性未知。里程碑把可得性核实放在 W6，若触发"退为让行单类"退路，前面信号灯相关开发部分作废。Boston-only 后样本量再砍一半，风险更高。

**修复方案**：提前到 **W1 跑统计 spike**，统计：
- Boston ∩ DriveLM 交集的 keyframe 总数
- 含 traffic-element Status 字段的 keyframe 数及相位分布（red / yellow / green 各多少）
- 各让行细分（行人在横道内 / 对向车 / emergency 类别）的可用样本数

拿到三组数字后，场景范围（信号灯类去留、EMERGENCY 细分去留）可直接定死。

---

#### A3. 冲突区几何来源缺失 🔴

**问题**：`min_distance_to_conflict_m`（轨迹到冲突区的距离）是 Hard Filter 规则引擎和 Preference Ranker 的核心输入特征，但全套文档未说明停止线 / 人行横道坐标从何获取。

**修复方案**：引入 **nuScenes Map Expansion v1.3**（Boston-only 后仅需 `boston-seaport.json`），具体图层见第二部分数据清单。设计文档需补充 map expansion 作为新增外部依赖，`DriveLMDatasetLoader` 需同时初始化 `NuScenesMap` 对象，把附近 stop_line / ped_crossing 多边形预读进 `RawScene`，供 `TrajectoryAnalyzer` 计算冲突区距离。

---

#### A4. 轨迹窗口太短太稀 🟡

**问题**：设计采用 2Hz 关键帧 × 6 帧 = 3 秒 7 个点，可能覆盖不到完整"减速—停车"事件（正常路口停车减速过程约 4–6 秒）。nuScenes 的 ego_pose 实际随 LIDAR 帧高频记录（约 20Hz），完全可以取到密得多的位姿序列。

**修复方案**：通过 `sample_data`（LIDAR_TOP 通道）沿 `next` 链遍历取密集位姿（~20Hz），窗口放宽到 **≥6 秒**。重建的 waypoint 数量从 7 个增加到约 120 个，`came_to_stop` 和 `deceleration_start_ratio` 等特征的可靠性大幅提升。Layer 1 的性能预算（≤250ms）需重新评估是否仍可达。

---

#### A5. "GT = 合规"是未验证假设 🟡

**问题**：文档把 ego_pose 重建的真实轨迹直接标记 `expected_verdict="cleared"`，但人类司机也会抢黄灯、压线停车，benchmark 标签存在噪声。

**修复方案**：GT 轨迹在进入 `TrajectoryAugmentor` 之前先过一遍规则引擎快速筛查，明确违规的 GT 轨迹：(1) 从 benchmark 中排除，或 (2) 改标 `expected_verdict="vetoed"` 并记录。同时在论文局限章节讨论标签噪声。

---

#### A6. 样本统计单元口径模糊 🟡

**问题**：DriveLM 一个 scene 含多个 keyframe，文档混用 `scene_id` / `frame_token`，评估报告的统计单元（scene 还是 keyframe）未明确，影响论文样本量口径的前后一致性。

**修复方案**：明确将 **keyframe**（`sample_token`）作为处理和评估的基本单元，`scene_id` 只用于分组归属。所有评估指标（准确率、幻觉率、token 用量等）均以 keyframe 为分母报告。

---

### B. Layer 1：解析与合成

#### B7. 【最严重】traj_id 命名泄露变体标签 🔴

**问题**：合成轨迹 ID 格式为 `{scene_id}_synthetic_{variant_type}_{idx}`，含 "illegal" / "ground_truth" 等字样。而 traj_id 进入 LLM prompt（Context Builder 用它标记每条轨迹），Layer 3 §9.3 的测试代码也自己写了 `illegal_id = "..._illegal_..."`，坐实了 ID 里含标签。这直接打破"变体标签隔离"这一全文最重要的设计约束，整个"LLM 凭客观特征自主判断"的学术根基被瓦解。

**修复方案**：

1. `TrajectoryAugmentor` 内部仍用含 variant 的 ID 生成轨迹（用于 benchmark 侧映射）
2. 进入 `SceneContext` 前，在 `parse_scene` facade 中将候选 ID **重写为匿名 ID**（`traj_a`, `traj_b`, `traj_c`…）并**随机打乱顺序**（固定 seed=42 的随机排列）
3. variant → 匿名 ID 的映射表只保留在 benchmark 侧，绝不进入任何 Layer 调用路径
4. 隔离测试改为对**整个 prompt 文本**进行违禁 token 扫描（不只查 Pydantic 字段），检查"illegal"/"ground_truth"/"vetoed" 等词是否出现

```python
# parse_scene facade 中的匿名化示意
def _anonymize_candidates(augmented: list[AugmentedTrajectory],
                           rng: np.random.Generator
                           ) -> tuple[list[Trajectory], dict[str, str]]:
    """返回 (匿名化候选列表, {匿名id: 原始variant_type})，后者只用于 benchmark。"""
    labels = [chr(ord('a') + i) for i in range(len(augmented))]
    rng.shuffle(labels)
    mapping = {}
    result = []
    for aug, label in zip(augmented, labels):
        anon_id = f"traj_{label}"
        mapping[anon_id] = aug.variant_type
        result.append(aug.trajectory.model_copy(update={"traj_id": anon_id}))
    return result, mapping
```

---

#### B8. 候选呈现顺序固定 + 位置偏置未检验 🟡

**问题**：GT 轨迹固定排第一也是隐性泄露（LLM 倾向于偏好首位选项）。LLM-as-Judge 的 **position bias**（位置偏置）是文献中公认的问题，但当前设计未提及任何检验或缓解措施。

**修复方案**：B7 的随机打乱顺序同时解决本问题。评估中额外报告一项指标：chosen 是否总是某个固定位置（如始终排在第 1 位），作为 position bias sanity check，写入论文实验章节。

---

#### B9. narrative 通道无泄露审计 🟡

**问题**：FR-03.2 只禁止 `natural_language_summary` 含合规性结论词，但 DriveLM QA 原文（可能含"应停车等待信号灯"类规范表述）不经任何审查直接进入 prompt（narrative 字段）。关键词表里 RED_LIGHT 的触发词"闯红灯"本身也是结论性词汇，若出现在 QA 文本中等于变相提示 LLM。

**修复方案**：在 `SceneTextExtractor.extract()` 处理 narrative 时，过滤 / 遮盖预定义的合规性结论词（扩展 FR-03.2 的禁止词表到 narrative 字段）；同时隔离测试改为扫描最终传入 LLM 的**完整 system + user prompt 文本**。

---

#### B10. 红灯违规扰动逻辑错误 🔴

**问题**：`case RED_LIGHT: return pts` 原样返回 GT，注释说"不插入减速段"，但 GT 本身在红灯场景就是减速停车轨迹——原样返回得到的仍是合规轨迹，`ILLEGAL` 变体的 `expected_verdict` 标记"vetoed"而轨迹本身是合规的，完全自相矛盾。

**修复方案**：红灯违规与让行违规的几何语义相同（"在应停处不停"），直接复用 `_remove_deceleration`：

```python
case ScenarioType.RED_LIGHT:
    # 红灯违规：抹掉停止线前的减速段，以稳定速度越线通过
    return self._remove_deceleration(pts)
```

---

#### B11. 黄灯扰动 `x *= 1.4` 不是加速 🔴

**问题**：纵向坐标整体乘以 1.4 改变的是路径几何（终点推远 40%、弯道形状畸变），不是速度剖面。如果轨迹本身有弯道，坐标缩放会使路径偏离道路边界，产生物理上不可行的轨迹。

**修复方案**：在时间维度重采样（相同路径、更短时间间隔），等效模拟加速：

```python
case ScenarioType.YELLOW_LIGHT:
    # 黄灯违规：加速抢行——相同路径、时间压缩到 70%
    t_orig = np.array([w[2] for w in pts_with_t])
    t_new  = t_orig * 0.7   # 时间缩短 → 速度提升约 1.43x
    # 对 x, y 按原时间重采样到 t_new
    return resample_trajectory(pts, t_orig, t_new)
```

---

#### B12. 关键词表语言不匹配 🔴

**问题**：DriveLM-nuScenes 的 QA 文本全是**英文**，而 `SCENARIO_KEYWORD_MAP` 与 Layer 2 YAML 中的 Condition `keywords` 全是中文（"红灯""行人"…）。在真实数据上关键词匹配率为零，所有场景退化为 `UNKNOWN`，信号灯和让行类均无法触发，整个 GraphRAG 机制失效。

**修复方案**：关键词表改为英文（或中英双语），示例：

```python
SCENARIO_KEYWORD_MAP = {
    ScenarioType.RED_LIGHT:    ["red light", "traffic light", "stop line", "red signal"],
    ScenarioType.YELLOW_LIGHT: ["yellow light", "amber light", "yellow signal"],
    ScenarioType.PEDESTRIAN:   ["pedestrian", "crosswalk", "crossing", "zebra crossing"],
    ScenarioType.ONCOMING:     ["oncoming", "opposite direction", "left turn", "yield"],
    ScenarioType.MINOR_ROAD:   ["minor road", "side road", "merge", "main road"],
    ScenarioType.EMERGENCY:    ["ambulance", "police car", "fire truck", "emergency vehicle", "siren"],
}
```

Layer 2 `rule_graph.yaml` 中各 Condition 的 `keywords` 字段同步改为英文。

---

### C. Layer 2：图谱与检索

#### C13. 条件匹配输入端脆弱，关键词判场景不可靠 🟡

**问题**：仅靠 QA 文本关键词判断"对向有直行车""行人在横道上"可靠性很低（"left turn"出现 ≠ 场景中真的有对向车）。论文也应诚实讨论：零检索幻觉（Layer 2 不虚构法条）≠ 零错误（Condition 误判照样召回错误法规，只是错误法规真实存在）。

**修复方案**：对让行类 Condition，以 nuScenes **3D 标注 + 地图几何**作为主路径，文本关键词降为兜底：

| Condition | 主路径（结构化） | 兜底（关键词） |
|---|---|---|
| C-PED-IN-CROSSWALK | `sample_annotation` 中有 `human.pedestrian.*` 且 annotation 坐标落在最近 `ped_crossing` 多边形内 + attribute `pedestrian.moving` | "pedestrian", "crosswalk" |
| C-ONCOMING-STRAIGHT | `sample_annotation` 中有对向 `vehicle.*` 且朝向与 ego 朝向夹角 > 150° + `vehicle.moving` | "oncoming", "left turn" |
| C-EMERGENCY-ACTIVE | `sample_annotation` 中有 `vehicle.emergency.*` + `vehicle.moving` | "ambulance", "siren" |

`ConditionMatcher.match()` 增加 `scene_facts: SceneFacts`（从 RawScene 预提取的结构化事实）入参，或将结构化事实预写入 `SceneContext`（推荐，对 Layer 2 透明）。

---

#### C14. "图"只有一跳，与字典查表无本质区别 🟢

**问题**：图深度仅一跳（Condition → Rule），`CO_OCCURS` / `related_rules` 定义了但不参与任何检索或判定逻辑。答辩被问"为什么需要 Graph，和一个 Python dict 有什么区别"时难以回应。

**修复方案（最划算加分项）**：增加 `OVERRIDES` / `EXCEPTION` 关系，设计**规则冲突场景**：

> 场景：路口红灯（触发 R-SIG-01 停车），同时后方有执行任务的救护车（触发 R-YLD-04 让行）。两条 hard rule 冲突——停车 vs 为救护车让行。需沿 `OVERRIDES` 边遍历优先级消解（R-YLD-04 `OVERRIDES` R-SIG-01 when C-EMERGENCY-ACTIVE）。

这一个场景让"图结构遍历"真正发挥作用，只需增加 1–2 条边和约 10 行 `rules_for_conditions` 的扩展逻辑，但论文亮点度大幅提升。YAML 示例：

```yaml
rules:
  - id: R-YLD-04
    ...
    overrides:
      - rule_id: R-SIG-01
        condition: C-EMERGENCY-ACTIVE
        description: 执行任务特种车辆让行优先级高于红灯停车义务
```

---

#### C15. frozen 模型上 mutate 🟡

**问题**：`rules_for_conditions` 参考实现对 `frozen=True` 的 `RuleNode` 执行 `matched_conditions.append(cid)`，违背 frozen 语义（虽然 list 内容技术上可变，但违反设计约定，且 Pydantic v2 可能在未来版本报错）。

**修复方案**：改为收集完所有 matched conditions 后再实例化 RuleNode：

```python
# 先聚合，后实例化
cond_by_rule: dict[str, list[str]] = defaultdict(list)
for cid in cond_set:
    for rid, _ in self._g.in_edges(cid):
        if self._g.nodes[rid].get("kind") == "TrafficRule":
            cond_by_rule[rid].append(cid)

return [self._build_rule_node(rid, conds, RetrievalMode.STRUCTURED)
        for rid, conds in cond_by_rule.items()]
```

---

### D. Layer 3：判定与打标

#### D16. 防幻觉只防"引用不存在"，不防"引用错误" 🟡

**问题**：`_validate_rule_references` 只校验 `cited_rule_id ∈ 可引用集合`。LLM 完全可以引用真实存在的 `R-SIG-01`（红灯停车）去解释一个让行违规——**错误归因**是更隐蔽、更危险的幻觉，当前机制对此完全透明。同时，"检索零幻觉（构造性）"这一 claim 是 trivially true（定义使然），作为学术贡献分量很轻，审稿人会指出"这是定义不是实验结果"。

**修复方案**：
1. 在论文中区分两类幻觉：**citation validity**（引用存在性，当前已防）vs **citation accuracy**（引用恰当性，当前未防）
2. 对 50–100 个有人工标注的子集，检验每条 veto 的 `cited_rule_id` 是否对应正确的违规类型（可借助 benchmark 的 `variant_type` 半自动核对：ILLEGAL 轨迹在信号灯场景 → 应引用 R-SIG-xx，让行场景 → R-YLD-xx）
3. 论文 claim 重心从"检索零幻觉"移向"端到端引用有效率（100%）+ 引用恰当性（人工核对子集上 X%）"

---

#### D17. UNCERTAIN 状态语义未闭环 🔴

**问题**：`TrajectoryVerdict` 的 `_consistency` validator 只覆盖 VETOED / CLEARED 两种状态，UNCERTAIN 轨迹在以下场景均无定义：有无 score、进不进 preference_ranking、排序排哪里、参不参与 preference_pairs。降级路径 `_uncertain_for_all` 恰恰会批量产生 UNCERTAIN，使降级后的 AuditLabel 结构不确定。`judge()` 伪代码中 `verdicts_have_cleared` 也是未定义变量。

**修复方案**：

```python
# 补充 UNCERTAIN 语义约定
class TrajectoryVerdict(BaseModel):
    ...
    @model_validator(mode="after")
    def _consistency(self):
        if self.veto.status == VetoStatus.VETOED and self.score is not None:
            raise ValueError("vetoed trajectory must not have score")
        if self.veto.status == VetoStatus.CLEARED and self.score is None:
            raise ValueError("cleared trajectory must have score")
        # UNCERTAIN：无 score，排在 CLEARED 之后、VETOED 之前
        if self.veto.status == VetoStatus.UNCERTAIN and self.score is not None:
            raise ValueError("uncertain trajectory must not have score")
        return self
```

排序约定：`CLEARED（按 overall_score 降序）> UNCERTAIN > VETOED`。UNCERTAIN 轨迹不参与 preference_pairs，但出现在 preference_ranking 中（居中位）。`chosen_trajectory_id` 只从 CLEARED 中取，CLEARED 为空时为 None。

---

#### D18. ±1 分浮动 vs 偏序不变自相矛盾 🟡

**问题**：NFR-04 同时声称"DimensionScore 允许 ±1 浮动"和"偏好偏序关系不变"。在 5 分制、两维加权（0.6/0.4）下，±1 分浮动完全可以翻转相邻轨迹的排名（例如：轨迹 A 得 4/4，轨迹 B 得 3/5，overall 相差仅 0.2，±1 浮动即翻转）。两个约束逻辑上不相容。

**修复方案（推荐）**：将 Preference Ranker 从 **pointwise 打分改为 pairwise 比较**：

- 每次调用 LLM 比较两条轨迹哪条更优（A > B？），输出胜负 + 依据
- 用 Bradley-Terry 或简单多数投票从 pairwise 结果推导排序
- 直接产出 `PreferencePair`，与 RLAIF 叙事更贴合
- 一致性文献上 pairwise 显著优于 pointwise，可作为对比实验写入论文

若维持 pointwise，则将 NFR-04 改为："同输入连续 5 次，`preference_ranking` 的 **top-1 选择** 100% 一致（不要求全序不变）"。

---

#### D19. Anthropic API 无 seed 参数 🟡

**问题**：Layer3Settings 写了 "seed 可配"，但 Anthropic 原生 API 不支持 seed 参数。NFR-04 的可复现性声明在 `llm_provider=anthropic` 时不成立。

**修复方案**：两选一：
1. Preference Ranker 也用 `temperature=0`（与 Hard Filter 一致），完全规避随机性（代价：打分多样性下降）
2. 将可复现性指标弱化为统计一致性：连续 5 次运行，preference_ranking 的 top-1 一致率 ≥ 95%，full-rank Kendall's τ ≥ 0.8

---

#### D20. score_weights 与 ScoreDimension 无一致性校验 🟡

**问题**：`score_weights` 是自由 dict，`ScoreDimension` 是枚举，两者的 key 对齐和"权重之和 = 1"均无验证，配置错误只会在运行时体现为打分结果异常（而非启动时报错）。

**修复方案**：在 `Layer3Settings` 加 validator：

```python
@field_validator("score_weights")
@classmethod
def _validate_weights(cls, v):
    expected = {d.value for d in ScoreDimension}
    if set(v.keys()) != expected:
        raise ValueError(f"score_weights keys must match ScoreDimension: {expected}")
    total = sum(v.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"score_weights must sum to 1.0, got {total}")
    return v
```

---

### E. 评估与实验设计

#### E21. 【高危】评估循环性：生成与检测是同一逻辑的正反面 🔴

**问题**：违规轨迹按规则合成（"抹掉减速段" → `came_to_stop=False`），规则引擎按相同特征检测（`came_to_stop=False` → 违规）。两者共享逻辑，规则引擎当然能检出自己造的违规，benchmark 准确率近乎同义反复。这是评审最可能攻击的点，会被质疑"LLM 在这个实验里究竟贡献了什么？"

**修复方案**：
1. 在 telemetry 中必须报告 `decided_by=rule_engine` vs `llm` 的占比，在论文中坦诚区分两类判定
2. 设计**难度梯度变体**（规则引擎判不了、必须 LLM 语义推理的案例），例如：
   - **临界停车**：在停止线前 0.3m 停车后缓慢越线（几何上"stopped"，但仍违规）
   - **缓慢滚动通过**（creeping）：速度降到 0.5 m/s 但未完全停止
   - **停后提前起步**：完全停车后在信号未变时提前起步
   - **让行后期**：大幅减速但在行人离开前 0.5 秒提前起步
3. 规则引擎对上述模糊案例应返回 `UNCERTAIN`，LLM 接管，这才能真正体现 LLM 的语义推理价值

---

#### E22. 合成变体区分度过低，实验缺乏区分力 🟡

**问题**：当前只有"匀速越线 vs 减速停车"这种极端对比，对 LLM 是送分题，准确率贴天花板，无法区分"有 GraphRAG"和"无 GraphRAG"两个系统的实质性差异（与 E21 同根）。

**修复方案**：与 E21 的难度梯度变体一并实施。变体难度分三级（Easy / Medium / Hard），分级报告各级准确率，展示"Hard 案例上 GraphRAG 显著优于基线"才是有说服力的实验结论。

---

#### E23. 缺乏独立 ground truth，所有指标均为自我验证 🟡

**问题**：FR-07.3 将人工标注列为"可选"，但没有独立于系统自身的参照，所有指标（准确率、幻觉率、偏好一致性）都是内部自洽验证，缺乏外部有效性。

**修复方案**：升级为**必做项**：对 50–100 个场景做人工标注（每个场景标注"哪条轨迹更优"和"是否违规"），计算：
- 系统判定 vs 人工标注的一致性（Cohen's κ）
- 偏好排序的 Kendall's τ

样本量 100 个约需 3–4 小时（每场景约 2 分钟），在 W6 联调阶段完成可行。这是让"下游可用"声称有最小依据的最低门槛。

---

#### E24. 基线幻觉率判定标准未操作化 🟡

**问题**："无 RAG 基线幻觉率预期 ≥ 20%"缺乏明确的测量方法。基线没有可引用集合，它引用"道交法第 38 条"算不算幻觉？是核对条号不存在才算，还是条号存在但内容引错也算？否则"≥ 20%"只是预设结论。

**修复方案**：预先建立**幻觉判定标准文档**（一页），包含：
1. Citation validity（引用存在性）：对照真实条文数据库（可用教材/法规全文）核对条号是否存在
2. Citation accuracy（引用恰当性）：核对引用条文内容是否与判定结论匹配
3. 幻觉率 = (invalid_citations + inaccurate_citations) / total_citations

基线与 GraphRAG 系统用同一标准，对比才有意义。

---

#### E25. 对比实验归因不清 🟡

**问题**：系统 = GraphRAG 接地 + rule_id 回校验拦截，基线两者皆无。观察到的幻觉率差异无法归因于是"接地"的效果还是"回校验"的效果。

**修复方案**：做 **2×2 消融实验**（成本极低，就是参数开关组合）：

| 实验条件 | GraphRAG | 回校验拦截 | 预期幻觉率 |
|---|---|---|---|
| 完整系统（本文） | ✅ | ✅ | 最低（构造性 0% + 重试拦截） |
| 消融 A | ✅ | ❌ | 低（接地限制引用范围，但无拦截） |
| 消融 B | ❌ | ✅ | 中（拦截有效，但 LLM 可任意引用） |
| 基线 | ❌ | ❌ | 最高 |

四格数据写入论文 Ablation Study 章节，清晰归因两个组件各自的贡献。

---

#### E26. "检索零幻觉"claim 学术分量轻 🟡

**问题**：Layer 2 确定性检索的零幻觉是构造性成立（定义使然，不调 LLM 则不可能虚构），不是实验发现，审稿人会说"这是实现选择不是科学结果"。

**修复方案**：论文叙事重心从"检索零幻觉（Layer 2 零虚构）"移向：
- **端到端引用有效率**：最终 AuditLabel 中所有 rule_id 100% 存在于图中（这跨了所有三层，是系统性质）
- **引用恰当率**：人工子集上 cited rule 与违规类型匹配的比例
- 对比实验中，基线的幻觉（虚构 / 错误引用）vs 本系统的对应指标

---

#### E27. 下游有效性零验证 🟡

**问题**：系统定位是"产出 RLAIF 可用的偏好标签"，但全套文档没有任何下游验证，"可直接作为训练信号"始终只是声称。

**修复方案**：最小补救（不需完整训练）：以系统产出的偏好对 vs 人工标注的偏好，计算**偏好一致性**（E23 人工标注子集复用）。在论文中坦诚"完整 DPO 训练验证"是局限与后续工作，但给出偏好一致性作为 proxy 指标。

---

### F. 跨层接口与工程

#### F28. 隔离靠纪律不靠类型系统 🟡

**问题**：Layer 3 接收完整 `SceneContext` 但"约定"不读 waypoints；Layer 2 接收完整 `SceneContext` 但实际只用两个字段，且造成 `layer2 → layer1` 的类型依赖（Layer 2 需 import Layer 1 的 model）。这两处隔离都靠人工纪律而非编译器/类型系统保证，容易在未来迭代中悄悄破坏。

**修复方案**：按**最小可见性**原则收窄层间契约：

```python
# Layer 2 入参：窄接口，不依赖完整 SceneContext
class SceneQuery(BaseModel):
    scene_id: str
    scenario_type: ScenarioType
    keywords: list[str]
    scene_facts: SceneFacts  # 结构化事实（C13 新增）

# Layer 3 入参：窄接口，LLM 物理上无法见到坐标
class JudgeInput(BaseModel):
    scene_id: str
    narrative: str           # 已过违禁词审计（B9）
    trajectory_features: list[TrajectoryFeatures]  # 只含特征，无 waypoints
    # traj_id 已匿名化（B7）
```

编排层（`__main__` / CLI）负责从 `SceneContext` 适配到 `SceneQuery` 和 `JudgeInput`。"LLM 见不到坐标/标签"从"测试保证"升级为"类型层面不可能"，与"构造性防幻觉"的设计哲学一脉相承。

---

#### F29. 测试细节缺陷 🟡

**问题一**：隔离测试 `assert not hasattr(ctx, "variant_type")` 太弱——问题不在 Pydantic 字段，在于 traj_id 字符串本身含标签（见 B7）。

**问题二**：L1 §9.3 测试代码 `parse_scene(load_fixture(...))` 传入 JSON fixture，但 `parse_scene` 签名接收 `RawScene` 对象，类型不符。

**修复方案**：
- 隔离测试改为：`assert "illegal" not in full_prompt_text and "ground_truth" not in full_prompt_text`，需 mock Context Builder 暴露完整 prompt 文本
- 测试 fixture 改为 `parse_scene(RawScene(**load_fixture(...)))`，或提供专门的 `parse_scene_from_json` 测试辅助函数

---

## 二、数据依赖精确清单（Boston-only）

### 2.1 场景筛选：剔除新加坡

**筛选链**：`scene.json → log_token → log.json → location`

| 文件 | 路径 | 用到的字段 | 用途 |
|---|---|---|---|
| `scene.json` | `v1.0-trainval/scene.json` | `token`, `log_token`, `first_sample_token`, `name`, `description` | 场景枚举与索引 |
| `log.json` | `v1.0-trainval/log.json` | `token`, **`location`** | `location == "boston-seaport"` 保留；`singapore-*` 全部剔除 |

在 `DriveLMDatasetLoader.__init__` 中构建 Boston scene_token 白名单，DriveLM 顶层 key 直接与白名单做交集过滤。

---

### 2.2 自车轨迹重建（密集位姿，A4 修复后）

**链路**：`sample → sample_data(LIDAR_TOP) → ego_pose`

| 文件 | 路径 | 用到的字段 | 用途 |
|---|---|---|---|
| `sample.json` | `v1.0-trainval/sample.json` | `token`, `timestamp`, `scene_token`, `next`, `prev` | 关键帧定位（DriveLM 的 frame_token = sample token）；沿 `next` 链取未来帧 |
| `sample_data.json` | `v1.0-trainval/sample_data.json` | `token`, `sample_token`, `ego_pose_token`, `calibrated_sensor_token`, `timestamp`, `is_key_frame`, `next`, `filename` | LIDAR_TOP 通道全帧（含非关键帧，~20Hz），沿 `next` 链跨 sample 取未来 ≥6 秒的密集位姿 |
| `calibrated_sensor.json` | `v1.0-trainval/calibrated_sensor.json` | `token`, `sensor_token` | sample_data → 传感器的桥接 |
| `sensor.json` | `v1.0-trainval/sensor.json` | `token`, `channel` | 过滤 `channel == "LIDAR_TOP"` |
| `ego_pose.json` | `v1.0-trainval/ego_pose.json` | `token`, **`translation`** `[x,y,z]`（全局坐标）, **`rotation`** `[w,x,y,z]`（四元数）, **`timestamp`**（微秒） | 重建轨迹的核心数据：全局位姿 → 关键帧局部坐标系；相邻帧 Δposition / Δt 计算 `speed_mps` |

> **注**：用 nuscenes-devkit 时这五张表由 `NuScenes` 对象封装，但设计文档应明确依赖的就是这五张表，方便 fixture 构造和脱离 devkit 的纯函数单元测试。

---

### 2.3 让行场景的结构化事实（C13 修复所需）

| 文件 | 路径 | 用到的字段 | 用途 |
|---|---|---|---|
| `sample_annotation.json` | `v1.0-trainval/sample_annotation.json` | `token`, `sample_token`, `instance_token`, `translation` `[x,y,z]`, `size` `[w,l,h]`, `rotation`（四元数）, `attribute_tokens`, `next`, `prev` | 关键帧内全部物体 3D 框：行人位置（配合地图判"在横道内"）、对向车位置与朝向（判"对向直行"）、跨帧跟踪判运动趋势 |
| `instance.json` | `v1.0-trainval/instance.json` | `token`, `category_token` | annotation → 类别的桥接 |
| `category.json` | `v1.0-trainval/category.json` | `token`, `name` | 类别判定：`human.pedestrian.*`（行人）、`vehicle.car/bus/truck/...`（对向车）、**`vehicle.emergency.ambulance` / `vehicle.emergency.police`**（特种车辆，EMERGENCY 场景） |
| `attribute.json` | `v1.0-trainval/attribute.json` | `token`, `name` | `pedestrian.moving` vs `pedestrian.standing_and_waiting`（行人是否正在通过）；`vehicle.moving`（对向车是否行驶中） |

> **警告**：nuScenes 中 emergency 类别样本极少，Boston-only 后 EMERGENCY 场景大概率不足。W1 spike 时一并统计，若可用样本 < 20，建议从 Layer 2 的法规范围中删除 R-YLD-04 和 C-EMERGENCY-ACTIVE（符合范围铁律）。

---

### 2.4 冲突区几何（A3 修复：新增数据依赖）⚠️

需单独下载 **nuScenes Map Expansion v1.3**（Boston-only 后仅需一张地图）。

**访问方式**：
```python
from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(dataroot="/data/nuscenes", map_name="boston-seaport")
# 按 ego 全局坐标查询附近图层
stop_lines = nusc_map.get_records_in_radius(ego_x, ego_y, radius=50, layer_names=["stop_line"])
```

| 文件 | 路径 | 图层与字段 | 用途 |
|---|---|---|---|
| `boston-seaport.json` | `maps/expansion/boston-seaport.json` | **`stop_line`**（多边形）：`token`, `polygon_token`, `stop_line_type` ∈ {TRAFFIC_LIGHT, PED_CROSSING, STOP_SIGN, TURN_STOP, YIELD} | 信号灯场景冲突区 = `stop_line_type == TRAFFIC_LIGHT` 的停止线；`min_distance_to_conflict_m` 和 `came_to_stop` 的几何基准 |
| 同上 | 同上 | **`ped_crossing`**（多边形）：`token`, `polygon_token` | 行人让行场景冲突区；配合 `sample_annotation` 判"行人在横道内" |
| 同上 | 同上 | **`traffic_light`**（点+朝向）：`token`, `pose` | 确认场景中确有信号灯（注意：仍无相位！相位只能从 DriveLM Status 获取）；辅助 scenario_type 推断 |
| 同上 | 同上 | `road_segment`：`token`, `polygon_token`, `is_intersection` | 判断关键帧是否处于路口（`LocationType.INTERSECTION`）；MINOR_ROAD 场景辅助判断 |
| 同上 | 同上 | `lane`：`token`, `polygon_token`, `from_edge_line_token`, `to_edge_line_token`, `lane_divider_segment_tokens` | 支路/主路关系判定（MINOR_ROAD 场景）；车道归属判断 |

---

### 2.5 场景语义与信号灯相位（DriveLM）

**文件**：`v1_0_train_nus.json`（DriveLM-nuScenes 训练集，单一文件）

**顶层结构**：
```
{
  "<scene_token>": {                     ← 与 nuScenes scene.token 对齐
    "key_frames": {
      "<sample_token>": {                ← 与 nuScenes sample.token 对齐（= frame_token）
        "key_object_infos": {
          "<id,CAM_FRONT,x,y>": {
            "Category": "...",           ← 关键对象类别
            "Status":   "...",           ← ★ 信号灯相位的唯一来源
            "Visual_description": "...", ← 辅助（如 "red traffic light ahead"）
            "2d_bbox": [x1, y1, x2, y2]
          }
        },
        "QA": {
          "perception":  [{"Q": "...", "A": "..."}, ...],
          "prediction":  [...],
          "planning":    [...],
          "behavior":    [...]
        }
      }
    }
  }
}
```

| 字段 | 用途 | 关联问题 |
|---|---|---|
| 顶层 scene_token（与 nuScenes 对齐） | Boston 白名单过滤的操作对象 | A1 |
| `key_frames` 的 sample_token | **统计单元**（keyframe 级，A6 决策），对应 `parse_dataset` 的迭代粒度 | A6 |
| `QA.perception / prediction / planning` 的 `Q` / `A` 文本 | 拼 narrative、提 keywords；**全部为英文**，关键词表必须配英文（B12）；进 prompt 前过违禁词审计（B9） | B9, B12 |
| `key_object_infos.*.Category` 含 "traffic" / "signal" 关键词的对象 + `.Status` | **唯一信号灯相位来源**；扫描所有 key_object_infos 条目，匹配 traffic element 类别后读 Status（"red" / "green" / "yellow" 等字符串）；W1 spike 统计 Boston 子集中有效 Status 的 keyframe 覆盖率 | A2 |
| `key_object_infos.*.Category`（pedestrian / vehicle 等） | 让行场景辅助先验（主路径用 nuScenes 3D 标注，此处为补充） | C13 |

---

### 2.6 完整数据依赖汇总

```
nuScenes v1.0-trainval/（共 12 张 JSON 表）
├── 场景筛选（Boston-only）
│   ├── scene.json          → token, log_token, first_sample_token
│   └── log.json            → token, location ★过滤条件
│
├── 密集轨迹重建
│   ├── sample.json         → token, timestamp, scene_token, next, prev
│   ├── sample_data.json    → token, sample_token, ego_pose_token,
│   │                          calibrated_sensor_token, timestamp,
│   │                          is_key_frame, next
│   ├── calibrated_sensor.json → token, sensor_token
│   ├── sensor.json         → token, channel（过滤 LIDAR_TOP）
│   └── ego_pose.json       → token, translation, rotation, timestamp ★核心
│
└── 让行场景结构化事实
    ├── sample_annotation.json → token, sample_token, instance_token,
    │                            translation, size, rotation, attribute_tokens
    ├── instance.json       → token, category_token
    ├── category.json       → token, name（human.pedestrian.* / vehicle.emergency.*）
    └── attribute.json      → token, name（pedestrian.moving / vehicle.moving）

nuScenes Map Expansion v1.3/（★ 新增下载项，Boston-only 仅 1 张地图）
└── maps/expansion/boston-seaport.json
    ├── stop_line           → polygon, stop_line_type ★冲突区几何
    ├── ped_crossing        → polygon              ★冲突区几何
    ├── traffic_light       → pose                 辅助 scenario_type 推断
    ├── road_segment        → polygon, is_intersection
    └── lane                → polygon              MINOR_ROAD 场景

DriveLM-nuScenes/（单一文件）
└── v1_0_train_nus.json
    ├── key_object_infos.*.Status  ★唯一信号灯相位来源（覆盖率待 W1 spike 确认）
    └── QA.{perception,prediction,planning,behavior}  narrative / keywords（英文）
```

**相比现有文档，实质性新增的只有两项**：
1. **map expansion**（一张地图文件，几十 MB，免费下载）——修复冲突区几何缺失（A3）
2. 明确 `sample_data.json` + `sensor.json` + `calibrated_sensor.json` 三张表——支撑密集位姿重建（A4）

其余都是将"用哪张表的哪个字段"从隐含假设落到实处。

---

### 2.7 W1 Spike 统计目标

在 W1 开发前，先对 Boston ∩ DriveLM 交集运行统计脚本，获取：

| 统计指标 | 用途 |
|---|---|
| Boston keyframe 总数 | 确认实验样本规模 |
| 含有效 traffic-element Status（red/green/yellow）的 keyframe 数及比例 | 决定信号灯场景是否可行 |
| 红灯 / 黄灯 / 绿灯各自数量 | 决定 RED_LIGHT / YELLOW_LIGHT 是否分别可行 |
| 含 `human.pedestrian.*` 且在 ped_crossing 内的 keyframe 数 | 决定 PEDESTRIAN 细分样本量 |
| 含 `vehicle.emergency.*` 的 keyframe 数 | 决定 EMERGENCY 细分是否保留 |
| 各场景下地图内有 stop_line / ped_crossing 的比例 | 验证冲突区几何可得性 |

根据上述数字，在 W1 末可做一次最终的范围确认（是否维持信号灯+让行双类，还是退为让行单类）。

---

*—— 文档结束，RegGround-AV 设计审查报告 v1.0 ——*
