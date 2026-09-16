# RegGround-AV 修订开发手册

**基于 2026-06-15 首次端到端运行（output_20260615_010902.json）的诊断与修复方案**

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 日期 | 2026-06 |
| 适用范围 | Layer 1 v3.1 / Layer 2 v3.0 / Layer 3 v3.0 实现修订 |
| 诊断依据 | mini 数据集 13 keyframes 运行日志 |

---

## 0. 诊断摘要

| 症状 | 根因 | 修复编号 |
|---|---|---|
| 墙钟 35 分钟，record 合计仅 12.7 分钟 | 约 22 分钟未计时区间 + 单场景均值 59s（超预算 6 倍） | D3 / E / G1 |
| 7/13 PARTIAL 降级 | 关键词污染召回错误硬规则（A）；narrative 噪音致超时（B）；退化样本无解（C） | A / B / C / D |
| 判定"不完美" | illegal 与 GT 特征逐位相同（C）；planning 答案泄露（B）；标签与事实不同源（新1） | C / B / 新1 |
| 8/13 场景标为 red_light 但无相位数据 | scenario_type 由关键词推断，与结构化事实脱节 | 新1 |
| chosen 中 GT 出现 0 次 | 裁判保守偏置 + 匿名置换疑似恒定 | 新2 / 新3 |

**修复间的关键耦合：修复 A（matcher）与新1（标签一致性）必须同批上线。** 只修 A 会导致伪 red_light 场景的 "vetoed" 标签失去对应规则支撑，benchmark 准确率反常下降。

---

## 1. 修复 A — Layer 2 ConditionMatcher：facts 优先

### 问题证据（record 0）

- `scene_facts.pedestrian_in_crosswalk = false`（几何计算正确：行人在 ego 后方）
- 但 `matched_conditions` 含 `C-PED-IN-CROSSWALK`，`retrieval_mode = "mixed"`
- 结果：硬规则 R-YLD-01（行人在横道必须停车）被错误注入，六条候选无一停车，裁判陷入矛盾 → 重试 → PARTIAL

现实现为"结构化 + 关键词取并集"，违反 Layer 2 v3.0 文档第 6.1 节匹配规则第 3 条（"若结构化事实为空，才走关键词兜底"）。

### 改动（condition_matcher.py，约 10 行）

```python
def match(self, query: SceneQuery) -> tuple[list[str], RetrievalMode]:
    structured = self._conditions_from_facts(query.scene_facts)

    if structured:                        # 有结构化命中即返回，关键词不参与
        return sorted(set(structured)), RetrievalMode.STRUCTURED

    if self.enable_keyword_fallback:
        keyword = self._conditions_from_keywords(query.keywords)
        if keyword:
            return sorted(set(keyword)), RetrievalMode.KEYWORD_FALLBACK

    return [], RetrievalMode.STRUCTURED
```

运行期不再产生 `MIXED`；枚举保留但加断言防止意外产出。

### 验收

- record 0 重跑仅召回 R-YLD-02、R-YLD-05
- 全部记录 `retrieval_mode ∈ {structured, keyword_fallback}`
- telemetry 新增 `keyword_fallback_rate`（目标 ≤10%，超出说明事实提取不足）

### 单测

1. facts 非空时 keywords 完全被忽略（即使 keywords 命中其他条件）
2. facts 全空且 fallback 开启时走关键词
3. 两者皆空返回空列表，不抛异常

---

## 2. 修复 B — Layer 1 narrative 白名单重建

### 问题证据（record 0）

- narrative 约 3000 字符，为 92 条 QA 答案的原样拼接
- 含大量无上下文噪音（"No. Yes. Yes. ..."）、相机坐标 token、结尾 "there is no s" 硬截断
- **含答案泄露**：planning QA 答案 "The action is to keep going at the same speed" 直接告诉裁判正确行为
- 该文本进入本场景每一次 LLM 调用（hard filter + 全部 pairwise + label gen），每次多耗 1000+ token

### 设计原则

从"黑名单删词"改为"白名单构建"：只从安全来源取材，planning / behavior 类 QA **整类不进入** narrative，问题从源头消失。禁词表（LeakageGuard）降级为最后一道保险丝，不再是主要防线。

### 改动（scene_text_extractor.py）

```python
def build_narrative(self, raw_scene: RawScene, facts: SceneFacts) -> str:
    parts = [raw_scene.drivelm_scene_description]   # DriveLM 一句话场景描述，纯感知

    # 结构化事实的中性转述，措辞自己控制，天然无泄露
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

    narrative = " ".join(parts)[:600]
    self.leakage_guard.assert_safe(narrative)       # 保险丝保留
    return narrative
```

### 验收

- 全部 narrative ≤ 600 字符，无截断残句、无相机 token
- 扫描 13 条重跑输出：不含 "keep going"、"should"、"the action is" 等 planning 措辞
- 每次 LLM 调用 prompt token 下降 ≥ 1000

---

## 3. 修复 C — 变体有效性：退化检查 + 前置门控 + 标签一致性

### 问题证据（record 0）

traj_e（ground_truth，expected cleared）与 traj_f（illegal，expected vetoed）特征到小数点后 13 位完全相同。原因：该场景 GT 本无减速（min 7.7 m/s），oncoming 违规扰动 `_remove_deceleration` 对无减速轨迹是空操作。任何裁判都无法对两段相同文本给出相反判定，此类样本系统性拉低准确率。

### C1 前置条件门控

生成违规变体前检查对应事实是否成立，事实为假不生成该类违规：

```python
VIOLATION_PRECONDITION = {
    ScenarioType.RED_LIGHT:    lambda f: f.has_red_light,
    ScenarioType.YELLOW_LIGHT: lambda f: f.has_yellow_light,
    ScenarioType.PEDESTRIAN:   lambda f: f.pedestrian_in_crosswalk,
    ScenarioType.ONCOMING:     lambda f: f.oncoming_vehicle_moving,
    ScenarioType.EMERGENCY:    lambda f: f.emergency_vehicle_active,
    ScenarioType.MINOR_ROAD:   lambda f: f.ego_on_minor_road is True,
}
```

### C2 退化检查

违规 / hard_case 变体生成后与 GT 特征比对，差异低于阈值则该样本 `expected_verdict="exclude"`（模型中已有该取值，评估脚本跳过 exclude）：

```python
def _is_degenerate(self, variant_feat, gt_feat) -> bool:
    return (
        abs(variant_feat.mean_speed_mps - gt_feat.mean_speed_mps) < 0.5
        and abs(variant_feat.min_speed_mps - gt_feat.min_speed_mps) < 0.5
        and abs(variant_feat.total_distance_m - gt_feat.total_distance_m) < 2.0
    )
```

同时对**所有**候选两两做"特征文本全等"检测，记录 tie 关系供修复 D2 使用（GT 与 suboptimal 也可能近似退化）。

### C3 标签一致性（与新1 联动，与修复 A 同批上线）

`expected_verdict="vetoed"` 当且仅当触发的 fact 会导致对应硬规则被 Layer 2 召回。实现：augmentor 与 matcher 复用同一张 `FACT_TO_CONDITION` 映射，生成标签时调用纯函数：

```python
def would_retrieve_hard_rule(facts: SceneFacts, scenario_type: ScenarioType) -> bool:
    """标签生成与检索同源：事实不成立则不得生成 vetoed 标签。"""
```

### 验收

- benchmark 中不存在"vetoed 标签但对应 fact=false"的样本
- 不存在与 GT 特征全等且 verdict≠exclude 的违规变体
- 评估脚本正确跳过 exclude 样本并单独报告 exclude 数量

---

## 4. 修复 D — Layer 3 降级治理与单场景提速（不含并发）

### D1 准入过滤 + 空子图短路

编排层在 retrieve 之前过滤：`scenario_type == UNKNOWN` 或结构化事实全 False 的 keyframe 直接跳过，记录 `skipped_by_admission` 及原因，不进裁判。若通过准入但子图仍为空（理论不应发生），输出带原因的 FAILED 标签而不调 LLM。

### D2 相同特征对跳过 + 确定性平局

pairwise 前对 CLEARED 集合按特征摘要哈希分组：组内强制并列（不调 LLM），组间才比较；聚合器平局按 traj_id 字典序打破（确定性）。telemetry 记录 `tied_pairs_count`。record 0 中 traj_e vs traj_f 这类比较问了也是白问，跳过既省钱也保护 Kendall's tau 稳定性指标不被硬币噪音污染。

### D3 超时与重试参数

- `llm_timeout_seconds`: 20 → 30（narrative 瘦身后可先留 20 观察超时率再定）
- `llm_max_retries`: 2 → 1（一次超时烧 40–60s 是 PARTIAL 的直接推手）
- 重试加抖动退避：首次等 1–2s，第二次 4–8s（并发化后同样复用）

### D4 LeakageGuard 扫描范围修正（spec 修订项）

Layer 3 v3.0 要求禁词扫描覆盖"完整 system + user prompt"，但法规文本必然含"必须 / 禁止 / violate"，按字面实现会大面积误报。修订为：

- **运行时扫描对象** = Layer 1 产出文本（narrative + 每条 feature summary）
- **prompt 模板与法规文本** 不做运行时禁词扫描，改为模板单元测试静态检查（模板本身不得含标签词）
- 同步修订 Layer 1 / Layer 3 文档中"100% 覆盖完整 prompt"的表述

### 验收

- 重跑 mini：PARTIAL ≤ 2，无 FAILED
- `citation_validity_retries` 无重试风暴
- 泄露扫描不再对 rules_text 报警

---

## 5. 修复 E — 并发化

瓶颈 95% 在等待 LLM 返回，IO-bound，用 asyncio 而非多进程（避免 nuScenes 表重复加载）。

### 两层并发

1. **场景内**：pairwise 比较相互独立，`asyncio.gather` 一波发出，直接压单场景延迟
2. **场景间**：多 keyframe 并行，提吞吐

### 限流设计

不做两层嵌套限流，**全局一个 Semaphore 直接包在 LLM 调用函数上**（真正稀缺的是在飞请求数）。从全局 5 路起步，观察 429 与超时率，逐步加到 16。

### 骨架

```python
async def process_one(bundle, retriever, judge):
    subgraph = retriever.retrieve(bundle.scene_query)   # 同步 <10ms
    label = await judge.judge_async(bundle.judge_input, subgraph)
    return bundle, label

async def run_all(bundles):
    tasks = [process_one(b, retriever, judge) for b in bundles]
    with open(out_path, "a") as f:
        for coro in asyncio.as_completed(tasks):
            try:
                bundle, label = await coro
                f.write(json.dumps(serialize(bundle, label), ensure_ascii=False) + "\n")
            except Exception as e:
                log_scene_failure(e)          # 单场景失败不拖垮整批
```

要点：

- 输出改 **JSONL 逐行追加**：中途崩溃已完成结果不丢，重跑按 frame_token 跳过已有记录（免费断点续跑）
- 保留 `--serial` 开关：mini 冒烟测试串行跑，日志可读
- 确定性不受影响：单场景输入输出关系与串行一致，完成顺序打乱后按 frame_token 排序即可
- telemetry 区分：单场景 `total_duration_ms` 语义不变（论文 P95 用它），整批墙钟时间单独记录；新增 429 计数

### 验收

- 修复 A–D 后单场景串行 15–25s；叠 10 路并发后 200 keyframes 一轮 ≤ 20 分钟
- 人为中断后重跑可续，无重复记录

---

## 6. 修复 F — 数据迁移与评估集锁定

### 关键事实：不需要下载完整 nuScenes

管线只读元数据 JSON 表 + 地图 + DriveLM 文本，不读图像与点云：

| 需要 | 大小 | 状态 |
|---|---|---|
| `v1.0-trainval_meta.tgz`（仅 JSON 表） | ~0.4 GB | **唯一需下载项** |
| Map Expansion `boston-seaport.json` | 已有 | 不分 mini/trainval，直接复用 |
| DriveLM `v1_0_train_nus.json` | 已有 | 复用 |

改动：`nuscenes_version` 从 `v1.0-mini` 切到 `v1.0-trainval`。建议在 loader 加断言：整个 run 未打开任何 `samples/`、`sweeps/` 传感器文件——既防呆也是论文"轻量元数据管线"卖点。

### 流程

1. 下载 meta 包，切版本
2. 跑 `availability_spike.py`（几分钟），得到 Boston ∩ DriveLM 各细分真实样本数
3. 按 spec 规则：细分有效样本 < 20 → 砍出 MVP 范围
4. **分层抽样锁定评估集**：每细分 40–50 个，总计 150–300 keyframes，`seed=42` 抽取后将 token 列表存文件并纳入版本管理，此后主实验 / 2×2 消融 / 人工标注全部使用同一集合
5. mini 13 条保留为冒烟测试：每次改代码先跑 mini 验证管线，再上评估集

### 成本估算

修复后单场景 15–25s × 200 keyframes ÷ 10 并发 ≈ 10–15 分钟/轮；消融 4 条件 1 小时内；DeepSeek 调用成本可忽略。

---

## 7. 观测、可复现与日志

### G1 计时补洞（定位 22 分钟黑洞）

编排层记录相邻 record 之间的 gap 时间。怀疑方向：逐场景重复初始化 nuScenes/map、重试 sleep 未计入 record、巨型 JSON 序列化。先测量再优化。

### G2 Run manifest

每个输出文件头写入：git commit SHA、`graph_version`、模型名、seed、评估集文件 hash、prompt 模板 hash。论文可复现性的硬要求，成本极低。

### G3 日志瘦身

当前输出把同一 narrative / context 重复内嵌 3–4 次，200 场景会产出巨型文件。默认只存引用（token）与判定结果；`archive_llm_io=True` 时才存 prompt 原文。输出格式改 JSONL（与修复 E 联动）。

### G4 新增 telemetry

| 字段 | 用途 |
|---|---|
| `keyword_fallback_rate` | 修复 A 后监控事实提取充分性 |
| `skipped_by_admission` + 原因 | 准入过滤透明化 |
| `tied_pairs_count` | 平局规模，保护稳定性指标解释 |
| `gt_chosen_rate` / `gt_top2_rate` | 评估侧 join benchmark labels 计算，监控保守偏置（新3） |
| `decided_by_rule_engine / llm` 占比 | spec 已有，确认实现落地 |
| 整批墙钟 + 429 计数 | 并发调参 |

---

## 8. 实施顺序与工作量

| 优先级 | 项目 | 预估 | 说明 |
|---|---|---|---|
| **P0（重跑前必做，同批上线）** | 修复 A matcher | 0.5 天 | 含 3 条单测 |
| P0 | 修复 B narrative | 0.5 天 | 含泄露回归测试 |
| P0 | 修复 C（C1+C2+C3） | 1 天 | 与 A 强耦合 |
| P0 | 新2 匿名化 seed 修正 | 0.5 小时 | 一行改动 + 核查 |
| P0 | 修复 D1/D2/D3/D4 | 1 天 | 降级治理 |
| **P1** | 修复 E 并发化 | 0.5–1 天 | judge 改 async |
| P1 | G1–G4 观测项 | 0.5 天 | |
| **P2** | 修复 F 数据迁移 + 评估集 | 0.5 天 + 下载 | spike 先行 |
| P2 | 新4 特征词表扩展 | 1 天 | 唯一较大新开发 |

P0 合计约 3 天，完成后重跑 mini 验证，再进 P1/P2。对应原里程碑 W3–W6 区间，不需要调整总体计划。

---

## 9. 重跑 mini 的预期与验收 Checklist

修复 P0 后重跑 13 条，预期变化（部分场景被跳过是**正确行为**，mini 本就没有相位数据）：

- [ ] 8 条伪 red_light 被准入过滤跳过或转 UNKNOWN，`skipped_by_admission` 有原因记录
- [ ] PARTIAL ≤ 2，无 FAILED
- [ ] 全部 `retrieval_mode ∈ {structured, keyword_fallback}`，无 mixed
- [ ] 不存在 GT 与 illegal 特征全等且双双进入评测的样本对
- [ ] narrative 全部 ≤ 600 字符且无 planning 措辞
- [ ] 单场景串行耗时 ≤ 25s
- [ ] 匿名置换跨场景不同（traj_e 不再恒为 GT）
- [ ] 输出为 JSONL，含 run manifest 头

---

## 10. 新识别的遗漏项（本次评审新增，未在此前讨论中展开）

### 新1 标签–事实–检索一致性闭环（与修复 A 强耦合，最高优先）

**证据**：8/13 场景 scenario_type=red_light 但相位来源为 none → 分类来自关键词。
**连锁**：augmentor 按伪分类生成"红灯违规"变体并标 vetoed；修完 matcher 后 Layer 2 不召回 R-SIG-01；裁判无规则可依、判 CLEARED；benchmark 记为"判错"。**系统是对的，标签是错的。**
**原则**：scenario_type 推断 facts 优先（与 matcher 同一原则）；关键词推断只能产出 UNKNOWN-with-hint 并被准入过滤跳过；expected_verdict 由 `would_retrieve_hard_rule()` 与检索同源推导（见 C3）。
**上线要求**：与修复 A 同一批合入，禁止单独上线 A。

### 新2 匿名化置换疑似全数据集恒定

**证据**：record 0 中 traj_e=GT；13 条 chosen 直方图 traj_e 出现 0 次。若 `CandidateAnonymizer` 每场景重建 `Random(42)`，置换恒定，traj_e 永远是 GT。
**核查**（2 分钟）：grep 13 条记录中 traj_e 的 `original_internal_id`，若全为 `planning_gt` 即确认。
**修法**：`rng = random.Random(f"{seed}:{frame_token}")`——跨场景不同、单场景可复现。
**影响**：不修则 position bias 指标失效、跨场景聚合统计被混杂。

### 新3 保守偏置：GT 从未被选为 chosen

**证据**：chosen 直方图 traj_b（conservative）4 次居首；空旷道路 5 m/s 爬行胜过人类司机 8.4 m/s。
**风险**：RLAIF 下游用这批偏好会训出过度谨慎的规划器。
**缓解**：pairwise prompt 判据加入"无规则区分时考虑通行效率与交通流畅"；评估侧新增 `gt_chosen_rate`（依赖新2 修复后才可跨场景解释）。
**论文角度**：这是"LLM 裁判价值偏置"的一个可量化发现，值得写进讨论章节。

### 新4 特征词表不足以表达法律判断 + 事实精度两处核查

当前特征只有速度统计与模板化的"后半段…"描述，缺少判定真正依赖的量：

```python
# TrajectoryFeatures 建议新增字段
entered_conflict_zone: bool          # 是否进入冲突区多边形
min_distance_to_conflict_m: float    # 全程距最近冲突区的最小距离
full_stop: bool                      # 是否完全停车（v<0.3 m/s 持续≥1s）
stop_duration_s: float               # 停车时长
min_speed_in_zone_mps: float | None  # 冲突区内最低速度
stopped_before_zone: bool            # 是否在进入冲突区前停车
```

**红利**：补齐布尔量后 RuleEngine 才能按 spec 裁定 easy 案例（"未停车且进入横道"可确定性 veto），`decided_by=rule_engine` 占比上升，LLM 调用进一步减少——这同时是准确率和速度的双重收益。

**核查两处**：

1. 六条轨迹的冲突区描述措辞完全统一（"后半段…"），怀疑分析器用"轨迹后半段"近似而非真实多边形相交测试，需确认实现。
2. `C-ONCOMING-STRAIGHT` 的 YAML 定义是"**转向路径**与对向车冲突"，但事实提取只查了朝向夹角>150°+moving——ego 直行时（如 record 0 的停车场直行）R-YLD-02 根本不该适用。修法：从 GT 轨迹的净航向变化（>20° 视为转向）推导 ego 转向意图，与对向夹角合取后才置 `oncoming_vehicle_moving`。

### 新5 平局与排序稳定性保护

已并入修复 D2。动机记录：同文候选的 pairwise 结果是硬币噪音，会污染 Kendall's tau 稳定性指标，使"排序不稳定"被误归因于裁判质量。强制并列 + 确定性 tie-break 后，稳定性指标才可解释。

### 新6 配置漂移与结构化输出

实际运行 `llm_provider="openai_compatible"` + deepseek-v4-pro，与 Layer 3 spec 的 `Literal["openai", "anthropic"]` 不符。扩展枚举、写入 manifest（G2）。同时确认 DeepSeek 调用启用了 JSON mode / response_format——结构化输出解析失败是重试的另一来源。

---

## 附录：spec 文档同步清单

| 文档 | 需同步的修订 |
|---|---|
| Layer 1 v3.1 → v3.2 | narrative 白名单构建（§6.2）；变体门控与 exclude 语义（§6.5）；匿名化 per-scene seed（§6.6）；TrajectoryFeatures 字段扩展（新4） |
| Layer 2 v3.0 → v3.1 | matcher 实现须严格遵循既有 §6.1 语义（文档本身正确，加实现警示）；MIXED 运行期不再产生的说明 |
| Layer 3 v3.0 → v3.1 | LeakageGuard 扫描范围（D4）；准入短路语义（D1）；超时/重试参数（D3）；平局处理（D2）；provider 枚举扩展（新6） |
| SRS / 架构 v2.0 | 无结构性变化；更新"泄露扫描 100% 覆盖完整 prompt"的措辞为"覆盖 Layer 1 产出文本 + 模板静态检查" |

---

*文档结束 — RegGround-AV 修订开发手册 v1.0*
