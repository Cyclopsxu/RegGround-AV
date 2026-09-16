# RegGround-AV 修改补丁 2.1 · 2026-07-12

> 依据：`output_patch_full_deepseek_2026071102.jsonl` 的验收结果、
> `RegGround_修改补丁2_20260711.md` 的评审，以及现有代码行为核对。
>
> 目标：消除剩余的数字歧义，焊死 uncertain 安全边界，完成真正盲法的几何谓词审计，
> 并把 trainval 的统计与抽样单位从 keyframe 修正为独立场景。

---

## 0. 范围与设计决定

### 本补丁不做

| 不做 | 原因 |
|---|---|
| 性能优化 | 当前速度可接受，不改变既定范围 |
| 相同特征 pairwise 短路 | 仍调用 LLM；本轮只验证 tied 计数与 partial 原因 |
| 全面改造报告状态 | `complete/partial/failed` 继续只描述报告流程是否完整 |
| 自动阻止 dirty tree 调试运行 | 日常调试允许 dirty；正式 trainval run 由人工检查 clean commit |

### 关键语义决定

1. `ReportStatus` 与“是否有候选可选”分离：新增 `selection_outcome`，不把
   `no_choosable_candidate` 塞进 `complete/partial/failed`。
2. 规则层不报告 vs benchmark accuracy；其外部效标只能是人工谓词一致性。
3. 几何审计必须先盲判、后 join 谓词 sidecar；旧的带谓词标题图片不能用于盲审。
4. 正式实验的统计与抽样单位是独立场景，不是同一停车事件中的多个 keyframe。

---

## 1. 补丁 A · 清理循环指标与空分母语义（P0）

### 改动

1. 删除 summary 顶层 `candidate_verdict_accuracy` 以及只为该字段服务的累计变量。
2. 保留 `benchmark_evaluation.verdict_layers` 作为唯一 verdict 分层入口。
3. LLM 层 `evaluated=0` 时：
   - `benchmark_accuracy=null`；
   - 增加 `accuracy_status="not_evaluable"`；
   - 不再用 `0.0` 表达“没有样本”。
4. 规则层继续输出：
   - `benchmark_accuracy_reported=false`；
   - `required_external_metric="predicate_human_agreement"`。

### 验收

- summary 中不存在任何规则层 vs benchmark 的准确率数字。
- LLM 分母为 0 时 accuracy 为 `null`，而不是 `0.0`。
- 旧日志评分回归仍得到原有口径数字；新日志得到 illegal-chosen `0/2`。

---

## 2. 补丁 B · uncertain 选择政策与模型不变量（P0）

> 当前实现已经做到：chosen 只从 cleared 产生、排序为 cleared → uncertain → vetoed、
> uncertain 不进入 preference pair。本补丁不重写算法，只把现有行为升格为模型约束和回归测试。

### B.1 新增独立选择结果

新增枚举或等价的计算字段：

```text
selection_outcome:
- selected                  # 存在 cleared，且 chosen 指向 cleared
- no_choosable_candidate    # 有裁决结果，但 cleared=0，chosen=null
- unavailable               # 流程失败或没有有效 verdict
```

`ReportStatus` 保持原语义。允许以下合法组合：

```text
status=complete
selection_outcome=no_choosable_candidate
chosen_trajectory_id=null
```

### B.2 焊死不变量

1. `cleared_count > 0` 时，chosen 必须存在且指向 cleared。
2. `cleared_count == 0` 时，chosen 必须为 `None`。
3. `preference_ranking` 分层固定为 cleared → uncertain → vetoed。
4. uncertain 不得出现在任何 preference pair 中。
5. 非降级运行中：
   - pairwise judgment 数 = `C(cleared, 2)`；
   - compliance pair 数 = `cleared × vetoed`；
   - preference pair 总数为两者之和。
6. summary 增加 `selection_outcomes` 计数。

### B.3 回归测试

- cleared=0、全部 uncertain：无 chosen，`selection_outcome=no_choosable_candidate`。
- cleared=0、全部 vetoed：无 chosen，但报告可以是 complete。
- cleared/uncertain/vetoed 混合：检查排序分层、chosen 和 pair 计数。
- 人为构造“有 cleared 但 chosen=None”与“chosen 指向 uncertain”：模型必须拒绝。
- 孪生轨迹测试继续调用 Fake LLM；Fake LLM 明确返回：
  - `confidence <= 0.5`；
  - reasoning 含“无法区分”；
  - 断言 `tied_pairs_count=1` 且 partial reason 含 `low_confidence_pair`。

---

## 3. 补丁 C · 真正盲法的谓词可视化与人工审计（P0）

### C.1 先重做 records 6/8 的 12 张盲图

现有图标题包含 expected 和谓词输出，不能继续作为盲审材料。本轮不是简单追加 6 张，
而是把 records 6/8 的全部 12 条候选重新生成：

- ground_truth
- conservative
- suboptimal
- aggressive
- hard_case
- illegal

重点样本：record 6 的 traj_f（suboptimal）。它是本轮唯一由谓词弃权并升级给 LLM 的候选，
需要人工回答：

1. 人可以明确判 cleared/vetoed，但谓词选择了保守弃权；
2. 还是人也无法判断，说明 uncertain 路由正确；
3. 或者人认为应当 cleared，说明谓词判定域可能过窄。

### C.2 图片规范

每条候选单独一张 PNG，图片只包含几何证据：

- 轨迹分段按速度着色，并显示带 `m/s` 单位的 colorbar；
- 停止线代理多边形；若还有独立真实停止线几何，使用不同样式区分；
- ego 起点、朝向箭头、终点；
- 等比例坐标轴、米制刻度或比例尺；
- 仅使用随机 `audit_id`，不显示 variant、expected、谓词值或裁判结果。

### C.3 文件隔离

```text
artifacts/predicate_audit_v2_1/
  blind_images/
    audit_001.png
    ...
  blind_answers.csv
  predicate_sidecar.jsonl
  joined_audit.csv          # 全部盲判结束后才生成
  audit_report.md
```

`blind_answers.csv` 只保存 `audit_id` 与人工答案；`predicate_sidecar.jsonl` 保存真实
candidate_id、variant、谓词值、当前 expected verdict 和 predicate version。盲判期间不读取 sidecar。

### C.4 固定问题

1. 轨迹是否进入图示停止线代理多边形？——是 / 否 / 图上无法判断
2. 在进入代理多边形前是否完整停车？——是 / 否 / NA / 图上无法判断
3. 在已知 R-SIG-01 生效的前提下，应判？——cleared / vetoed / uncertain
4. 停止线代理多边形与可见真实停止线是否一致？——一致 / 偏移 / 无法判断

第 1/2 问审计几何谓词；第 3 问审计最终直裁；第 4 问单独审计代理有效性，
避免把“进入代理区”等同于“法律语义上已经越线”。

### C.5 分层抽样

审计范围限定为谓词适用的红灯场景：

| 层 | 规则 |
|---|---|
| 必进 | record 6 traj_b、record 6 traj_f、全部 uncertain、全部 hard_case |
| 必进 | 红灯场景中全部 `is_degenerate=true`，包括被 stationary admission 跳过的样本 |
| 必进 | 全部被 RuleEngine vetoed 的候选 |
| 随机 | 常规 cleared 补足至 30–50 条 |

UNKNOWN 场景中的 degenerate 不进入 R-SIG-01 谓词审计。审计工具必须支持读取红灯但被
stationary admission 跳过的候选，不能沿用“只画 admitted 场景”的过滤条件。

### C.6 指标与 gate

- verdict 级：`predicate_human_agreement`；
- 字段级：human vs `entered_conflict_zone`、`stopped_before_zone`；
- 代理级：停止线代理一致 / 偏移 / 无法判断分布；
- 每条不一致归因为：谓词 bug / 代理偏移 / 本质模糊；
- 工程 gate：可判定样本一致率 ≥95%，所有 vetoed 分歧必须有 root cause；
- 该数字只作为单人工程验证，不冒充论文外部效标；正式论文仍由 2–3 人审计并报告 IAA。

发现谓词或代理问题时：修复 → bump `predicate_version` → 重推导 benchmark → 复审受影响样本。

### C.7 旧审计记录归档说明

在原 `geometry_audit_record.md` 顶部增加说明，不悄悄覆盖历史列：

> 本表生成于 benchmark v2 几何重建前；`expected` 列为当时的历史标签，不代表当前谓词
> 推导结果。当前版本中 record 6 traj_b 已改判为 vetoed。该表用于记录第一次代理几何检查，
> 不作为 v2.1 盲审效标。

新 joined 表分别保留：`human_verdict`、`predicate_verdict`、
`benchmark_expected_verdict`，避免三种含义再次混淆。

---

## 4. 补丁 D · spike 与正式抽样改为独立场景口径（P0）

### 问题

mini 的 5 个红灯 keyframe 全部来自 scene-0757 的同一次停车过程，因此：

- keyframe 数 = 5；
- 独立场景数 = 1；
- 两个通过准入的 keyframe 也不能当作 n=2 的独立样本。

### 改动

spike 同时输出 keyframe 和独立场景两套数字：

- `total_keyframes` / `unique_scenes`；
- `admitted_keyframes` / `admitted_unique_scenes`；
- 各 scenario 的 keyframe 数与 unique scene 数；
- 每个 scene 的 admitted keyframe 数分布；
- stationary 比例同时按 keyframe 和 scene 报告。

正式候选集按 `(scene_token, scenario_type)` 分组；每组只选一个已准入 frame，使用 seed=42
对 frame token 做稳定抽样，不人工挑“最好看”的帧，也不依据裁判结果选帧。若未来能可靠切分同一
scene 内多个独立事件，再显式加入 event_id；在此之前采用更保守的场景级口径。

### 验收

- mini spike 明确报告 red-light unique scene = 1。
- 固定 seed 重跑得到完全相同的代表 frame token 列表。
- 正式实验的 `n`、抽样清单和论文表格均按 unique scene 口径。
- 若 trainval 不足 150 个独立适用场景，缩小实验声明，不用相邻 keyframe 补数量。

---

## 5. 补丁 E · 三项低成本语义修正（P1）

### E.1 GT 预检状态

当前 UNKNOWN 场景因谓词不适用而被写为 `gt_precheck_status=failed`，语义不准确。扩展为：

```text
passed | failed | not_evaluable | not_gt
```

- 谓词明确 cleared → passed；
- 谓词明确 vetoed → failed；
- 场景/几何不在谓词判定域 → not_evaluable；
- 非 GT → not_gt。

### E.2 telemetry 总计

- `llm_telemetry.total` 始终显式包含 calls/input_tokens/output_tokens/retries；
- retries 为零时也保留 `retries: 0`；
- 未配置价格时输出 `cost_status="pricing_not_configured"`，不伪造成本；
- 配置可信价格后才输出 `estimated_cost_usd`。

### E.3 manifest

manifest 直接加入字符串 `predicate_version`，同时继续保留 predicate 源文件 SHA-256。
正式 trainval run 启动前人工确认：clean commit、审计 gate 通过、token 清单已版本化。

---

## 6. 执行顺序

| 序 | 事项 | 验收检查点 |
|---:|---|---|
| 1 | 补丁 A + B + E | 单元测试、旧/新日志评分回归、模型不变量 |
| 2 | 重写盲审图片与 sidecar | records 6/8 共 12 张无泄露图片 |
| 3 | 优先盲判 record 6 traj_f，再完成 mini 审计 | joined 表、逐条归因、工程 gate |
| 4 | trainval metadata 到位后运行场景级 spike | keyframe/scene 双口径报告 |
| 5 | 评审 spike，固化独立场景 token 列表 | seed=42 可复现 |
| 6 | clean commit 上运行正式评估与多人外部审计 | illegal-chosen、分层指标、IAA |

---

## 7. 完成定义

- 不存在规则层 vs 同源 benchmark 的 accuracy 数字；零分母不再显示为 0%。
- uncertain 永不成为 chosen；无 cleared 时产生明确的 `selection_outcome`。
- tied、排序分层、pair 数量均有回归测试。
- records 6/8 的 12 条候选全部完成无泄露盲图；record 6 traj_f 有人工裁决与路由结论。
- 旧审计表明确标注历史 expected，不与当前谓词标签混用。
- 谓词人工审计有字段级、verdict 级和代理级结果，并对所有 veto 分歧完成归因。
- spike 与正式实验同时报告 keyframe 数和 unique scene 数，论文主 `n` 使用独立场景。
- GT 不适用不再记为 failed；telemetry 零重试显式可见；manifest 可直接定位 predicate version。
