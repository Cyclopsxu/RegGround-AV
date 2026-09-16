# RegGround-AV 修改补丁 · 2026-07-11

> 依据:`output_postfix_full_deepseek_20260711.jsonl`(mini,13 场景,git 8aa6ad7,graph 2026-06-v3.0,seed 42)的逐条评审
> 原则:**先修"数字是否有意义",再谈别的**。依赖方向:口径 → 插桩 → 数据侧 → 判定侧 → 正式评估;谓词先审计后使用。

---

## 0. 范围声明:本补丁不做什么

| 不做 | 理由 |
|---|---|
| 性能优化(场景内并行、全局并发) | 当前 56–80s/场景可接受,搁置;将来要做时首选场景内并行,不是全局并发 |
| 平局**短路**(省 pairwise 调用) | 属性能项,随上条搁置;但 tied **计数一致性**保留在补丁 2(那是正确性不是性能) |
| narrative 收紧(~600 字符 → 更短) | 无泄露、非延迟主因,降级为卫生项,顺手做不立项 |

---

## 1. 执行序总览

| 序 | 补丁 | 产出 | 依赖 |
|---|---|---|---|
| 1 | 指标口径与分层报告定义 | 评分脚本改版 + 口径文档 | 无 |
| 2 | partial_reasons + telemetry + tied 一致性 | 插桩,之后每次重跑均为诊断 | 无 |
| ∥ | trainval 可用性 spike(**并行线,现在起跑**) | 分类计数表,gate 补丁 6 | 无 |
| 3 | 几何谓词人工审计 | 审计记录 + record 6 traj_b 裁决 | 无 |
| 4 | 数据侧合并重建(去重/静止准入/谓词打标/GT 预检) | 新 benchmark → **mini 重跑 A** | 3 |
| 5 | 已审计谓词接入 RuleEngine | → **mini 重跑 B**(分层报告已就位) | 1, 3, 4 |
| 6 | trainval 正式评估 + 人工外部审计 | 论文主实验数据 | 1–5 + spike |

---

## 2. 逐项细则

### 补丁 1 · 指标口径与分层报告定义(纯分析侧,半天)

**改动点**
1. 主安全指标改为 **illegal-chosen rate**(选中候选的 expected_verdict ∈ {vetoed} 的场景占比,目标 0)。本轮值:0/5 ✅——这个数字目前没被报告,它才是头条。
2. 选中 hard_case / exclude 候选的场景记为 **unscorable**,单列报告,不再计入 gt_chosen 的失败分母。本轮:3/5 unscorable。
3. `gt_chosen_rate` / `gt_top2_rate` 降为参考指标,并注明:合规候选间"最优"不唯一,GT 非唯一正解;红灯场景区分度天然低(正确行为就是停),真正有区分度的结论等让行场景。
4. **分层报告定义**(为补丁 5 预埋):verdict 指标拆两层——
   - 规则引擎判定子集:报"谓词 vs 人工审计一致性"(外部效标是人,不是 benchmark——benchmark 同源,不能自证);
   - LLM 判定子集(模糊 case):报 vs benchmark 准确率,**这才是对"法规接地裁判"的真评估**。
5. hard_case 保留在候选池(系统本就不可见 debug 标签),只改记分方式。

**验收断言**
- 对本轮 jsonl 重跑评分脚本,输出:illegal-chosen 0/5、scorable 2/5(全部选中 compliant)、unscorable 3/5、verdict 20/22(全 LLM 层)。数字与人工核对一致。
- 口径写成一页文档,进版本管理(后续论文方法节直接引用)。

---

### 补丁 2 · partial_reasons + telemetry + tied 一致性(插桩先行,半天~一天)

**改动点**
1. `final_label` 增 `partial_reasons: list[enum]`,枚举至少含:`uncertain_verdict` / `low_confidence_pair` / `citation_repair` / `stage_degraded`。status 由该列表推导,不再独立赋值——**status 与计数器永不说两套话**。
2. `tied_pairs_count`:confidence ≤ 0.5 且理由为"无法区分"的 pair 计入 tie(即修 record 11 计数为 0 的不一致)。
3. telemetry 补齐(逐阶段):LLM 调用数、输入/输出 token(修 `llm_tokens_used=0`)、重试次数、引用校验失败明细(原始 vs 回退引用集)、特征文本哈希。
4. 引用校验若发生"修复/回退",必须留痕:修了什么、从什么修成什么。

**验收断言**
- mini 重跑:record 10 报 `["uncertain_verdict"]`;record 11 报出**机器可读的真实原因**(裁决"引用回退"vs"低置信 pair"两个假说——不猜,让数据说)。
- 任意 record 的 token 数 > 0 且逐阶段可加总;summary 含全批 token/成本合计。

---

### 补丁 ∥ · trainval 可用性 spike(并行线,即刻可做)

> 这是你马上要写的"筛选拼接脚本"。三条铁则:**只下 metadata、复用管线分类逻辑、多数一个静止维度。**

**脚本规格**
- 输入:nuScenes **v1.0-trainval metadata 包**(不下相机/雷达数据)+ DriveLM-nuScenes 标注文件。
- 拼接:trainval 场景 ∩ DriveLM 覆盖 ∩ location 过滤(与现行 boston-seaport 口径一致;顺手统计放开 location 后的量,供 fallback 决策)。
- 分类:**直接 import 管线的 context_building / scene_facts / scenario_type 判定代码**,只跑到准入判断为止,不进 LLM。禁止新写一套过滤规则——spike 的意义是"预测正式跑的准入结果",逻辑不同源则预测无效。
- 统计输出(一张表):
  - 各 `scenario_type` 计数:red_light / yellow / pedestrian_crosswalk / oncoming / emergency / unknown;
  - `traffic_light_status_source` 分布(none 占比 = 信号灯信息缺失率);
  - **静止 ego 占比**(ego 总位移 < 阈值,按补丁 4 的准入阈值预演)——红灯场景刨掉静止后剩多少,直接决定 150–300 keyframes 的抽样是否成立;
  - 每 keyframe 的 QA/key_object 计数分布(评估上下文丰度参考)。

**验收断言**
- 产出 `spike_report.md`:分层计数表 + "按当前准入,预计可评估 N 场景 / 类"。
- 决策输出:N 足够 → 按补丁 6 分层抽样;不够 → 触发 fallback 讨论(放开 location / 调准入 / 收窄场景声明)。

---

### 补丁 3 · 几何谓词人工审计(小样本肉眼,半天)

**改动点**
1. 审计对象:`entered_conflict_zone` / `stopped_before_zone` / `full_stop` / `min_distance_to_conflict` 这组"冲突区代理停止线"谓词。
2. 方法:抽 10–20 条候选(必含 record 6 traj_b),画轨迹 × 冲突区/停止线叠加图,人工判定谓词输出是否符合几何事实。
3. **裁决 record 6 traj_b**:aggressive/期望 cleared,但特征显示入区且区内停车——判定是(a)扰动器造出了实际违规的"aggressive"(标签错,补丁 4 修),还是(b)冲突区代理偏移(谓词错,先修谓词)。

**验收断言**
- 审计记录:每条样本一行(谓词输出 / 人工判定 / 一致与否),一致率量化;traj_b 有明确归因。
- **谓词未过审计,补丁 4/5 不得使用它**——这是硬 gate。

---

### 补丁 4 · 数据侧合并重建(一次重建,mini 重跑 A)

> 原"P0-2 病根 + P0-5 + P0-6"三合一。**两个改动都合入后只重建一次 benchmark。**

**改动点**
1. **意图/结果分离**:`variant_type` 只记"扰动想造什么";`expected_verdict` 一律由已审计的几何谓词对最终轨迹**验证推导**,不再由扰动意图指定。
2. **GT 预检**:真实轨迹同样过谓词;违反触发规则的 GT 标 `exclude`(或 vetoed),从源头消灭"GT 默认 cleared"的标签噪声。
3. **退化修复**:扰动后按**舍入特征向量**去重(与 pairwise prompt 可见文本同一舍入精度),特征全同的变体重造或标 `degenerate`;准入增加"ego 最小位移/最小速度"阈值(record 11 型静止场景零信息,早筛)。
4. 重建 benchmark(bump 数据版本号),记录新 `input_sha256`。

**验收断言(mini 重跑 A,验数据侧)**
- 零重复变体(舍入特征哈希无碰撞,除非显式标 degenerate);
- 每条候选的 expected_verdict 有谓词推导记录可查;
- record 6 traj_b 按补丁 3 裁决结果被正确归类;
- 与上轮 jsonl 做 manifest diff:仅 `input_sha256` 变,指标变化全部可归因到数据侧。

---

### 补丁 5 · 谓词接入 RuleEngine(mini 重跑 B)

**改动点**
1. RuleEngine 消费同一套已审计谓词(**与补丁 4 的 benchmark 打标共用同一份代码**,写一次三用:预检 / 生产判定 / 回归测试)。
2. 明确情形直接 veto/clear(`decided_by=rule_engine`, confidence 1.0);谓词不足或模糊时**降级给 LLM**(`decided_by=llm`)——分层结构保留,不是全盘替换。
3. `uncertain` 弃权路径保留原语义(record 10 是 fail-safe 正确工作的证据,不得被"优化"掉)。

**验收断言(mini 重跑 B,验判定侧)**
- `decided_by_rule_engine > 0`;明确 case(如 record 6/8 的 illegal)由规则层判定;
- **分层报告生效**:规则子集不再报 vs-benchmark accuracy(同源自证),只报谓词-人工一致性;LLM 子集单独报;
- 回归测试落地:红灯规则直裁 / GT 预检 / 舍入同特征即 tie / partial 原因可读 / 保守候选不因"停更久"自动获胜 / degenerate 场景被拒 / summary 记账不变量(如 pairs = C(cleared,2) + cleared×vetoed)。

---

### 补丁 6 · trainval 正式评估 + 人工外部审计

**改动点**
1. 按 spike 结果分层抽样 **150–300 keyframes**,`seed=42` 固化 token 列表并进版本管理;主实验、消融、人工标注共用同一集合。
2. 人工外部审计:重点抽 **LLM 判定子集**(它才是被评估的主体)+ 规则子集小样本抽检;2–3 人统一规则,报 IAA。
3. 验收指标修订(替换原"PARTIAL ≤2 / 无 FAILED / P95 ≤25s"):
   - 每个 partial 有机器可读原因(枚举齐全率 100%);
   - **uncertain 率单独报告、不设"越低越好"目标**——弃权是 fail-safe,不是缺陷;
   - 无 FAILED、无标签-事实矛盾(回归测试兜底);
   - illegal-chosen = 0;unscorable 率、规则/LLM 判定占比、exclude 样本数及原因,全部进论文局限性一节;
   - 速度指标本轮**不设验收**(范围声明 §0)。

---

## 3. 重跑检查点与归因纪律

| 检查点 | 时机 | 验什么 | diff 基线 |
|---|---|---|---|
| mini 重跑 A | 补丁 4 后 | 数据侧:去重、谓词打标、GT 预检 | 本轮 jsonl(仅 input_sha 变) |
| mini 重跑 B | 补丁 5 后 | 判定侧:规则/LLM 分层、口径产出 | 重跑 A(仅 prompt/代码 sha 变) |

manifest 的 sha256 溯源已支持"每次只变一个哈希"的对照——**每个检查点与上一轮 diff 指标,所有变化必须可归因到当轮改动**;归因不了的变化即为新 issue,入册再查。

---

## 4. 本轮 jsonl 评审结论存档(供对照)

- ✅ 溯源/记账全自洽:0.909 = 20/22;top2 0.4 = 排名{3,2,4,4,2};pairs = C(cleared,2);excluded 8 = 5 hard_case + 3 illegal-precheck-fail。
- ✅ 两条 illegal-vetoed 全部正确否决,引用法条正确;illegal-chosen = 0(未报告的头条)。
- ⚠️ 2 处 verdict 扣分性质不坏:record 10 = 正确弃权(信息不足);record 6 traj_b = 疑似 benchmark 标签噪声(补丁 3 裁决)。
- 🐛 record 11:traj_b(aggressive)≡ traj_d(conservative)完全同轨迹——静止场景扰动退化(补丁 4);partial 无可见成因、tied 计数不一致(补丁 2)。
- 🐛 `llm_tokens_used` 全 0(补丁 2)。
- 📌 8/13 skipped = 仅 red_light 已准入,属预期;让行类接入后部分复活。
