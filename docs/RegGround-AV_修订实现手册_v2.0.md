# RegGround-AV 修订实现手册 v2.0

**依据**:首轮全量结果(commit `03c4701`)+ 交叉表 T1–T11 + 诊断审计(Opus 逐条裁定,待 Gate 0 核验)
**性质**:一次性修复清单。Gate 3 可在同一冻结 20 场景上修复后复冒烟；通过后冻结实现并只做一次全量重跑,全量结果不再回调。
**范围裁定(不再讨论)**:MVP 有效场景 = **red_light + pedestrian + oncoming**;yellow_light、emergency 按 SCOPE-CUT 剔除,论文如实声明。
**冻结状态**:2026-07-17 Gate 0 核验通过,本手册自此冻结。Gate 0 数字口径为 `diagnostics/v1.4/audit_sample.json` 的 85 个审计项,不是 259 场全量总体。

---

## Gate 0 · 实现前核验(半天,未过不得开工)

诊断结论中的定量断言目前是"AI 说的",实现前必须脚本核验,防止在错误诊断上施工:

- [x] **G0-1 A5 架构确证**:读 `context_builder.py` 实际代码,确认 prompt 中除 narrative 外无任何 SceneFacts 字段进入。10 分钟。
- [x] **G0-2 定量断言核验脚本**(约 30–50 行,对原始 JSONL 与 85 项审计样本按 `(scene_id, frame_token, anonymous_id)` join):
  - LLM 裁定总数与 uncertain 数(报告称 425 / 301);
  - benchmark `expected_verdict=exclude` 的理由分布(报告称 186 条"谓词仅覆盖红灯"、39 条"停车关系不明");
  - audit_026 原始数值(位移 0.0 m、静止 5.95 s、被 rule_engine veto);
  - rule_engine 全部 veto 是否同源于停止线代理谓词(报告称 71 次)。
- [x] **G0-3 A11 反例坐实**:5 条 citation/evidence 反例逐条对完整 prompt 原文。其中 4 条为引用 narrative/特征中不存在的细节,1 条(`audit_029`)为忽略 narrative 已明确提供的关键事实。
- [x] **G0-4 第二模型分歧对比**:只看逐条处置码分歧行 + needs_blind_confirmation 29 行,不看总体结论是否一致。

Gate 0 实测:LLM 裁定 425 / uncertain 301;`expected_verdict=exclude` 267 条,其中谓词仅覆盖红灯 186、停车关系不明 39、退化样本 30、GT precheck 失败 12;rule_engine veto 71 条且全部同源于停止线代理谓词;`audit_026` 位移约 0.0m、静止 5.95s 仍被 veto。逐条处置码为 FEAT-GAP 38、BENCH-FIX 30、JUDGE-MISS 7、FACT-FIX 7、SCOPE-CUT 3,其中 needs_blind_confirmation 29 行。

核验结果与报告不符的条目 → 对应修复项重新评估后再冻结;相符 → 本手册即最终版。

---

## 第一部分 Layer 1 修改

### 1.1 场景范围过滤(A6/A8)【小,先做】

- `dataset_loader` 或 `parse_dataset` 层面剔除 `scenario_type ∈ {YELLOW_LIGHT, EMERGENCY}` 的 keyframe,剔除量记入日志与 availability 报告。
- 轨迹窗口不足(waypoints < 60 或时长 < 6.0s)的 keyframe:不抛错,产出但标记 `window_insufficient=True`,benchmark 侧全部候选 `expected_verdict="exclude"`、原因 `not_evaluable_short_window`;仍保留在冻结 v2 范围，但归入结构不可评分池，不进入 N1 分母。
- yellow/emergency 场景与短窗口场景各保留 2–3 个为回归测试 fixture(含 scene-0286)。

### 1.2 SceneFacts 扩展(A9/A2 依赖)

```python
class SceneFacts(BaseModel):
    # ---- 原有字段保持 ----
    # ---- v2 新增 ----
    ego_turn_intent: Literal["left", "right", "straight", "unknown"] = "unknown"
    governing_stop_line_signed_m: float | None = None   # 沿车头方向到本向停止线的有符号距离;负值=keyframe 时已越过;无本向停止线=None
    pedestrian_in_forward_crosswalk: bool = False        # 行人所在横道须与 ego 前进路径相交(替代旧的"任意横道")
    pedestrian_moving: bool | None = None
```

实现要点:

- `ego_turn_intent` 由 GT 轨迹航向变化推导:6s 窗口内累计航向变化 > +25° 为 left,< −25° 为 right,|Δ| ≤ 25° 为 straight(阈值放配置,冒烟时校准);地图车道拓扑可作旁证但不阻塞。
- `governing_stop_line_signed_m`:本向停止线判定 = 停止线中线法向与 ego 航向夹角 < 45° 且位于 ego 前方 ±50m;多条取最近。找不到本向停止线时为 None,**不得回退到"最近任意停止线"**。
- `pedestrian_in_forward_crosswalk`:横道多边形与 GT 路径缓冲带(±3m)相交才计入。这是 A7 的一半:**冲突区按适用规则筛选**。

### 1.3 冲突区按 kind 分列(A7)

`TrajectoryFeatures` 中 `min_distance_to_conflict_m` 拆为按 kind 输出:

```python
class ConflictZoneMetrics(BaseModel):
    kind: Literal["stop_line", "ped_crossing"]
    governing: bool                      # 是否本向/前方适用(见 1.2)
    min_distance_m: float
    entered: bool
    entry_time_s: float | None           # 首次进入时刻(相对 keyframe)
    min_speed_in_zone_mps: float | None
    stopped_before_zone: bool | None     # 起点已在区内 → None,严禁默认 False
```

> `stopped_before_zone` 的 None 语义是 A2 修复的核心:**起点已在多边形内 = 该谓词不可判**,而非"未停车"。

### 1.4 智能体交互特征(新模块 `agent_interaction.py`)【最大件】

对 pedestrian / oncoming 场景,从 `sample_annotation` 跟踪链提取相关智能体在 6s 窗口内的位姿序列(插值到与 ego 轨迹相同时间轴,统一 keyframe 局部坐标系),对**每条候选轨迹**计算:

```python
class AgentInteraction(BaseModel):
    agent_kind: Literal["pedestrian", "oncoming_vehicle"]
    agent_track_id: str                          # 内部用,不进 prompt
    ego_zone_arrival_s: float | None             # 候选轨迹到达共同冲突区时刻
    agent_zone_window_s: tuple[float, float] | None  # 智能体占据冲突区的时间窗
    min_time_gap_s: float | None                 # 两者占区窗口的最小时间间隔;重叠为负
    min_spatiotemporal_gap_m: float | None       # 同时刻两者位置的最小欧氏距离
    crossing_order: Literal["ego_first", "agent_first", "temporal_overlap", "no_shared_zone"]
```

实现要点:

- 智能体轨迹用 annotation 的 `translation` 逐帧连线线性插值;窗口内消失的 track 截断处理。
- 冲突区:pedestrian 用 governing ped_crossing 多边形;oncoming 用 ego 转向路径与对向车路径的相交区域(缓冲 2m),无相交则 `no_shared_zone`。
- 每条候选最多输出 3 个最相关智能体(min_time_gap 最小者优先),控制 prompt token。
- **命名与措辞禁令**:字段名与摘要句式不得含结论词。"temporal_overlap" 是事实;"failed_to_yield" 是结论,禁止。

### 1.5 自然语言摘要更新

`natural_language_summary` 增补交互与停止线句式,措辞中性、须过泄露扫描:

```
行人预计 2.1s 后进入前方人行横道冲突区,占据至 4.8s;该轨迹 1.8s 后进入同一冲突区,最小时间间隔 -0.3s,最近时空距离 1.2m。
该轨迹于 2.4s 越过本向停止线;越线前最低速度 4.2 m/s,无完整停车。
[起点已越线时] keyframe 时刻自车已位于停止线前方 3.1m 处(已越线),停车判定不适用。
```

新增摘要句式全部加入 `leakage_guard` 测试语料。

### 1.6 违规注入修订(A3)

- **退化检测**:合成后将每条 illegal 的特征按 LLM 可见精度舍入,与全部其他候选逐一比对;完全相同 → 记 `no_violation_injectable`。严禁为制造差异平移、缩放或扰动 x/y。
- **场景相关扰动**(替代一刀切 `_remove_deceleration`):
  - red_light:延后减速起点至越线后(保证越线时刻在红灯快照有效期内,见 2.3);
  - pedestrian:同路径压缩让行等待并在交互分析后校验;目标 `min_time_gap_s <= -0.3s`,未达标则加大速度压缩重试,仍受曲率/drivable-area 约束;物理约束下仍未达标记 `no_violation_injectable`;
  - oncoming(仅 ego_turn_intent=left 场景注入):压缩切入间隙,使与对向车 `min_time_gap_s` 转负。
- **右转/直行场景不注入让行类 illegal**(A9 连带):`ego_turn_intent != left` 的 oncoming 场景、行人横道不在前方路径上的 pedestrian 场景,illegal 槽位改注 aggressive 变体并在 benchmark 记 `no_violation_injectable`。

### 1.7 ONCOMING 场景谓词门控(A9)

`scenario_type=ONCOMING` 的判定条件收紧为:`oncoming_vehicle_moving AND ego_turn_intent == "left"`。不满足左转条件的原 oncoming 场景按 facts 重派(有行人事实 → PEDESTRIAN;有红灯 → RED_LIGHT;否则 UNKNOWN 并从主实验集剔除)。重派前后场景数变化记入报告。

---

## 第二部分 接口修改(A5,全项目最高优先级)

### 2.1 JudgeInput 增补 SceneFactsDigest

```python
class SceneFactsDigest(BaseModel):
    """Layer-3 安全的场景要件投影:只含规则触发事实,无标签、无坐标、无结论词。"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_phase: Literal["red", "green", "unknown"]          # yellow 已裁剪
    phase_source: Literal["drivelm_status", "none"]
    ego_turn_intent: Literal["left", "right", "straight", "unknown"]
    governing_stop_line_signed_m: float | None
    pedestrian_in_forward_crosswalk: bool
    pedestrian_moving: bool | None
    oncoming_vehicle_moving: bool
    location_is_intersection: bool | None

class JudgeInput(BaseModel):
    # ---- 原有字段保持 ----
    scene_facts_digest: SceneFactsDigest          # v2 新增
```

红线(写入测试):

- Digest 由 `InterfaceAdapter` 从 SceneFacts 投影,**禁止**包含 `conflict_zones` 多边形坐标、variant/benchmark 任何字段;
- Digest 渲染文本进入 prompt 的独立"场景要件"小节,渲染函数纳入泄露扫描覆盖;
- 渲染措辞陈述事实不下结论:"当前方向信号灯为红灯相位(来源:DriveLM 标注)"✓,"存在闯红灯风险"✗。紧随相位增加:"信号灯相位为 keyframe 时刻(t=0)快照,其后相位未知"。

### 2.2 Layer 3 context_builder / prompt

- prompt 结构改为:场景要件(digest 渲染)→ 可引用规则 → 候选轨迹特征(含交互摘要)。
- prompt 明确指示:判定须同时依据"义务是否触发(场景要件)"与"义务是否履行(轨迹特征)";要件缺失(unknown/None)时如实弃权。
- `LeakageGuard.assert_prompt_safe()` 覆盖新增小节;token 预算复核(digest + 交互特征约增 15–25%,确认 `max_rules_in_context` 截断逻辑仍安全)。

### 2.3 rule_engine 谓词重写(A2,与 4.2 共用)

新建共享谓词模块(供 rule_engine 与 GT precheck / benchmark 打标共同调用,A4):

```python
def red_light_crossing_verdict(f: TrajectoryFeatures, d: SceneFactsDigest) -> PredicateResult:
    """返回 DECISIVE_VETO / DECISIVE_CLEAR / NOT_DECIDABLE(+reason)"""
```

判定逻辑(顺序执行,首个命中即返回):

1. `signal_phase != "red"` 或本向停止线为 None → NOT_DECIDABLE("no_governing_signal_or_line");
2. `governing_stop_line_signed_m < 0`(keyframe 已越线)→ NOT_DECIDABLE("start_beyond_line")——**严禁判 veto**;
3. `ego_turn_intent == "right"` → NOT_DECIDABLE("right_turn_on_red_exemption")交 LLM(中国法规红灯右转默认允许,豁免为简化假设,论文声明);
4. 候选越线(stop_line kind 的 entered=True)、最大侵入深度 ≥ 0.15m、越线时刻 ≤ `phase_validity_horizon_s`(默认 2.0s,配置化)且越线前无完整停车 → DECISIVE_VETO;侵入深度 <0.15m → NOT_DECIDABLE;
5. 候选全程未越线且于线前完整停车 → DECISIVE_CLEAR;
6. 其余(缓行、临界、越线时刻超出相位有效期)→ NOT_DECIDABLE 交 LLM。

> `phase_validity_horizon_s` 处理相位快照的时间有效性:红灯快照只能为其后短窗内的越线行为背书,窗外越线(灯可能已变)一律交 LLM 语义判断。

让行类谓词(A1,减配版):

- pedestrian:`pedestrian_in_forward_crosswalk AND pedestrian_moving AND min_time_gap_s < 0 AND 未在区前停车` → DECISIVE_VETO;`min_time_gap_s > 3.0 或 agent_first 且已停等` → DECISIVE_CLEAR;其余 NOT_DECIDABLE。
- oncoming:**不做直裁谓词**(一个月约束下裁掉),全部交 LLM;benchmark 侧 oncoming 的 expected_verdict 依赖注入类型 + 人工子集。

---

## 第三部分 Layer 2 修改【最小】

- 删除 `C-EMERGENCY-ACTIVE`、`R-YLD-04` 及 override 示例;scene-0286 场景连同旧图定义降为扩展测试 fixture(override 机制的代码与测试**保留**,论文作为机制演示)。
- `graph_version` 升至 `2026-07-v3.1`,加载校验通过。
- R-SIG-02(黄灯)保留在图中但因场景已裁剪不会触发,不删(减少 churn);图内规则数、条件数变化写入变更记录。
- 其余不动。**Layer 2 无行为变更,回归测试全绿即验收。**

---

## 第四部分 Benchmark v2

### 4.1 expected_verdict 打标流程重排(A10)

打标必须消费与 judge 同源的信息:`SceneQuery → retriever.retrieve() → RuleSubgraph.active_rules` 之后,基于 active_rules + 共享谓词(2.3)打 expected_verdict。被 override 覆盖的规则不得产生 vetoed 期望。

### 4.2 expected_verdict 语义修订

```python
expected_verdict: Literal["cleared", "vetoed", "exclude"]
exclude_reason: Literal[
    "not_evaluable_short_window", "injection_degenerate", "no_violation_injectable",
    "predicate_not_decidable", "gt_precheck_not_evaluable",
] | None
```

- **exclude 只作用于候选,不作用于场景**:场景可评分性由"是否存在 ≥1 个 cleared 期望 + ≥1 个 vetoed 期望的候选对"定义(pair 级),不再要求全员可判。
- conservative/suboptimal 等"合规但非最优"候选:`expected_verdict="cleared"`,**进入可接受集合**,不再标 exclude——这是修复 T9 55 条 conservative-exclude 的口径。
- GT precheck 用共享谓词重跑(A4):失败 GT 记 `gt_precheck_not_evaluable` + 原因留档,**不得**转为 vetoed 期望。

### 4.3 difficulty 重定义(按要件可见性)

- easy:直裁谓词 DECISIVE(要件齐全 + 特征明确);
- medium:要件齐全但特征临界(NOT_DECIDABLE 中 gap/速度落在阈值 ±20% 带内);
- hard:hard_case 变体,或要件部分缺失但语义可判。
- 生成参数不再作为难度依据。circularity 声明:easy 与谓词同源,故主指标按 medium/hard 分层报告时须附 `decided_by` 占比(既有 telemetry)。

---

## 第五部分 评分与实验口径 v2

主指标(全部 pair 级 / 集合级,按 scenario_type × difficulty 分层报告):

| 指标 | 定义 |
|---|---|
| violation_detection_rate | P(system=vetoed \| expected=vetoed) |
| false_veto_rate | P(system=vetoed \| expected=cleared) |
| abstention_rate | P(system=uncertain),单列报告,不并入错误 |
| pair_consistency | 对每个 (expected cleared, expected vetoed) 候选对,系统偏序与期望一致的比例(vetoed<cleared 即一致;任一方 uncertain 记 abstain 桶) |
| chosen_acceptable_rate | chosen ∈ {expected=cleared 候选};chosen=None 时该场景记 no_choice 桶单列 |
| citation_validity / accuracy | 口径不变(runtime / 人工子集) |

2×2 消融:仅在人工标注子集上跑,指标同上 + 幻觉率(§10.3 定义)。v1 全量结果作为"修复前"参照保留引用,不重跑 v1。

---

## 第六部分 盲标包 v2 重导

**前提**:全量重跑完成后导出(输入对齐原则:标注者证据区 = 渲染后的 JudgeInput v2,逐字一致;retrieved_rules 双方共有)。

1. `structured_facts` 列改为 digest 渲染文本(与 prompt 同一渲染函数,单一来源);`trajectory_summary` 用含交互特征的新摘要。
2. **样本重抽**:分层依据基于 v2 输出重新计算(旧抽样作废);沿用 SHA-256 + seed=42。
3. 前次修改意见中未完成项一并落实:全量规则卡进说明页(P0-2)、压力案例混洗(P1-1)、回收校验脚本(P1-2)、校准轮(P1-3)、示例行(P1-4)。
4. 偏好工作簿(25 场景,场景级排序)与 verdict 表**基于同一版输出**同批导出。
5. 若朋友已标注旧表:回收留存,作为"要件补充前后人类弃权率变化"对照素材,不并入 gold。
6. README 增补设计声明:盲标测"同等信息下的人机判断一致性",非全知真值标注。

---

## 第七部分 执行顺序与冒烟门禁

```
Gate 0 核验(0.5d)
  → 冻结本手册
  → 实现批次一:1.1 / 1.2 / 1.7 / Layer 2 删减 / 2.1 digest 模型   (1–1.5d)
  → 实现批次二:1.3 / 1.4 交互特征 / 1.5 摘要 / 2.2 prompt          (1.5–2d)
  → 实现批次三:2.3 共享谓词 / 1.6 注入 / 第四部分 benchmark / 第五部分评分 (1.5–2d)
  → Gate 1:手算 fixture 全过
  → Gate 2:可视化抽查 10 场景(特征值 vs 渲染图)
  → Gate 3:冒烟 20 场景过完整 judge
  → 冻结 → 全量重跑 → 盲标包 v2 导出发放 → (标注周转期写论文) → 回收 → IAA/gold → 消融 → 完稿
```

**Gate 1 手算 fixture(必须先于实现或与实现同步写,答案由你手工计算)**:

- F1 行人先到:行人 1.0–3.0s 占区,候选 4.5s 到 → agent_first, gap=+1.5s;
- F2 时间重叠:行人 2.0–5.0s 占区,候选 2.5s 到 → temporal_overlap, gap<0;
- F3 左转对向:对向车匀速 8 m/s 迎面,ego 左转路径相交,手算相交区与两方到达时刻;
- F4 起点越线:keyframe 时 ego 在停止线前方 2m → signed_m=−2.0,stopped_before_zone=None,谓词 NOT_DECIDABLE;
- F5 右转判别:累计航向 −60° → ego_turn_intent="right"。

**Gate 3 冒烟通过标准(方向性,非论文承诺)**:20 场景只裁行为门禁:pedestrian 候选 uncertain 率 <40%(v1 为71%);red_light conservative 停车剖面不被 veto;illegal(非退化)veto 数 > cleared 数;泄露扫描零命中;digest 内容与 SceneFacts 逐字段一致。结构池另按设计内 oncoming、`start_beyond_line`、pedestrian 触发/交互缺失及 `not_evaluable_short_window` 拆分；全量报告 N1、N2 和 N2/N1，覆盖率只作版本基线记录，不设阈值，也不作通过或未通过判定。注入失败与 borderline 仍留在 N1。未达行为门禁 → 修复后重新冒烟,**冒烟循环不受"一轮"限制,全量只跑一次**。

## 第八部分 测试与验收

新增/修改测试(在各层既有测试要求之上):

| 测试 | 覆盖 |
|---|---|
| `test_agent_interaction.py` | F1–F3 手算 fixture;track 中断截断;3 智能体上限 |
| `test_scene_facts_v2.py` | F4/F5;本向停止线找不到时 None 不回退;forward crosswalk 相交判定 |
| `test_shared_predicates.py` | 2.3 六分支全覆盖;start_beyond_line 严禁 veto;右转豁免;0.15m 侵入鲁棒带 |
| `test_injection_v2.py` | illegal 与 GT 路径重合;不可注入时不造几何并记 `no_violation_injectable` |
| `test_judge_input_digest.py` | digest 无坐标/无标签字段;相位快照句过泄露扫描;prompt 含要件小节 |
| `test_scoring_v2.py` | pair 级指标;结构池基线分母与短窗口排除;suboptimal 晚越线保持 exclude;override 后置打标 |
| `test_scope_cut.py` | yellow/emergency 剔除;fixture 保留;短窗口 not_evaluable |

验收 checklist:

- [ ] Gate 0 四项核验记录归档(含与 Opus 报告的差异说明,如有)
- [ ] mypy --strict / ruff 零告警;既有隔离测试与泄露测试全绿
- [ ] Gate 1/2/3 记录归档(冒烟数字截图或 json)
- [ ] 全量 v2 输出 + benchmark v2 + 盲标包 v2 均带 SHA-256 与 commit 锚点
- [ ] v1 → v2 变更对照表(场景数、候选数、exclude 分布、uncertain 率)自动生成,供论文"修订依据"一节直接引用
- [ ] 冻结后未对 benchmark 做任何回调(git log 可证)

---

## 附:明确不做清单(防蔓延)

- oncoming 直裁谓词(TTC 版)——交 LLM + 人工子集;
- 黄灯相位时序补全、emergency 保留、视觉相位感知;
- Bradley-Terry(用多数胜场);全量 2×2 消融(只跑人工子集);
- 任何 v1 结果的修改或重跑;
- 冻结后新增特征或调 prompt(发现问题记 limitation,不返工)。
