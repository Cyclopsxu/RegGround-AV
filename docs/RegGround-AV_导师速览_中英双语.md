# RegGround-AV 导师速览 | Supervisor Brief (Bilingual)

> 目的：用最少篇幅说明研究问题、已完成工作、当前证据与下一步。数字均按冻结 v2 运行口径；v1 与 v2 不混算。

## 1. 项目一句话 | One-sentence summary

- **中文**：RegGround-AV 研究如何为自动驾驶候选轨迹生成**可追溯、法规接地且允许弃权**的偏好/合规性监督信号：确定性极端案例由几何谓词处理，证据不足或边界案例交给 RAG+LLM，并以人工标注作独立验证。
- **English**: RegGround-AV studies how to generate **traceable, regulation-grounded, and abstention-aware** preference/compliance signals for candidate driving trajectories: geometric predicates handle decisive cases, while RAG+LLM handles evidence-limited or borderline cases, with human annotation as independent validation.

## 2. 研究动机与方法 | Motivation and approach

- **中文**：v1 虽工程运行完整（259/259、零失败），但监督质量不可用：仅 13/259 可评分，候选级 `uncertain` 为 58.4%。这表明“系统跑通”不等于“LLM 裁判可作为可靠偏好信号”。
- **English**: Although v1 ran end-to-end (259/259, zero failures), its supervision was not usable: only 13/259 cases were scorable and candidate-level abstention was 58.4%. A working pipeline is therefore not the same as a reliable LLM judge.

- **中文**：v2 采用三层分工：①从 nuScenes 地图、轨迹和标注抽取中性 `SceneFacts`；②图式 RAG 检索适用法规与共享几何谓词；③LLM 在不确定区域作裁决、排序和法条引用。关键原则是：**隔离标签，但不能隔离裁决所需事实；不可得证据必须显式可见。**
- **English**: v2 uses a three-part division of labour: (1) neutral `SceneFacts` from nuScenes maps, trajectories, and annotations; (2) graph RAG for applicable rules and shared geometric predicates; and (3) an LLM for uncertain verdicts, ranking, and rule citations. The central principle is: **hide labels, not decision-relevant facts; make unavailable evidence explicit.**

## 3. 当前进度与已得证据 | Progress and evidence to date

- **中文**：核心实现、单元/静态检查和 v2 全量运行已完成；全量处理 259 条记录，运行 4.14 小时、零失败，产生 510 个候选裁定，其中 382 个（75%）进入 LLM 层。
- **English**: Core implementation, unit/static checks, and the v2 full run are complete. The run processed 259 records in 4.14 hours with zero failures, producing 510 candidate verdicts; 382 (75%) reached the LLM layer.

- **中文**：修订后弃权更具语义性：总体 `uncertain` 34.5%，行人场景从 v1 的 71.3% 降至 8.3%；红灯场景的 33 个弃权均可归因于“关键帧后的灯态不可确认”，而非随机失效。
- **English**: Abstention is now more semantically meaningful: overall `uncertain` is 34.5%, and the pedestrian rate fell from 71.3% in v1 to 8.3%. All 33 red-light abstentions are attributable to unknown signal state after the keyframe rather than random failure.

- **中文**：影子实验直接检验 LLM 的价值：移除规则引擎后，LLM 与冻结几何标签的一致率为 68.6%，但 `false_veto=0%`；排除弃权后为 81/82（98.8%）。说明当前主要失效模式是审慎弃权，而非错误否决。
- **English**: A shadow experiment directly tests LLM value: without the rule engine, agreement with frozen geometric labels is 68.6%, with `false_veto=0%`; conditional on a non-abstaining answer it is 81/82 (98.8%). The dominant failure mode is cautious abstention, not wrongful rejection.

## 4. 当前风险与下一步 | Current risks and next steps

- **中文**：最高优先级风险是 A133：停止线“有符号距离/侵入深度”的实现与字段语义不一致，且该量还参与候选生成。修复会改变部分候选几何，因此必须作为**新版本实验**，不能与冻结 v2 混算。
- **English**: The highest-priority risk is A133: the implemented stop-line signed distance/intrusion depth does not match its stated semantics, and it also affects candidate generation. A fix changes some candidate geometry, so it requires a **new experimental version**, not a mixture with frozen v2 results.

- **中文**：论文收尾前仍需：完成 A133 版本升级与重跑；回收双人盲标、计算 IAA/人工 gold；在人工子集运行 2×2 消融；补齐少量数据池对账与案例归档。自动满分指标只报告为“构造一致性”，不称为系统准确率。
- **English**: Before paper completion: implement and rerun the A133 version upgrade; collect double-blind annotations and compute IAA/human gold; run the 2×2 ablation on the annotated subset; and close a small number of data-pool reconciliations and case archives. Automatic perfect scores will be reported only as “construction consistency,” not system accuracy.

## 5. DriveLM 数据集简述 | DriveLM dataset at a glance

- **中文**：本项目使用的是 **DriveLM-nuScenes** 的语言信息，并与 nuScenes 的轨迹、3D 标注和 Boston 地图结合。DriveLM 原始工作还发布了仿真数据集 DriveLM-CARLA；本项目不把 CARLA 数据作为主实验来源。
- **English**: This project uses the language information in **DriveLM-nuScenes**, combined with nuScenes trajectories, 3D annotations, and Boston maps. The original DriveLM work also released the simulated DriveLM-CARLA dataset; CARLA is not the source of this project’s main experiment.

- **中文**：DriveLM-nuScenes 来自 nuScenes 视频片段：先选关键帧与关键对象，再围绕感知、预测、规划（以及行为/运动）构建问答。部分感知问答由 nuScenes 与 OpenLane-V2 的真值自动生成；绝大多数问答由人工标注。
- **English**: DriveLM-nuScenes is built from nuScenes video clips: key frames and key objects are selected first, then QAs are constructed around perception, prediction, planning (and behaviour/motion). Some perception QAs are generated from nuScenes and OpenLane-V2 ground truth; the majority are manually annotated.

- **中文**：其“描述性对话”更准确地说是**图结构、多轮依赖的 QA**，而非自由聊天：5 名领域专家设计问题模板；标注者按模板描述对象状态、未来运动、风险与可行动作；同一帧的 QA 以对象关系和“感知→预测→规划”的依赖边连接。每个批次抽检 10%，不达标则整批返标，保证自然语言描述与逻辑链的一致性。
- **English**: Its “descriptive dialogue” is more precisely **graph-structured, multi-step QA**, rather than free-form chat: five domain experts designed question templates; annotators describe object state, future motion, risks, and feasible actions; QAs in each frame are linked by object relations and Perception→Prediction→Planning dependencies. Ten percent of every batch is quality-checked, and an entire batch is relabelled if it fails the threshold.

## 6. 导师可快速判断的结论 | Takeaway for supervision

- **中文**：项目已从“能跑的 LLM 判定器”推进为“可审计的混合裁决与偏好标注框架”，并获得了 LLM 在边界案例中保守弃权的实证；当前关键不是继续堆指标，而是以 A133 版本升级和独立人工验证守住证据可信度。
- **English**: The project has progressed from a runnable LLM judge to an auditable hybrid verdict and preference-labelling framework, with evidence that the LLM abstains conservatively on borderline cases. The priority is now not more headline metrics, but protecting evidential validity through the A133 version upgrade and independent human validation.

---

**依据 | Basis:** `docs/论文素材汇总_v2.md`、`paper/论文素材汇总_从各报告提取.md`、`paper/A133_停止线几何口径_论文补充素材.md`，以及 DriveLM (Sima et al., ECCV 2024) 本地论文。
