# RegGround-AV 人工盲标包 v1

本目录独立保存人工验证的抽样代码、盲标工作簿和揭盲 sidecar，不修改或依赖
`src/`、`tests/`、冻结评估集及现有运行日志中的任何代码路径。

## 给标注者：开始前必读

项目会提供三份工作簿：正式 verdict 表 `blind_annotation_template.xlsx`、场景级偏好表
`preference_annotation_template.xlsx` 和发放前校准轮 `calibration_annotation_template.xlsx`。
请只使用自己的副本，不要与另一名标注者讨论，也不要查看本目录的 `private/` 文件。两名
标注者必须先独立完成正式表和偏好表，计算一致性后才能讨论分歧；校准轮不计入正式样本。

每行代表一个匿名候选轨迹。请依据该行提供的场景描述、结构化事实、法规、override 关系和
轨迹数值摘要填写以下六列：

1. `human_verdict`
   - `cleared`：现有证据不支持硬规则否决；
   - `vetoed`：现有证据明确支持硬规则否决；
   - `uncertain`：信息不足、规则冲突或无法可靠裁定。
2. `human_applicable_rule_ids`：填写与当前场景和判断相关的规则 ID，多个 ID 用英文分号 `;` 分隔；无适用规则填 `NONE`。
3. `human_violated_rule_ids`：只填写已有证据确认违反的规则 ID；`cleared` 或 `uncertain` 必须填 `NONE`；`vetoed` 至少填写一条 hard rule。
4. `human_confidence`：填写 1–5，1 表示很不确定，5 表示很确定。
5. `human_reason`：用一句完整的话说明判断依据。
6. `notes`：可选；记录代理区偏移、描述冲突或其他异常。

标注纪律：

- 不猜测表格中没有提供的信息；证据不足时选择 `uncertain`。
- `overridden` 规则只作为审计上下文；应结合 `override_relations` 判断当前有效约束。
- 不根据轨迹 ID 字母、行顺序或语言风格推测系统答案。
- 不搜索或打开原始 JSONL、benchmark 标签、系统 verdict、chosen 结果或 `private/hidden_sidecar.jsonl`。
- 保存时不要删除、插入或重排记录行，也不要修改 `audit_id`。

规则列示例：如果 `R-YLD-01` 和 `R-YLD-05` 与当前场景相关，但证据不足以确认违规，填写
`human_applicable_rule_ids=R-YLD-01;R-YLD-05`、`human_violated_rule_ids=NONE`，并将
`human_verdict` 标为 `uncertain`。

完成后，将个人文件分别命名为 `rater_a_completed.xlsx` 和 `rater_b_completed.xlsx` 交回项目负责人。

## 场景级偏好表

`preference_annotation_template.xlsx` 每行代表一个主样本场景，并列展示该场景全部匿名候选
轨迹及特征摘要。候选顺序使用独立于 verdict 表的固定 SHA-256 排序。请填写：

- `human_best_trajectory`：单选最优轨迹；没有可接受轨迹填 `NONE`；
- `human_ranking`：全部候选从优到劣排序，如 `traj_c>traj_a=traj_e`；
- `human_confidence`：1–5；
- `human_reason`：一句话说明排序依据。

该表只用于场景级 chosen 一致率、Kendall's tau 和由全排序展开的 pairwise 偏好分析，不能与
verdict 表互相查看或根据 verdict 表抽中的候选推断答案。

## 抽样设计

主样本为 120 个候选，来自 120 个不同场景：

- LLM 判定候选 90 个：legacy 全场景包保持 pedestrian 49、oncoming 21、red_light 13、
  yellow_light 5、emergency 2；v2 MVP 包（裁掉 yellow/emergency）按原比例重分配为
  pedestrian 53、oncoming 23、red_light 14；脚本按输入场景集合自动选择 profile，并写入 manifest；
- RuleEngine 判定候选 30 个：均来自 red_light。

另有 20 个压力案例：1 个真实 override、4 个多条件法规场景、5 个罕见 LLM veto、
5 个 no-choosable uncertain 和 5 个 citation repair。压力案例用于失败分析，不与
120 个主样本合并估计总体准确率。首轮全量结果中只有 `scene-0286` 实际触发 override，
因此不人为凑足 5 个 override。

抽样使用 SHA-256 固定排序，seed 为 42。主样本严格执行“一场景一候选”。盲标工作簿不会显示
样本分组、系统判决来源、系统 verdict、chosen、benchmark、变体类型或候选来源。

v2 输出中，`pedestrian_gap_borderline`（兼容 `gap_borderline`）和
`borderline_or_phase_horizon_unknown` 候选作为 medium 难度阈值带单独分层：verdict 表以 30% 为
目标，并强制校验占比处于 25%–35%。实际候选池、抽中数量和占比写入 manifest；这些字段仅用于
抽样审计，不进入发给标注者的证据列。

## 目录结构

```text
human_annotation_v1/
├── README.md
├── code/
│   ├── annotation_schema.mjs
│   ├── export_human_audit.mjs
│   └── validate_returns.mjs
├── results/
│   ├── blind_annotation_template.xlsx
│   ├── blind_samples.jsonl
│   ├── preference_annotation_template.xlsx
│   ├── preference_blind_samples.jsonl
│   ├── calibration_annotation_template.xlsx
│   ├── calibration_blind_samples.jsonl
│   ├── artifact_hashes.json
│   └── sample_manifest.json
└── private/
    ├── hidden_sidecar.jsonl
    ├── preference_sidecar.jsonl
    └── calibration_sidecar.jsonl
```

只把 `results/blind_annotation_template.xlsx` 的副本发给标注者。不要共享整个目录。

## 重新生成

脚本默认读取项目现有首轮全量输出，并且只写入本目录：

```bash
cd human_annotation_v1
node code/export_human_audit.mjs
```

如需显式指定输入和输出：

```bash
node code/export_human_audit.mjs \
  --input ../logs/eval_set_v1_full_v3_5.jsonl \
  --output-root . \
  --seed 42
```

脚本会校验主样本分层、120 个独立场景、20 个压力案例、`scene-0286/traj_a` 强制纳入、
盲标字段隔离、偏好场景数量（30–40）、校准类别覆盖及输出哈希。重新生成会覆盖本目录已有的
模板和 sidecar，请先保管已回收的标注结果。

规则卡会从 `--rule-graph` 指定的 YAML 动态读取；当前版本包含 8 条规则。若标注者认为某条
规则适用于场景但不在 `retrieved_rules` 中，应根据说明页的全量规则卡填写，并在 `notes` 中
注明“未检索到”，以支持 citation recall 统计。

## 回收校验

收到标注者文件后，先分别执行：

```bash
node code/validate_returns.mjs \
  --input results/rater_a_completed.xlsx \
  --template results/blind_annotation_template.xlsx \
  --output results/return_validation_rater_a.json
```

校验器会检查 140 行、`audit_id` 顺序、verdict、规则 ID（包括 `R-YLD-1`、中文分号等格式
归一化）、hard-rule 约束、confidence 和 reason，并生成逐行退回清单。只有 `ok: true` 后才进入
IAA 和揭盲评分。

## 校准轮

`calibration_annotation_template.xlsx` 从主样本之外确定性抽取 8 行，覆盖临界减速、creeping、
明显违规和明显通过四类，每类尽量两行。两名标注者先独立完成，再讨论分歧，并把实际形成的
裁定标准补写到负责人保管的说明页；校准行不进入 120 条 core 或 20 条 stress 的统计。

## 哈希锚点

`results/artifact_hashes.json` 记录 `blind_samples.jsonl`、`sample_manifest.json`、正式模板以及
偏好/校准模板的 SHA-256，并以评估结果 commit `03c4701` 为引用锚点。该文件不包含人工答案或
private sidecar 内容。

> 运行环境需要 Node.js 和 `@oai/artifact-tool`。本次生成使用 Codex 工作区随附运行时；
> `node_modules` 仅为本地运行时链接并已在本目录 `.gitignore` 中排除。
