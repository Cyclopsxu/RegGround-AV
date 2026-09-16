# RegGround-AV 人工盲标包 v2

本目录由冻结的 v2 正式运行导出。盲标测量的是标注者与系统在**相同可见信息**下
的判断一致性，不把标注者当作拥有原始坐标、未来时序或 benchmark 标签的全知裁判。

## 当前抽样

- 抽样框：LLM 裁定 382 条；rule-engine 决定性候选 118 条。
- 正式 verdict 核心样本 180 条：LLM 主池 144 条、predicate-human 层 36 条。
- 压力案例 20 条，与核心样本统一混洗后形成 200 行正式工作簿。
- borderline 54/180（30.0%，目标带 25%–35%）。
- 行人场景覆盖 12/12；行人 LLM 候选 33 条全取。
- 偏好工作簿 36 场，全部来自至少 2 条候选被系统判 cleared 的场景。
- 校准轮 8 行：明显违规、明显通过、临界减速、creeping 各 2 行；不计入正式样本。

精确分层、来源哈希、输出哈希与逐字校验结果见
`results/sample_manifest.json`。分层归属、系统判决和 benchmark 只存在于
`private/`，不得发给标注者。

## 发放顺序

1. 先给两名标注者分别复制 `calibration_annotation_template.xlsx`，独立填写。
2. 按 `results/CALIBRATION_PROTOCOL.md` 对齐 uncertain、creeping 和临界减速口径；
   把形成的判例写入协议的“最终共识”区，再开始正式标注。
3. 分别复制并发放 `blind_annotation_template.xlsx` 和
   `preference_annotation_template.xlsx`。不得发放 JSONL、manifest、哈希文件或
   `private/`。
4. 回收后先运行校验脚本；未通过的行退回补标，不进入 IAA、揭盲或评分。

## 措辞敏感补充批

`results/outputs/wording_sensitive_supplement_v1/` 是影子评估关闭后追加的 5 行
盲标仲裁批：`022558…` 的 4 条措辞敏感候选与 `575b55…/traj_a` 孤例。
它不改写原 200 行正式样本，也不启动新的评估版本。发放时只给标注者
`supplemental_blind_annotation_template.xlsx`；选择原因、v_a/v_b 判定和谓词标签
只保存在 `private/outputs/` 的 sidecar 中。

## 回收校验

```bash
uv run python human_annotation_v2/code/validate_returns.py \
  --input path/to/rater_a_completed.xlsx \
  --template human_annotation_v2/results/blind_annotation_template.xlsx \
  --output human_annotation_v2/results/return_validation_rater_a.json
```

偏好表和校准表使用同一个脚本，只需更换 `--template`。

## 重新导出

工作簿必须使用 Codex `load_workspace_dependencies` 返回的 Node executable 与
node_modules，不允许改用系统 Excel 库：

```bash
uv run python human_annotation_v2/code/export_human_audit.py \
  --artifact-node <bundled-node-executable> \
  --artifact-node-modules <bundled-node-modules>
```

导出会依次执行冻结输入哈希检查、JudgeInput v2 重放、运行日志特征哈希核对、
泄露扫描、artifact-tool 渲染、公式错误扫描和最终 xlsx 逐单元格回读。任一门禁
失败即停止，不产生可发放声明。
