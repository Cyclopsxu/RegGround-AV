# RegGround-AV 补丁实施记录 · 2026-07-11

## 已完成

1. 分层评估口径：新增 illegal-chosen、scorable/unscorable、GT 参考指标，以及 RuleEngine/LLM/fallback 分层报告。
2. 诊断插桩：`partial_reasons` 成为 status 的唯一来源；逐阶段记录调用数、输入/输出 token、重试、特征文本哈希与引用修复明细。
3. trainval 可用性探针：复用生产事实提取、场景分类和准入逻辑，不调用 LLM。
4. 几何人工审计：本地 mini 抽取 15 条轨迹，四项谓词与叠加图一致率 15/15；record 6 traj_b 归因为标签错误，不是停止线代理偏移。
5. 数据侧重建：扰动意图与期望标签分离；最终标签只由实际几何特征推导；GT 同样预检；相同 LLM 可见特征显式标记 degenerate；静止 ego 提前拒绝。
6. RuleEngine 接入：只对具有 `R-SIG-01` 且几何字段充分的明确样本直裁；其他场景和模糊样本保持 LLM/uncertain 路径。
7. 可复现清单：manifest 增加 benchmark 数据版本、builder 哈希和合规谓词哈希。原始输入文件未被改写，因此 `input_sha256` 保持原始数据文件语义。

## mini 离线预检

- 输入：`data/layer1/raw_scene_dataset.json`，13 条。
- 准入：2 条通过；8 条 `scenario_type_unknown`；3 条 `ego_stationary_low_information`。
- record 6 对应场景的 aggressive 候选：由实际几何改判为 `vetoed`。
- 静止场景中的舍入特征碰撞：均显式记录 `is_degenerate=true` 并标为 `exclude`。
- 每条候选均记录 `expected_verdict_reason`、`predicate_version` 和 `feature_text_hash`。

## 验证结果

- `pytest`：277 passed。
- `ruff check src tests`：通过。
- `mypy --strict --explicit-package-bases src`：通过。
- `git diff --check`：通过。

## 审计材料

- 几何审计记录：`artifacts/geometry_audit_20260711/geometry_audit_record.md`
- 几何叠加总览：`artifacts/geometry_audit_20260711/geometry_audit_overview.png`
- 评估口径：`docs/RegGround-AV_评估口径_v1.0.md`

## 尚未执行的外部步骤

1. mini 联网重跑 B：运行需要向已配置的外部 DeepSeek API 发送数据集派生的场景文本和轨迹特征；当前未获得针对该数据传输风险的明确授权，因此没有发送数据。
2. trainval spike 实数报告：当前工作区没有 nuScenes `v1.0-trainval` metadata，脚本与测试已完成，但无法生成真实分层计数。
3. trainval 正式评估与多人外部审计：依赖上一项数据和独立人工标注，不应由同源几何谓词自证。
