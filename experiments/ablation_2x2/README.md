# 2×2 消融实验（技术文档 v1.0）

本目录是独立评估臂，不修改或 monkey-patch `src/`。四条件固定为：

| condition | GraphRAG | citation | rule engine |
|---|---:|---|---:|
| `full` | on | enforce | off |
| `no_validation` | on | observe | off |
| `no_rag` | off | enforce（全量图谱法名/条号目录） | off |
| `baseline` | off | observe | off |

所有条件固定使用 digest `v_a`、相同候选顺序和模型配置。`no_rag` / `baseline`
prompt 不含规则文本或 `rule_id`，但强制模型在 `cited_provisions` 中给出“法名 +
条号”。observe 只打 `passed|violated` 标记，不重试、不拦截；enforce 最多做一次
引用修复。首发和终态引用都写入 record。

## 测试

```bash
uv run pytest experiments/ablation_2x2/tests -q
```

## Dry-run

工作簿使用 Codex bundled artifact runtime 生成，避免给生产环境增加 Python 表格依赖：

```bash
uv run python -m experiments.ablation_2x2.run_ablation \
  --dry-run \
  --backend dry \
  --artifact-node /path/from/load_workspace_dependencies/node \
  --artifact-node-modules /path/from/load_workspace_dependencies/node_modules
```

默认产物目录是 `results/ablation_2x2/DRY_RUN/`。可用 `--stop-after 1`
先写一个场景，再用同一命令续跑；幂等键是 `condition + frame_token`，每条 append
后都会 flush + fsync。

## 正式数据：与人工标注 v2 完全对齐

正式清单不是另行抽样，也不是全量数据。它取两张正式人工 gold 的 frame-token
并集：verdict 200 行覆盖 78 场，preference 36 场中有 32 场与 verdict 重叠、4 场
只在 preference 表出现，因此最终是 82 场。校准 8 行和 wording-sensitive 补充批
不进入主 2×2。

两名标注者完成 IAA 和 adjudication 后，先冻结两个 CSV：

- verdict：`audit_id,human_verdict,human_applicable_rule_ids,human_violated_rule_ids`
- preference：`audit_id,human_best_trajectory,human_ranking`

随后生成带全部源文件 SHA-256 的正式场景清单：

```bash
uv run python -m experiments.ablation_2x2.build_gold_scenarios \
  --verdict-adjudicated human_annotation_v2/results/adjudicated_verdict.csv \
  --preference-adjudicated human_annotation_v2/results/adjudicated_preference.csv \
  --output experiments/ablation_2x2/units/human_gold_82.json
```

构建器会硬校验 200/78/36/32/4/82 口径、public/private sidecar 对齐、候选映射、
偏好全排序以及每个输入文件的哈希；任一不一致即停止。

## 正式运行（gold 冻结后）

场景清单必须使用 `ablation_2x2_v1.0` 协议、包含 `gold_finalized: true` 及最终
`gold`。确认 API 配额后显式运行：

```bash
uv run python -m experiments.ablation_2x2.run_ablation \
  --backend live \
  --confirm-gold-final \
  --scenarios experiments/ablation_2x2/units/human_gold_82.json \
  --output-dir results/ablation_2x2 \
  --artifact-node /path/to/bundled/node \
  --artifact-node-modules /path/to/bundled/node_modules
```

正式与 dry-run 目录、manifest 标记不能混用。四条件串行执行。输出包含全局和条件
manifest、独立 `records.jsonl`、六份 prompt diff、引用评级工作簿、`metrics.json`
和配对 `comparison_report.md`。

人工填写 `citation_grading_sheet.xlsx` 的三个评级列后，无需重跑 LLM：

```bash
uv run python -m experiments.ablation_2x2.score_ablation \
  --output-dir results/ablation_2x2 \
  --workbook results/ablation_2x2/citation_grading_sheet.xlsx \
  --artifact-node /path/to/bundled/node \
  --artifact-node-modules /path/to/bundled/node_modules
```

重评分只在内存中合并人工评级并覆盖派生的 metrics/report；四条件原始 JSONL 永不改写。

## 盲态先推理（gold 尚在仲裁时）

如果两名 verdict 已回收，但 IAA/仲裁或 preference 尚未完成，可以先冻结同一个
82 场输入并执行模型生成。该路径不写临时 gold、不计算人工一致性指标，并强制使用
独立的 `BLINDED_INFERENCE` 目录；只有在正式 gold 冻结并完成评分后才能发布指标。

```bash
uv run python -m experiments.ablation_2x2.build_inference_scenarios \
  --output experiments/ablation_2x2/units/blinded_inference_82.json

uv run python -m experiments.ablation_2x2.run_ablation \
  --backend live \
  --blinded-inference \
  --workers 4 \
  --scenarios experiments/ablation_2x2/units/blinded_inference_82.json \
  --output-dir results/ablation_2x2/BLINDED_INFERENCE \
  --artifact-node /path/to/bundled/node \
  --artifact-node-modules /path/to/bundled/node_modules
```
