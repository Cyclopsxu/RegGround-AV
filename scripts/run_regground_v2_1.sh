#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

eval_set_path="artifacts/eval_set_v1.json"
output_path="${1:-logs/eval_set_v1.jsonl}"

if [[ ! -f "$eval_set_path" ]]; then
  echo "缺少已冻结的 ${eval_set_path}；先完成 spike 审阅并以 --freeze 生成评估集。" >&2
  exit 1
fi

: "${LAYER3_LLM_API_KEY:?请先设置 LAYER3_LLM_API_KEY}"

.venv/bin/python -m src.audit_run_logger \
  --input data/layer1/raw_scene_trainval.json \
  --eval-set "$eval_set_path" \
  --output "$output_path" \
  --seed 42 \
  --serial

.venv/bin/python -m src.evaluation_report "$output_path" --benchmark-version v2
