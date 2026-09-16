"""benchmark_e2e.py —— 端到端基准测试。

对主实验集跑完整 judge，输出 metrics.json（P50/P95 耗时、幻觉率等）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# 基准测试占位——完整数据集运行在论文实验阶段
# 当前仅提供框架结构，具体 metric 收集逻辑在论文实验阶段完善


def run_benchmark(scenes: list, subgraphs: list, output_dir: Path) -> dict:
    """运行端到端基准测试。

    Args:
        scenes: SceneContext 列表。
        subgraphs: RuleSubgraph 列表（与 scenes 一一对应）。
        output_dir: 输出目录。

    Returns:
        metrics dict。
    """
    from src.layer3 import AuditJudge
    from src.layer3.models import Layer3Settings

    settings = Layer3Settings()
    judger = AuditJudge(settings)

    durations: list[int] = []
    labels: list[dict] = []

    for scene, subgraph in zip(scenes, subgraphs, strict=False):
        t0 = time.perf_counter()
        label = judger.judge(scene, subgraph)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        durations.append(dur_ms)
        labels.append(label.model_dump(mode="json"))

    durations_sorted = sorted(durations)

    metrics = {
        "n_scenes": len(scenes),
        "total_duration_seconds": sum(durations) / 1000,
        "duration_p50_ms": _percentile(durations_sorted, 50),
        "duration_p95_ms": _percentile(durations_sorted, 95),
        "durations": durations,
        "labels": labels,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    return metrics


def _percentile(sorted_data: list[int], p: int) -> int:
    """计算百分位数。"""
    if not sorted_data:
        return 0
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


if __name__ == "__main__":
    # 占位：运行时需要提供真实数据集
    print("Benchmark entry point — requires real dataset and API keys")
