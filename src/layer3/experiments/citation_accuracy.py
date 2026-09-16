"""Citation accuracy 实验指标。"""

from __future__ import annotations

from collections.abc import Mapping


def citation_accuracy(
    predicted: Mapping[str, set[str]],
    expected: Mapping[str, set[str]],
) -> float:
    """计算轨迹级 cited rule 与人工标注的精确匹配率。"""
    if not expected:
        return 0.0
    correct = 0
    for trajectory_id, expected_ids in expected.items():
        if predicted.get(trajectory_id, set()) == expected_ids:
            correct += 1
    return correct / len(expected)
