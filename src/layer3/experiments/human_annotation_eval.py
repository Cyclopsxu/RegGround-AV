"""人工标注子集一致性指标。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations


def preference_pair_accuracy(
    predicted: set[tuple[str, str]],
    expected: set[tuple[str, str]],
) -> float:
    if not expected:
        return 0.0
    return len(predicted & expected) / len(expected)


def kendall_tau(predicted_ranking: Sequence[str], expected_ranking: Sequence[str]) -> float:
    common = [item for item in expected_ranking if item in set(predicted_ranking)]
    if len(common) < 2:
        return 0.0
    pred_pos = {item: idx for idx, item in enumerate(predicted_ranking)}
    exp_pos = {item: idx for idx, item in enumerate(expected_ranking)}
    concordant = 0
    discordant = 0
    for left, right in combinations(common, 2):
        pred_order = pred_pos[left] < pred_pos[right]
        exp_order = exp_pos[left] < exp_pos[right]
        if pred_order == exp_order:
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def cohens_kappa(
    predicted: Sequence[bool],
    expected: Sequence[bool],
) -> float:
    if len(predicted) != len(expected) or not expected:
        return 0.0
    total = len(expected)
    observed = sum(p == e for p, e in zip(predicted, expected, strict=True)) / total
    pred_positive = sum(predicted) / total
    exp_positive = sum(expected) / total
    pred_negative = 1.0 - pred_positive
    exp_negative = 1.0 - exp_positive
    chance = pred_positive * exp_positive + pred_negative * exp_negative
    if chance == 1.0:
        return 1.0
    return (observed - chance) / (1.0 - chance)
