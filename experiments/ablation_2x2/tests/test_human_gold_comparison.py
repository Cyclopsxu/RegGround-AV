from __future__ import annotations

import pytest

from experiments.ablation_2x2.human_gold_comparison import (
    _gold_rule_alternatives,
    classification_metrics,
    factorial_effects,
)


def _row(
    gold: str,
    full: str,
    no_validation: str,
    no_rag: str,
    baseline: str,
    *,
    frame: str,
) -> dict:
    return {
        "frame_token": frame,
        "gold_verdict": gold,
        "predictions": {
            "full": {"status": full},
            "no_validation": {"status": no_validation},
            "no_rag": {"status": no_rag},
            "baseline": {"status": baseline},
        },
    }


def test_gold_rule_alternatives_preserve_annotator_disjunction() -> None:
    assert _gold_rule_alternatives("R-YLD-01 | ymh:R-SIG-01") == [
        frozenset({"R-YLD-01"}),
        frozenset({"R-SIG-01"}),
    ]


def test_classification_metrics_use_three_class_macro_f1() -> None:
    rows = [
        _row("cleared", "cleared", "cleared", "cleared", "cleared", frame="a"),
        _row("vetoed", "vetoed", "vetoed", "cleared", "cleared", frame="b"),
        _row("uncertain", "cleared", "cleared", "cleared", "cleared", frame="c"),
    ]
    metrics = classification_metrics(rows, "full")
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["vetoed"]["recall"] == 1.0
    assert metrics["per_class"]["uncertain"]["f1"] == 0.0
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)


def test_factorial_effects_follow_frozen_condition_matrix() -> None:
    rows = [
        _row("cleared", "cleared", "cleared", "vetoed", "vetoed", frame="a"),
        _row("vetoed", "vetoed", "cleared", "vetoed", "cleared", frame="b"),
    ]
    effects = factorial_effects(rows)
    assert effects["rag_main_effect"]["accuracy_difference"] == pytest.approx(0.5)
    assert effects["validation_main_effect"]["accuracy_difference"] == pytest.approx(0.5)
    assert effects["rag_x_validation_interaction"]["accuracy_difference"] == pytest.approx(0.0)
