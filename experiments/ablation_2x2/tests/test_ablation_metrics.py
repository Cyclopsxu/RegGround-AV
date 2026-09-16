from __future__ import annotations

import json

from experiments.ablation_2x2.conditions import CONDITIONS
from experiments.ablation_2x2.metrics import cohen_kappa, kendall_tau
from experiments.ablation_2x2.run_ablation import DEFAULT_SCENARIOS, DryRunBackend, run


def test_known_kappa_and_tau_values() -> None:
    assert (
        cohen_kappa(["cleared", "vetoed", "uncertain"], ["cleared", "vetoed", "uncertain"]) == 1.0
    )
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_dry_gold_produces_complete_metrics_and_paired_report(tmp_path) -> None:
    output = tmp_path / "DRY_RUN"
    assert run(
        scenario_path=DEFAULT_SCENARIOS,
        output_dir=output,
        dry_run=True,
        confirm_gold_final=False,
        backend=DryRunBackend(),
        skip_workbook=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["conditions"]) == {item.name for item in CONDITIONS}
    for values in metrics["conditions"].values():
        assert {
            "verdict_kappa",
            "kendall_tau",
            "citation_validity_first",
            "citation_validity_final",
            "hallucination_rate",
            "citations_per_decision",
        } <= set(values)
    report = (output / "comparison_report.md").read_text(encoding="utf-8")
    assert "配对逐场景明细" in report
    assert "Small-n caveat" in report
