from __future__ import annotations

import json

import pytest

from experiments.ablation_2x2.run_ablation import DEFAULT_SCENARIOS, DryRunBackend, run
from experiments.ablation_2x2.score_ablation import apply_ratings


def test_human_ratings_are_applied_without_mutating_raw_records(tmp_path) -> None:
    output = tmp_path / "DRY_RUN"
    run(
        scenario_path=DEFAULT_SCENARIOS,
        output_dir=output,
        dry_run=True,
        confirm_gold_final=False,
        backend=DryRunBackend(),
        skip_workbook=True,
    )
    records = {
        name: [
            json.loads(line)
            for line in (output / name / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("full", "no_validation", "no_rag", "baseline")
    }
    original = json.loads(json.dumps(records))
    rows = json.loads((output / "citation_grading_rows.json").read_text(encoding="utf-8"))
    scored = apply_ratings(records, rows)

    assert records == original
    assert scored["baseline"][0]["citations_final"][0]["accuracy"] is False


def test_unfinished_human_rating_is_rejected(tmp_path) -> None:
    output = tmp_path / "DRY_RUN"
    run(
        scenario_path=DEFAULT_SCENARIOS,
        output_dir=output,
        dry_run=True,
        confirm_gold_final=False,
        backend=DryRunBackend(),
        skip_workbook=True,
    )
    records = {
        name: [
            json.loads(line) for line in (output / name / "records.jsonl").read_text().splitlines()
        ]
        for name in ("full", "no_validation", "no_rag", "baseline")
    }
    rows = json.loads((output / "citation_grading_rows.json").read_text(encoding="utf-8"))
    rows[0]["applicability_rating"] = "待评级"
    with pytest.raises(ValueError, match="unfinished"):
        apply_ratings(records, rows)
