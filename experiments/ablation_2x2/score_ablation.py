"""Re-score saved records after human citation ratings are completed."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.conditions import CONDITIONS
from experiments.ablation_2x2.metrics import write_metrics

MODULE_DIR = Path(__file__).resolve().parent
RATING_EXPORTER = MODULE_DIR / "export_citation_ratings.mjs"
_RATED_VALUES = {"是", "否", "不确定"}


def _load_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = output_dir / condition.name / "records.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing condition records: {path}")
        result[condition.name] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    return result


def _read_workbook_ratings(
    workbook_path: Path,
    output_dir: Path,
    *,
    artifact_node: Path,
    artifact_node_modules: Path,
) -> list[dict[str, Any]]:
    if not artifact_node.is_file() or not artifact_node_modules.is_dir():
        raise FileNotFoundError("artifact-tool Node runtime paths are invalid")
    runtime = Path(tempfile.mkdtemp(prefix="ablation_xlsx_read_", dir=output_dir))
    ratings_path = runtime / "ratings.json"
    try:
        (runtime / "node_modules").symlink_to(artifact_node_modules, target_is_directory=True)
        runtime_exporter = runtime / RATING_EXPORTER.name
        shutil.copy2(RATING_EXPORTER, runtime_exporter)
        subprocess.run(
            [str(artifact_node), str(runtime_exporter), str(workbook_path), str(ratings_path)],
            check=True,
            cwd=runtime,
        )
        return json.loads(ratings_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(runtime)


def apply_ratings(
    records_by_condition: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return a scored copy; immutable raw JSONL records remain untouched."""
    ratings = {str(row.get("citation_id", "")): row for row in rows if row.get("citation_id")}
    scored = json.loads(json.dumps(records_by_condition, ensure_ascii=False))
    for condition, records in scored.items():
        free_text_track = condition in {"no_rag", "baseline"}
        for record in records:
            for phase in ("citations_first", "citations_final"):
                for event in record[phase]:
                    citation_id = event["citation_id"]
                    if citation_id not in ratings:
                        raise ValueError(f"missing human citation rating: {citation_id}")
                    rating = ratings[citation_id]
                    fields = (
                        "existence_rating",
                        "content_fidelity_rating",
                        "applicability_rating",
                    )
                    if any(str(rating.get(field, "")) not in _RATED_VALUES for field in fields):
                        raise ValueError(f"unfinished human citation rating: {citation_id}")
                    if free_text_track:
                        event["valid"] = (
                            rating["existence_rating"] == "是"
                            and rating["content_fidelity_rating"] == "是"
                        )
                    event["accuracy"] = rating["applicability_rating"] == "是"
    return scored


def score(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = apply_ratings(_load_records(output_dir), rows)
    dry_run = bool(next(iter(records.values()))[0].get("dry_run"))
    return write_metrics(output_dir, records, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score ablation records with human ratings")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook", type=Path)
    source.add_argument("--ratings-json", type=Path)
    parser.add_argument("--artifact-node", type=Path)
    parser.add_argument("--artifact-node-modules", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.ratings_json:
        rows = json.loads(args.ratings_json.read_text(encoding="utf-8"))
    else:
        if args.artifact_node is None or args.artifact_node_modules is None:
            parser.error("--workbook requires --artifact-node and --artifact-node-modules")
        rows = _read_workbook_ratings(
            args.workbook.resolve(),
            output_dir,
            artifact_node=args.artifact_node.resolve(),
            artifact_node_modules=args.artifact_node_modules.resolve(),
        )
    score(output_dir, rows)
    print(f"updated metrics and report in {output_dir}")


if __name__ == "__main__":
    main()
