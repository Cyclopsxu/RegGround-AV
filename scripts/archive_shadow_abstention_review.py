"""归档 v_a 红绿灯弃权的逐条语义复核证据链。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/shadow_no_rule_engine_v2/shadow_verdicts.jsonl"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/shadow_no_rule_engine_v2_vb_phase_continuity"
    / "evidence_chain"
)
SAMPLE_SEED = 42
SEMANTIC_CATEGORY = "crossing_phase_at_decision_time_unverifiable"


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_hash(record: dict[str, Any]) -> str:
    value = (
        f"{SAMPLE_SEED}:{record['frame_token']}:{record['trajectory_id']}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_semantic_match(reason: str) -> None:
    phase_cues = (
        "相位未知",
        "相位在t>0后未知",
        "后续相位",
        "其后相位",
        "信号变化时刻",
        "后续变化未知",
        "后续信号相位信息",
        "unknown thereafter",
        "signal turned green",
    )
    crossing_cues = (
        "越过停止线",
        "越线",
        "进入停止线区域",
        "红灯期间",
        "红灯相位下",
        "crosses the stop line area",
    )
    uncertainty_cues = ("无法", "不足", "不明确", "unclear", "Insufficient")
    if not (
        any(cue in reason for cue in phase_cues)
        and any(cue in reason for cue in crossing_cues)
        and any(cue in reason for cue in uncertainty_cues)
    ):
        raise ValueError(f"reason 未通过既定语义口径，需人工复核：{reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    records = _load_records(source)
    population = [
        record
        for record in records
        if record["scenario_type"] == "red_light"
        and record["shadow_status"] == "uncertain"
    ]
    if len(population) != 33:
        raise ValueError(f"v_a 红绿灯弃权总体应为 33 条，实际 {len(population)}")

    ranked = sorted(population, key=_sample_hash)
    sample_ranks = {
        (record["frame_token"], record["trajectory_id"]): rank
        for rank, record in enumerate(ranked[:10], start=1)
    }
    reviewed: list[dict[str, Any]] = []
    for record in sorted(
        population,
        key=lambda item: (item["frame_token"], item["trajectory_id"]),
    ):
        reason = record["shadow_reason"]
        _assert_semantic_match(reason)
        key = (record["frame_token"], record["trajectory_id"])
        reviewed.append(
            {
                "frame_token": record["frame_token"],
                "trajectory_id": record["trajectory_id"],
                "predicate_label": record["predicate_label"],
                "predicate_reason": record["predicate_reason"],
                "shadow_status": record["shadow_status"],
                "shadow_reason_original": reason,
                "semantic_category": SEMANTIC_CATEGORY,
                "semantic_match": True,
                "semantic_review": (
                    "弃权理由将关键证据缺口归因于越线或进入停止线区域时的"
                    "信号相位无法确认。"
                ),
                "quote_sample_sha256": _sample_hash(record),
                "selected_for_quote_sample": key in sample_ranks,
                "quote_sample_rank": sample_ranks.get(key),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "red_light_abstention_review.jsonl"
    with review_path.open("w", encoding="utf-8") as handle:
        for record in reviewed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "source": str(source),
        "source_sha256": source_sha256,
        "population_definition": (
            "v_a 全量记录中 scenario_type=red_light 且 shadow_status=uncertain；"
            "这是 33 条总体的全数复核，不是抽样。"
        ),
        "population_count": len(reviewed),
        "semantic_category": SEMANTIC_CATEGORY,
        "semantic_match_count": sum(r["semantic_match"] for r in reviewed),
        "quote_sample": {
            "count": 10,
            "method": (
                "按 SHA-256('42:' + frame_token + ':' + trajectory_id) 升序取前 10 条"
            ),
            "seed": SAMPLE_SEED,
        },
        "archived_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# v_a 红绿灯弃权复核证据链\n\n"
        "本目录归档 v_a 中全部 33 条红绿灯 `uncertain`，逐条保留 reason 原文并"
        "按 `crossing_phase_at_decision_time_unverifiable` 归类。这里是总体全数复核；"
        "此前展示的 10 条原文按 manifest 所记 SHA-256 顺序确定性抽取。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
