"""Prompt loader tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.layer3.prompt_loader import load_few_shot_examples

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEW_SHOT_DIR = _REPO_ROOT / "src" / "layer3" / "prompts" / "few_shots"


def test_few_shot_files_are_valid_json():
    for filename in [
        "hard_filter_examples.json",
        "preference_ranker_examples.json",
    ]:
        data = json.loads((_FEW_SHOT_DIR / filename).read_text(encoding="utf-8"))
        assert data["examples"]


def test_few_shot_loader_preserves_raw_json_examples():
    hard_filter = load_few_shot_examples("hard_filter")
    preference_ranker = load_few_shot_examples("preference_ranker")

    assert hard_filter.lstrip().startswith("{")
    assert preference_ranker.lstrip().startswith("{")
    assert '"results"' in hard_filter
    assert '"preferred_id"' in preference_ranker
    assert "示例 1:" not in hard_filter
    assert "'preferred_id'" not in preference_ranker
