"""正式评估集冻结与运行输入装载的回归测试。"""

from pathlib import Path
from typing import Literal

import pytest

from src.audit_run_logger import select_frozen_eval_records
from src.eval_set import (
    EvalSet,
    EvalSetEntry,
    assert_classification_code_matches,
    assert_input_file_matches,
    classification_code_sha256,
    classification_source_files,
    eval_set_json_bytes,
    load_eval_set,
    write_eval_set,
)


def _eval_set(
    *,
    state: Literal["draft", "frozen"] = "frozen",
    determinism_verified: bool = True,
    classification_hash: str | None = None,
    classification_files: list[str] | None = None,
) -> EvalSet:
    return EvalSet(
        state=state,  # type: ignore[arg-type]
        entries=[EvalSetEntry(
            frame_token="frame_1",
            scene_token="scene_1",
            scenario_type="red_light",
            horizon_displacement_m=7.5,
            timestamp=1,
        )],
        mini_scene_tokens=[f"mini_{index}" for index in range(10)],
        generation_script="src/layer1/availability_spike.py",
        generation_commit="abc123",
        spike_report="artifacts/trainval_availability_spike.md",
        input_source="data/layer1/raw_scene_trainval.json",
        input_sha256="0" * 64,
        input_provenance="prepared input is the formal-run baseline",
        classification_source_files=classification_files or classification_source_files(),
        classification_code_sha256=classification_hash or classification_code_sha256(),
        determinism_verified=determinism_verified,
    )


def test_eval_set_round_trip_is_stable_and_requires_frozen_state(tmp_path: Path) -> None:
    path = tmp_path / "eval_set_v1.json"
    write_eval_set(path, _eval_set())
    first = path.read_bytes()
    write_eval_set(path, _eval_set())

    assert path.read_bytes() == first
    assert load_eval_set(path, require_frozen=True).state == "frozen"

    write_eval_set(path, _eval_set(state="draft"))
    with pytest.raises(ValueError, match="draft"):
        load_eval_set(path, require_frozen=True)


def test_frozen_eval_set_requires_double_run_and_matching_classification_code() -> None:
    with pytest.raises(ValueError, match="byte-identical"):
        eval_set_json_bytes(_eval_set(determinism_verified=False))

    assert_classification_code_matches(_eval_set())
    with pytest.raises(ValueError, match="classification/admission"):
        assert_classification_code_matches(_eval_set(classification_hash="different"))
    with pytest.raises(ValueError, match="source file list"):
        assert_classification_code_matches(_eval_set(classification_files=["src/helper.py"]))


def test_frozen_eval_set_rejects_input_sha_or_missing_token(tmp_path: Path) -> None:
    input_path = tmp_path / "raw_scene.json"
    input_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        assert_input_file_matches(_eval_set(), input_path)


def test_select_frozen_eval_records_follows_frozen_order_and_validates_scene() -> None:
    records = [
        {"frame_token": "frame_2", "scene_id": "scene_2"},
        {"frame_token": "frame_1", "scene_id": "scene_1"},
    ]

    assert select_frozen_eval_records(records, _eval_set()) == [records[1]]

    with pytest.raises(ValueError, match="does not match"):
        select_frozen_eval_records(
            [{"frame_token": "frame_1", "scene_id": "other_scene"}],
            _eval_set(),
        )
    with pytest.raises(ValueError, match="missing"):
        select_frozen_eval_records([], _eval_set())
