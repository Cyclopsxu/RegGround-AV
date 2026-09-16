"""运行汇总与断点写入的非 API 回归测试。"""

from src.audit_run_logger import (
    _existing_entries,
    _write_jsonl,
    _write_record_once,
    llm_settings_log,
    summarize_entries,
)
from src.layer3.models import Layer3Settings


def _entry() -> dict:
    return {
        "ok": True,
        "layer1": {
            "context": {"scenario_type": "red_light"},
            "benchmark_labels_debug_only": {"labels": []},
        },
        "layer2": {"retrieval_mode": "graph"},
        "final_label": {
            "status": "complete",
            "selection_outcome": "no_choosable_candidate",
            "chosen_trajectory_id": None,
            "preference_ranking": [],
            "verdicts": [{
                "trajectory_id": "traj_a",
                "veto": {"status": "vetoed", "decided_by": "rule_engine"},
            }],
            "stage_diagnostics": [],
        },
    }


def test_summary_omits_circular_accuracy_and_keeps_zero_telemetry_fields() -> None:
    summary = summarize_entries([_entry()])

    assert "candidate_verdict_accuracy" not in summary
    assert summary["selection_outcomes"] == {"no_choosable_candidate": 1}
    assert summary["llm_telemetry"]["total"] == {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "retries": 0,
        "deferred_attempts": 0,
        "recovered_count": 0,
    }
    assert summary["llm_telemetry"]["connection_events"] == {}
    assert summary["llm_telemetry"]["deferred_recovered"] == 0
    assert summary["llm_telemetry"]["cost_status"] == "pricing_not_configured"
    assert "estimated_cost_usd" not in summary["llm_telemetry"]


def test_summary_estimates_cost_only_with_both_prices() -> None:
    entry = _entry()
    entry["final_label"]["stage_diagnostics"] = [{
        "stage": "hard_filter",
        "llm_call_count": 1,
        "llm_input_tokens": 1_000_000,
        "llm_output_tokens": 500_000,
        "llm_retry_count": 0,
    }]

    summary = summarize_entries(
        [entry],
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=2.0,
    )

    assert summary["llm_telemetry"]["cost_status"] == "estimated"
    assert summary["llm_telemetry"]["estimated_cost_usd"] == 2.0


def test_record_writer_is_idempotent_by_frame_token(tmp_path) -> None:
    output = tmp_path / "audit.jsonl"
    entry = _entry()
    entry["index"] = 4
    entry["raw_scene"] = {"frame_token": "frame-4"}

    assert _write_record_once(output, entry)
    assert not _write_record_once(output, entry)

    entries, completed = _existing_entries(output)
    assert entries == [entry]
    assert completed == {"frame:frame-4"}


def test_existing_entries_ignores_legacy_duplicate_units(tmp_path) -> None:
    output = tmp_path / "audit.jsonl"
    first = _entry()
    first["index"] = 3
    first["raw_scene"] = {"frame_token": "frame-3"}
    duplicate = {**first, "duration_ms": 999}
    _write_jsonl(output, {"type": "record", "data": first})
    _write_jsonl(output, {"type": "record", "data": duplicate})

    entries, completed = _existing_entries(output)
    assert entries == [first]
    assert completed == {"frame:frame-3"}


def test_existing_entries_tolerates_truncated_tail(tmp_path) -> None:
    output = tmp_path / "audit.jsonl"
    entry = _entry()
    entry["index"] = 5
    entry["raw_scene"] = {"frame_token": "frame-5"}
    _write_jsonl(output, {"type": "record", "data": entry})
    with output.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"record","data":')

    entries, completed = _existing_entries(output)

    assert entries == [entry]
    assert completed == {"frame:frame-5"}


def test_llm_settings_log_freezes_deepseek_runtime_parameters() -> None:
    logged = llm_settings_log(Layer3Settings())

    assert logged["llm_provider"] == "openai_compatible"
    assert logged["llm_model"] == "deepseek-v4-pro"
    assert logged["llm_model_version"] == "DeepSeek-V4-Pro"
    assert logged["llm_thinking_mode"] == "enabled"
    assert logged["llm_timeout_seconds"] == 180.0
    assert logged["llm_retry_backoff_base_seconds"] == 3.0
    assert logged["llm_retry_backoff_multiplier"] == 4.0
    assert logged["hard_filter_temperature"] == 0.0
    assert logged["pairwise_ranker_temperature"] == 0.0
    assert logged["label_temperature"] == 0.2
    assert "llm_api_key" not in logged


def test_summary_tracks_deferred_recovery_and_connection_events() -> None:
    entry = _entry()
    entry["final_label"]["stage_diagnostics"] = [{
        "stage": "hard_filter",
        "llm_call_count": 2,
        "llm_input_tokens": 10,
        "llm_output_tokens": 5,
        "llm_retry_count": 1,
        "deferred_attempts": [{"round_index": 1}],
        "recovered_count": 1,
    }]
    entry["final_label"]["connection_events"] = [{"kind": "timeout"}]
    entry["final_label"]["partial_reasons"] = ["stage_degraded"]

    summary = summarize_entries([entry])

    assert summary["llm_telemetry"]["total"]["deferred_attempts"] == 1
    assert summary["llm_telemetry"]["total"]["recovered_count"] == 1
    assert summary["llm_telemetry"]["deferred_recovered"] == 1
    assert summary["llm_telemetry"]["connection_events"] == {"timeout": 1}
    assert summary["degraded_records"] == 1
    assert summary["degraded_no_choosable_candidates"] == 1
