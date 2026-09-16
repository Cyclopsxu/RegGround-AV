"""ContextBuilder v3.0 测试。"""

import pytest

from src.layer1.models import JudgeInput
from src.layer3.context_builder import ContextBuilder
from src.layer3.exceptions import StructuredOutputValidationError


def test_build_from_judge_input(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    ctx = ContextBuilder(layer3_settings).build(sample_judge_input, sample_rule_subgraph)

    assert ctx.scene_id == "scene_001"
    assert ctx.frame_token == "frame_001"
    assert ctx.trajectory_ids == ["traj_a", "traj_b", "traj_c"]
    assert ctx.hard_rule_ids == ["R-SIG-01"]
    assert ctx.soft_rule_ids == ["R-SIG-03"]
    assert "逐步减速" in ctx.trajectories_text
    assert "信号灯相位为 keyframe 时刻（t=0）快照，其后相位未知" in (
        ctx.scene_facts_text
    )


def test_context_does_not_include_raw_coordinates(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    ctx = ContextBuilder(layer3_settings).build(sample_judge_input, sample_rule_subgraph)
    prompt_text = "\n".join(
        [
            ctx.scene_narrative,
            ctx.hard_rules_text,
            ctx.soft_rules_text,
            ctx.trajectories_text,
        ]
    )

    assert "candidate_trajectories" not in prompt_text
    assert "waypoints" not in prompt_text
    assert '"x":' not in prompt_text
    assert '"y":' not in prompt_text


def test_trajectory_truncation(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    settings = layer3_settings.model_copy(update={"max_trajectories_per_judge": 2})
    ctx = ContextBuilder(settings).build(sample_judge_input, sample_rule_subgraph)
    assert ctx.trajectory_ids == ["traj_a", "traj_b"]


def test_prompt_leakage_is_rejected(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    leaked = JudgeInput(
        scene_id=sample_judge_input.scene_id,
        frame_token=sample_judge_input.frame_token,
        narrative="this contains ground_truth metadata",
        scene_facts_digest=sample_judge_input.scene_facts_digest,
        trajectory_features=sample_judge_input.trajectory_features,
    )
    with pytest.raises(StructuredOutputValidationError):
        ContextBuilder(layer3_settings).build(leaked, sample_rule_subgraph)
