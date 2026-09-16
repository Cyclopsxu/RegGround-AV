"""LabelGenerator v3.0 测试。"""

import pytest

from src.layer3.exceptions import LLMTimeoutError
from src.layer3.label_generator import LabelGenerator
from src.layer3.models import (
    DecisionSource,
    PairwisePreference,
    TrajectoryVerdict,
    VetoResult,
    VetoStatus,
)


class FailingTextLLM:
    def complete_structured(self, *args, **kwargs):
        return None, 0

    def complete_text(self, *args, **kwargs):
        raise LLMTimeoutError("timeout")


class StaticTextLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete_structured(self, *args, **kwargs):
        return None, 0

    def complete_text(self, *args, **kwargs):
        return self.text, 10


def _verdicts() -> list[TrajectoryVerdict]:
    return [
        TrajectoryVerdict(
            trajectory_id="traj_a",
            veto=VetoResult(
                trajectory_id="traj_a",
                status=VetoStatus.CLEARED,
                reason="stopped",
                decided_by=DecisionSource.RULE_ENGINE,
            ),
            rank=1,
        ),
        TrajectoryVerdict(
            trajectory_id="traj_b",
            veto=VetoResult(
                trajectory_id="traj_b",
                status=VetoStatus.UNCERTAIN,
                reason="slow rolling",
                decided_by=DecisionSource.FALLBACK,
            ),
        ),
        TrajectoryVerdict(
            trajectory_id="traj_c",
            veto=VetoResult(
                trajectory_id="traj_c",
                status=VetoStatus.VETOED,
                violated_rule_ids=["R-SIG-01"],
                reason="high speed",
                decided_by=DecisionSource.RULE_ENGINE,
            ),
        ),
    ]


def test_fallback_summary_mentions_uncertain(
    layer3_settings,
):
    generator = LabelGenerator(layer3_settings, StaticTextLLM(""))
    summary = generator.fallback_summary(_verdicts(), [])

    assert "结论" in summary
    assert "否决说明" in summary
    assert "不确定说明" in summary
    assert "traj_b" in summary
    assert "不作为 chosen" in summary
    assert "[R-SIG-01]" in summary


def test_generate_returns_llm_text(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    text = "结论：chosen 为 traj_a。\n不确定说明：traj_b 未闭环。"
    generator = LabelGenerator(layer3_settings, StaticTextLLM(text))
    summary = generator.generate(
        sample_judge_input,
        sample_rule_subgraph,
        _verdicts(),
        [],
        [],
    )
    assert summary == text


def test_generate_propagates_llm_error(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    generator = LabelGenerator(layer3_settings, FailingTextLLM())
    with pytest.raises(LLMTimeoutError):
        generator.generate(
            sample_judge_input,
            sample_rule_subgraph,
            _verdicts(),
            [],
            [],
        )


def test_fallback_summary_reports_pairwise_basis(layer3_settings):
    generator = LabelGenerator(layer3_settings, StaticTextLLM(""))
    pairs = [
        PairwisePreference(
            preferred_id="traj_a",
            dispreferred_id="traj_d",
            basis="pairwise_judgment",
            confidence=0.8,
            reasoning="larger margin",
            referenced_rule_ids=["R-SIG-03"],
            decided_by=DecisionSource.LLM,
        )
    ]
    summary = generator.fallback_summary(_verdicts(), pairs)
    assert "偏好说明" in summary
    assert "[R-SIG-03]" in summary
