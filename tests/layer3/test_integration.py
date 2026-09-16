"""Layer 1/2/3 窄接口集成测试。"""

from src.layer1.interface_adapter import InterfaceAdapter
from src.layer1.models import JudgeInput
from src.layer2.models import RuleSubgraph
from src.layer3.hard_filter import _LLMFilterResponse
from src.layer3.judge import AuditJudge
from src.layer3.models import AuditLabel, ReportStatus
from src.layer3.preference_ranker import _LLMPairwiseResponse
from tests.layer3.test_judge import FakeJudgeLLM


def test_layer1_adapter_to_layer3_judge_input(sample_scene_context):
    judge_input = InterfaceAdapter().to_judge_input(sample_scene_context)
    assert isinstance(judge_input, JudgeInput)
    assert judge_input.trajectory_features
    assert not hasattr(judge_input, "candidate_trajectories")


def test_layer3_consumes_judge_input_and_rule_subgraph(
    layer3_settings,
    sample_scene_context,
    sample_rule_subgraph,
):
    judge_input = InterfaceAdapter().to_judge_input(sample_scene_context)
    assert isinstance(sample_rule_subgraph, RuleSubgraph)

    llm = FakeJudgeLLM(
        filter_response=_LLMFilterResponse(
            results=[
                {
                    "trajectory_id": "traj_b",
                    "status": "cleared",
                    "violated_rule_ids": [],
                    "reason": "slow roll resolved",
                }
            ]
        ),
        pairwise_response=_LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.8,
            reasoning="larger safety margin",
            referenced_rule_ids=["R-SIG-03"],
        )
    )
    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        judge_input,
        sample_rule_subgraph,
    )

    assert isinstance(label, AuditLabel)
    assert label.status in (ReportStatus.COMPLETE, ReportStatus.PARTIAL)
    assert label.frame_token == "frame_001"
    assert set(label.preference_ranking) == {"traj_a", "traj_b", "traj_c"}
