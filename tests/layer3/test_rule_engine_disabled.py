"""关闭规则引擎的配置，以及证据区拼装的单一事实源。"""

from scripts.run_shadow_no_rule_engine import (
    BASE_PHASE_SNAPSHOT_LINE,
    apply_phase_continuity_assumption,
    phase_continuity_statement,
)
from src.layer3.hard_filter import HardFilter, _FilterItem, _LLMFilterResponse
from src.layer3.judge import AuditJudge
from src.layer3.models import DecisionSource, Layer3Settings, VetoStatus
from src.layer3.rule_engine import DisabledRuleEngine, RuleEngine, build_rule_engine
from src.replay_context import ReplayedScene


class CapturingLLM:
    """记录收到的 user_prompt，并对每个候选回一个固定判定。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_structured(self, *args, **kwargs):
        prompt = kwargs["user_prompt"]
        self.prompts.append(prompt)
        trajectory_ids = [
            line.split("]")[0].lstrip("[")
            for line in prompt.splitlines()
            if line.startswith("[traj_")
        ]
        return (
            _LLMFilterResponse(
                results=[
                    _FilterItem(
                        trajectory_id=trajectory_id,
                        status=VetoStatus.CLEARED,
                        violated_rule_ids=[],
                        reason="fake",
                    )
                    for trajectory_id in trajectory_ids
                ]
            ),
            12,
        )

    def complete_text(self, *args, **kwargs):
        return "", 0


def test_build_rule_engine_honours_disabled_flag() -> None:
    assert isinstance(build_rule_engine(), RuleEngine)
    assert isinstance(build_rule_engine(disabled=True), DisabledRuleEngine)


def test_disabled_engine_never_decides(sample_judge_context, sample_features) -> None:
    engine = DisabledRuleEngine()

    for index, feature in enumerate(sample_features):
        decision = engine.check(feature, sample_judge_context, f"traj_{index}")
        assert not decision.is_decisive
        assert decision.status == VetoStatus.UNCERTAIN


def test_disabled_engine_routes_every_candidate_to_llm(
    sample_judge_context, sample_features
) -> None:
    """同一批候选在开启时部分被直裁，关闭后必须全部走 LLM。"""
    settings = Layer3Settings()
    baseline = HardFilter(settings, CapturingLLM(), rule_engine=RuleEngine())
    baseline_results, _ = baseline.filter(sample_judge_context, sample_features)
    assert any(
        result.decided_by == DecisionSource.RULE_ENGINE for result in baseline_results
    )

    shadow_llm = CapturingLLM()
    shadow = HardFilter(settings, shadow_llm, rule_engine=DisabledRuleEngine())
    shadow_results, _ = shadow.filter(sample_judge_context, sample_features)

    assert all(
        result.decided_by == DecisionSource.LLM for result in shadow_results
    )
    assert len(shadow_llm.prompts) == len(sample_features)


def test_settings_flag_reaches_the_hard_filter() -> None:
    judge = AuditJudge(Layer3Settings(disable_rule_engine=True))

    assert isinstance(judge._hard_filter._engine, DisabledRuleEngine)  # noqa: SLF001


def test_evidence_prompt_is_the_prefix_actually_sent(
    sample_judge_context, sample_features
) -> None:
    """证据区是唯一事实源：盲标包与影子评估据此复刻 prompt。

    这条断言守住的正是曾经出过的漂移——在别处重写拼装逻辑时漏掉换行。
    """
    llm = CapturingLLM()
    hard_filter = HardFilter(
        Layer3Settings(), llm, rule_engine=DisabledRuleEngine()
    )
    hard_filter.filter(sample_judge_context, sample_features)

    for index, prompt in enumerate(llm.prompts):
        evidence = HardFilter.build_evidence_prompt(
            sample_judge_context, [(index, sample_features[index])]
        )
        assert prompt.startswith(evidence)


def test_phase_continuity_changes_only_evaluation_context(
    sample_judge_context,
    sample_features,
) -> None:
    scene = ReplayedScene(
        frame_token="frame_001",
        scene_id="scene_001",
        scenario_type="red_light",
        context=sample_judge_context,
        features=sample_features,
        record={},
    )
    original_text = scene.context.scene_facts_text
    assert BASE_PHASE_SNAPSHOT_LINE in original_text

    changed = apply_phase_continuity_assumption(
        [scene],
        horizon_s=2.0,
    )[0]

    assert scene.context.scene_facts_text == original_text
    assert phase_continuity_statement(2.0) in changed.context.scene_facts_text
    assert changed.context.scene_facts_text.replace(
        phase_continuity_statement(2.0), BASE_PHASE_SNAPSHOT_LINE
    ) == original_text
