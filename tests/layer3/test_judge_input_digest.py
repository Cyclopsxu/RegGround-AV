"""JudgeInput digest 的隔离、渲染与 prompt 契约。"""

from src.layer1.interface_adapter import InterfaceAdapter
from src.layer3.context_builder import ContextBuilder
from src.layer3.leakage_guard import LeakageGuard


def test_digest_has_no_coordinates_or_benchmark_fields(sample_scene_context) -> None:
    judge_input = InterfaceAdapter().to_judge_input(sample_scene_context)
    dumped = judge_input.scene_facts_digest.model_dump()
    assert "conflict_zones" not in dumped
    assert "polygon_xy" not in dumped
    assert not {"variant_type", "expected_verdict", "benchmark_labels"}.intersection(dumped)


def test_digest_render_is_neutral_and_enters_its_own_prompt_section(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
) -> None:
    text = ContextBuilder.render_scene_facts_digest(sample_judge_input.scene_facts_digest)
    LeakageGuard.assert_prompt_safe("", text)
    assert "[场景要件]" in text
    assert "风险" not in text

    context = ContextBuilder(layer3_settings).build(
        sample_judge_input,
        sample_rule_subgraph,
    )
    assert context.scene_facts_text == text
