"""Layer 3 prompt 泄露扫描。"""

from __future__ import annotations

from src.layer3.exceptions import StructuredOutputValidationError


class LeakageGuard:
    """扫描完整 prompt 中不应暴露给 Layer 3 的内部字段。"""

    METADATA_TERMS: tuple[str, ...] = (
        "ground_truth",
        "ground truth",
        "expected_verdict",
        "variant_type",
        "benchmark_labels",
        "candidate_trajectories",
        "waypoints",
        '"x":',
        '"y":',
    )
    DYNAMIC_CONCLUSION_TERMS: tuple[str, ...] = (
        "traj_000_illegal",
        "illegal trajectory",
        "合成违规",
        "预设违规",
        "预设合规",
    )

    @classmethod
    def find_banned_terms(cls, text: str) -> list[str]:
        lowered = text.lower()
        hits: list[str] = []
        for term in cls.METADATA_TERMS + cls.DYNAMIC_CONCLUSION_TERMS:
            if term.lower() in lowered:
                hits.append(term)
        return hits

    @classmethod
    def assert_prompt_safe(cls, system_prompt: str, user_prompt: str) -> None:
        full_prompt = f"{system_prompt}\n{user_prompt}"
        hits = cls.find_banned_terms(full_prompt)
        if hits:
            joined = ", ".join(sorted(set(hits)))
            raise StructuredOutputValidationError(
                f"prompt contains Layer 3 leakage terms: {joined}"
            )

    @classmethod
    def assert_runtime_text_safe(cls, *fragments: str) -> None:
        """只扫描 Layer 1 运行时产出的叙述与特征文本。"""
        hits = cls.find_banned_terms("\n".join(fragments))
        if hits:
            joined = ", ".join(sorted(set(hits)))
            raise StructuredOutputValidationError(
                f"Layer 1 runtime text contains leakage terms: {joined}"
            )
