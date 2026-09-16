"""Layer 1 标签与合规结论词泄露扫描。"""

from __future__ import annotations

import re

from src.layer1.exceptions import SceneValidationError


class LeakageGuard:
    """扫描 narrative、feature summary 与 prompt fragment 中的泄露词。"""

    BANNED_TERMS: tuple[str, ...] = (
        "illegal",
        "ground truth",
        "vetoed",
        "cleared",
        "variant",
        "violate",
        "violation",
        "compliant",
        "non-compliant",
        "the action is",
        "keep going",
        "should",
        "should stop",
        "must yield",
        "违规",
        "违反",
        "合规",
        "非法",
        "应该",
        "必须",
    )

    _REPLACEMENT = "[redacted]"

    @classmethod
    def find_banned_terms(cls, text: str) -> list[str]:
        """返回文本中命中的泄露词，大小写不敏感。"""
        lowered = text.lower()
        hits: list[str] = []
        for term in cls.BANNED_TERMS:
            needle = term.lower()
            if needle in lowered:
                hits.append(term)
        return hits

    @classmethod
    def scrub_text(cls, text: str) -> str:
        """移除 narrative 中的标签词和合规结论词。"""
        scrubbed = text
        for term in cls.BANNED_TERMS:
            scrubbed = re.sub(
                re.escape(term),
                cls._REPLACEMENT,
                scrubbed,
                flags=re.IGNORECASE,
            )
        return scrubbed

    @classmethod
    def assert_safe_prompt_fragment(cls, text: str) -> None:
        """若 prompt 片段含泄露词则抛出 SceneValidationError。"""
        hits = cls.find_banned_terms(text)
        if hits:
            joined = ", ".join(hits)
            raise SceneValidationError(f"prompt fragment contains banned terms: {joined}")
