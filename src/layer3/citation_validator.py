"""运行期 citation validity 校验。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.layer3.exceptions import CitationValidityError


class CitationValidator:
    """只校验引用 rule_id 是否属于当前可引用集合。"""

    _SUMMARY_REF_RE = re.compile(r"\[(R-[A-Za-z]+-\d+)\]")

    def validate_validity(
        self,
        cited_ids: Iterable[str],
        allowed_ids: set[str],
        *,
        scene_id: str = "",
        stage: str = "",
    ) -> None:
        cited = set(cited_ids)
        unknown = cited - allowed_ids
        if unknown:
            raise CitationValidityError(
                "cited rule_ids must belong to the active rule set",
                scene_id=scene_id,
                stage=stage,
                cited_ids=cited,
                allowed_ids=allowed_ids,
            )

    def extract_summary_citations(self, text: str) -> list[str]:
        return self._SUMMARY_REF_RE.findall(text)
