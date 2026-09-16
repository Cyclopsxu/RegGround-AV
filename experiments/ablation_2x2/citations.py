"""Citation extraction and experiment-local enforce/observe validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.ablation_2x2.conditions import ValidationMode

_PROVISION_RE = re.compile(
    r"(?P<law>《?中华人民共和国道路交通安全法》?|《?道路交通安全法》?)"
    r"\s*第?\s*(?P<article>[0-9零〇一二三四五六七八九十百两]+)\s*条"
)
_CODE_RE = re.compile(r"RoadTrafficSafetyLaw-(?P<article>\d+)(?:-|$)")
_LAW_ALIASES = {
    "中华人民共和国道路交通安全法": "道路交通安全法",
    "道路交通安全法": "道路交通安全法",
}


@dataclass(frozen=True)
class ExtractedCitation:
    original_text: str
    law_name: str | None
    article_number: int | None

    @property
    def normalized_key(self) -> str | None:
        if self.law_name is None or self.article_number is None:
            return None
        return f"{self.law_name}:{self.article_number}"


@dataclass(frozen=True)
class CitationAssessment:
    original_text: str
    law_name: str | None
    article_number: int | None
    normalized_key: str | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleCatalog:
    """Audited rule-id and law/article catalog derived from the frozen graph YAML."""

    def __init__(
        self,
        rule_ids: set[str],
        provision_keys: set[str],
        rule_to_provision: dict[str, str],
    ) -> None:
        if not rule_ids or not provision_keys:
            raise ValueError("rule catalog must contain rule IDs and law/article keys")
        self.rule_ids = frozenset(rule_ids)
        self.provision_keys = frozenset(provision_keys)
        self.rule_to_provision = dict(rule_to_provision)

    @classmethod
    def from_yaml(cls, path: Path) -> RuleCatalog:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        rule_ids: set[str] = set()
        provision_keys: set[str] = set()
        rule_to_provision: dict[str, str] = {}
        for rule in rules:
            if not isinstance(rule, dict) or not rule.get("id"):
                continue
            rule_ids.add(str(rule["id"]))
            match = _CODE_RE.search(str(rule.get("code", "")))
            if match:
                key = f"道路交通安全法:{int(match.group('article'))}"
                provision_keys.add(key)
                rule_to_provision[str(rule["id"])] = key
        return cls(rule_ids, provision_keys, rule_to_provision)


def _chinese_integer(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if not text or any(char not in digits and char not in "十百" for char in text):
        return None
    total = 0
    current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char == "十":
            total += (current or 1) * 10
            current = 0
        else:
            total += (current or 1) * 100
            current = 0
    return total + current


def extract_provisions(text: str) -> list[ExtractedCitation]:
    """Extract law name + article; retain an unparsed item so hallucinations stay countable."""
    matches: list[ExtractedCitation] = []
    for match in _PROVISION_RE.finditer(text):
        law = match.group("law").strip("《》")
        matches.append(
            ExtractedCitation(
                original_text=match.group(0),
                law_name=_LAW_ALIASES.get(law, law),
                article_number=_chinese_integer(match.group("article")),
            )
        )
    if text.strip() and not matches:
        matches.append(ExtractedCitation(text.strip(), None, None))
    return matches


class CitationValidator:
    """The experiment-only validator; observe never raises or requests a retry."""

    def __init__(
        self,
        *,
        mode: ValidationMode,
        graph_rag: bool,
        allowed_rule_ids: set[str],
        catalog: RuleCatalog,
    ) -> None:
        self.mode = mode
        self.graph_rag = graph_rag
        self.allowed_rule_ids = frozenset(allowed_rule_ids)
        self.catalog = catalog

    def assess(
        self,
        *,
        cited_rule_ids: list[str],
        cited_provisions: list[str],
    ) -> list[CitationAssessment]:
        if self.graph_rag:
            return [
                CitationAssessment(rule_id, None, None, rule_id, rule_id in self.allowed_rule_ids)
                for rule_id in cited_rule_ids
            ]

        assessments: list[CitationAssessment] = []
        for value in cited_provisions:
            for citation in extract_provisions(value):
                key = citation.normalized_key
                assessments.append(
                    CitationAssessment(
                        citation.original_text,
                        citation.law_name,
                        citation.article_number,
                        key,
                        key in self.catalog.provision_keys if key is not None else False,
                    )
                )
        return assessments

    @staticmethod
    def passed(assessments: list[CitationAssessment]) -> bool:
        return all(item.valid for item in assessments)


def response_citations(
    response: dict[str, Any], validator: CitationValidator
) -> list[dict[str, Any]]:
    """Flatten verdict and preference citations while preserving their owner."""
    events: list[dict[str, Any]] = []
    owners: list[tuple[str, dict[str, Any]]] = []
    owners.extend(
        (f"verdict:{item.get('trajectory_id', '')}", item)
        for item in response.get("verdicts", [])
        if isinstance(item, dict)
    )
    owners.extend(
        (
            f"preference:{item.get('preferred_id', '')}>{item.get('dispreferred_id', '')}",
            item,
        )
        for item in response.get("preference_pairs", [])
        if isinstance(item, dict)
    )
    for owner, item in owners:
        assessed = validator.assess(
            cited_rule_ids=[str(value) for value in item.get("cited_rule_ids", [])],
            cited_provisions=[str(value) for value in item.get("cited_provisions", [])],
        )
        for assessment in assessed:
            events.append({"owner": owner, **assessment.to_dict()})
    return events


__all__ = [
    "CitationAssessment",
    "CitationValidator",
    "ExtractedCitation",
    "RuleCatalog",
    "extract_provisions",
    "response_citations",
]
