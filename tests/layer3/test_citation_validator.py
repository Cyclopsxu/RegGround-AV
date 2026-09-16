"""CitationValidator 测试。"""

import pytest

from src.layer3.citation_validator import CitationValidator
from src.layer3.exceptions import CitationValidityError


def test_validate_validity_accepts_allowed_ids():
    CitationValidator().validate_validity(
        ["R-SIG-01"],
        {"R-SIG-01", "R-SIG-03"},
    )


def test_validate_validity_rejects_unknown_ids():
    with pytest.raises(CitationValidityError) as exc_info:
        CitationValidator().validate_validity(["R-FAKE-01"], {"R-SIG-01"})
    assert "R-FAKE-01" in exc_info.value.cited_ids


def test_extract_summary_citations():
    refs = CitationValidator().extract_summary_citations(
        "轨迹被否决 [R-SIG-01]，偏好参考 [R-YLD-05]。"
    )
    assert refs == ["R-SIG-01", "R-YLD-05"]
