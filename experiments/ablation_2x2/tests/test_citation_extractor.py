from __future__ import annotations

from experiments.ablation_2x2.citations import RuleCatalog, extract_provisions
from experiments.ablation_2x2.run_ablation import RULE_GRAPH


def test_extracts_arabic_and_chinese_article_numbers_with_aliases() -> None:
    extracted = extract_provisions(
        "依据《中华人民共和国道路交通安全法》第三十八条及道路交通安全法第47条。"
    )
    assert [item.normalized_key for item in extracted] == [
        "道路交通安全法:38",
        "道路交通安全法:47",
    ]


def test_catalog_matches_existing_articles_and_rejects_unknown_text() -> None:
    catalog = RuleCatalog.from_yaml(RULE_GRAPH)
    assert "道路交通安全法:38" in catalog.provision_keys
    assert "道路交通安全法:999" not in catalog.provision_keys
    unknown = extract_provisions("据有关规定处理")
    assert len(unknown) == 1 and unknown[0].normalized_key is None
    assert extract_provisions("") == []
