"""LeakageGuard tests for narrative, summaries, and full prompt fragments."""

from __future__ import annotations

import pytest

from src.layer1.exceptions import SceneValidationError
from src.layer1.leakage_guard import LeakageGuard


class TestLeakageGuard:
    def test_find_banned_terms_is_case_insensitive(self) -> None:
        hits = LeakageGuard.find_banned_terms("This ILLEGAL variant was vetoed.")
        assert hits == ["illegal", "vetoed", "variant"]

    def test_scrub_text_redacts_english_and_chinese_terms(self) -> None:
        scrubbed = LeakageGuard.scrub_text("illegal trajectory 违反 rules, ground truth")

        lowered = scrubbed.lower()
        assert "illegal" not in lowered
        assert "ground truth" not in lowered
        assert "违反" not in scrubbed
        assert scrubbed.count("[redacted]") == 3

    def test_assert_safe_prompt_fragment_accepts_safe_text(self) -> None:
        LeakageGuard.assert_safe_prompt_fragment("Trajectory speed rises from 2.0 to 4.0 m/s.")

    def test_assert_safe_prompt_fragment_scans_full_prompt_text(self) -> None:
        prompt_fragment = """
        Scene narrative: a pedestrian is near the crosswalk.
        Candidate: this trajectory is non-compliant and should stop.
        """
        with pytest.raises(SceneValidationError):
            LeakageGuard.assert_safe_prompt_fragment(prompt_fragment)
