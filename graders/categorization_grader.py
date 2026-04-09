"""
Categorization Grader — used exclusively for HARD task difficulty.

In the hard task, the agent must classify the failure type AND identify
the root cause. This grader combines both requirements.

Failure categories:
    "real"         — genuine reproducible bug (OOM, crash loop, etc.)
    "flaky"        — intermittent failure, not caused by code changes
    "intermittent" — alias for "flaky"
    "env_specific" — failure only in a specific environment

Scoring breakdown (max 1.0):
    +0.6   Correct failure category
    +0.2   Correct root-cause service (via Signal A partial)
    +0.2   Root-cause type keywords matched (via synonym expansion)

Upgrade from v1:
    - Category matching still accepts underscore and spaced forms.
    - "flaky"/"intermittent" are treated as aliases.
    - Root-cause matching now uses synonym expansion (same as root_cause_grader).
    - Scores strictly in (0.0, 1.0), fully deterministic.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .synonyms import get_cause_synonyms

if TYPE_CHECKING:
    from ..models import GroundTruth

VALID_CATEGORIES = {"real", "flaky", "intermittent", "env_specific"}

# Hard bounds — platform requires strictly open interval (0, 1).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(score: float) -> float:
    """Enforce strictly open (0, 1) on categorization score."""
    val = float(score)
    if val <= _SCORE_MIN:
        return _SCORE_MIN
    if val >= _SCORE_MAX:
        return _SCORE_MAX
    return round(val, 4)


def grade_categorization(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Score the hard-task response: failure-type categorization + root cause.

    Args:
        answer:       The agent's free-text response.
        ground_truth: A GroundTruth instance. Must have failure_category set.

    Returns:
        float strictly in (0.0, 1.0) — higher is better.
    """
    if not answer or not answer.strip():
        return _clamp(0.0)

    answer_lower = answer.lower()
    score = 0.0

    # ── 0.6 pts: correct failure category ────────────────────────────────────
    category = ground_truth.failure_category.lower()
    category_spaced = category.replace("_", " ")

    if category in {"flaky", "intermittent"}:
        # Both terms are valid answers for this category
        if "flaky" in answer_lower or "intermittent" in answer_lower:
            score += 0.6
    elif category in answer_lower or category_spaced in answer_lower:
        score += 0.6

    # ── 0.2 pts: correct service named ───────────────────────────────────────
    service = ground_truth.root_cause_service.lower()
    service_parts = [
        p for p in service.replace("-", " ").replace("_", " ").split()
        if len(p) > 3
    ]
    if service in answer_lower or any(p in answer_lower for p in service_parts):
        score += 0.2

    # ── 0.2 pts: cause-type keywords (proportional, with synonym expansion) ───
    all_cause_terms = get_cause_synonyms(ground_truth.root_cause_type)
    phrase_hits = sum(1 for t in all_cause_terms if " " in t and t in answer_lower)
    word_hits   = sum(1 for t in all_cause_terms if " " not in t and t in answer_lower)

    if phrase_hits >= 1:
        score += 0.2
    elif word_hits >= 2:
        score += 0.15
    elif word_hits == 1:
        score += 0.08

    return _clamp(score)