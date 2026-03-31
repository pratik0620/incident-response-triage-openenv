"""
Categorization Grader — used exclusively for the HARD task difficulty.

In the hard task, the agent must not only find the root cause but also
classify what *kind* of failure it is:

    "real"         — a genuine, reproducible bug (e.g., OOM, crash loop)
    "flaky"        — an intermittent failure not caused by code changes
    "intermittent" — alias for "flaky" used by some scenarios
    "env_specific" — failure only in a specific environment (staging, prod, etc.)

Scoring breakdown (max 1.0):
  +0.6   Correct failure category named in the answer
  +0.2   Correct root-cause service named
  +0.2   Root-cause type keywords matched (proportional partial credit)

Design notes:
  - Category matching accepts both the underscore form ("env_specific") and
    the spaced form ("env specific") to be robust to agent formatting choices.
  - Root-cause sub-scores mirror root_cause_grader logic for consistency.
  - Fully deterministic — no LLM calls.
  - No red-herring penalty here; the category scoring already penalises
    agents that pick the wrong category (they get 0.0 on the 0.6 block).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GroundTruth

# The only valid failure categories.  Any value outside this set in
# ground_truth.failure_category is a data error, not a grader error.
VALID_CATEGORIES = {"real", "flaky", "intermittent", "env_specific"}


def grade_categorization(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Score the hard-task response: failure-type categorization + root cause.

    Args:
        answer:       The agent's free-text response.
        ground_truth: A GroundTruth instance for the current scenario.
                      Must have a non-empty `failure_category` field.

    Returns:
        float in [0.0, 1.0] — higher is better.
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()
    score = 0.0

    # ── 0.6 pts: correct failure category ────────────────────────────────────
    # We accept both "env_specific" (underscore) and "env specific" (spaced).
    category = ground_truth.failure_category.lower()           # e.g. "env_specific"
    category_spaced = category.replace("_", " ")               # e.g. "env specific"

    if category in {"flaky", "intermittent"}:
        if "flaky" in answer_lower or "intermittent" in answer_lower:
            score += 0.6
    elif category in answer_lower or category_spaced in answer_lower:
        score += 0.6

    # ── 0.2 pts: correct service named ───────────────────────────────────────
    if ground_truth.root_cause_service.lower() in answer_lower:
        score += 0.2

    # ── 0.2 pts: cause-type keywords (proportional partial credit) ────────────
    raw_cause = ground_truth.root_cause_type.lower().replace("_", " ")
    cause_keywords = [w for w in raw_cause.split() if w]

    if cause_keywords:
        matched = sum(1 for kw in cause_keywords if kw in answer_lower)
        score += 0.2 * (matched / len(cause_keywords))

    return round(min(score, 1.0), 4)