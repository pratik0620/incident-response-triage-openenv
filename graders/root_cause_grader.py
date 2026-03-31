"""
Root Cause Grader — used when the agent takes an 'identify_cause' action.

Scoring breakdown (max 1.0):
  +0.5   Correct service name present in answer
  +0.5   Cause-type keywords matched (proportional — partial credit allowed)
  -0.3   Per red-herring service the agent incorrectly blames (clamped to 0)

Design principles:
  - Fully deterministic: same inputs always produce same score.
  - No LLM calls: keyword/substring matching only.
  - Case-insensitive throughout.
  - Partial credit on keywords so agents that partially understand
    the cause still score better than agents that guess randomly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # GroundTruth is defined in models.py (same package root).
    # We import only for type hints so there is no circular dependency
    # when graders are imported before models in some test setups.
    from ..models import GroundTruth


def grade_root_cause(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Score whether the agent correctly identified the root cause.

    Args:
        answer:       The agent's free-text answer string.
        ground_truth: A GroundTruth instance attached to the current scenario.

    Returns:
        float in [0.0, 1.0] — higher is better.
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()
    score = 0.0

    # ── 0.5 pts: correct service named ───────────────────────────────────────
    if ground_truth.root_cause_service.lower() in answer_lower:
        score += 0.5

    # ── 0.5 pts: cause-type keywords (proportional partial credit) ────────────
    # Split on underscores and spaces so "memory_leak" → ["memory", "leak"]
    raw_cause = ground_truth.root_cause_type.lower().replace("_", " ")
    cause_keywords = [w for w in raw_cause.split() if w]

    if cause_keywords:
        matched = sum(1 for kw in cause_keywords if kw in answer_lower)
        score += 0.5 * (matched / len(cause_keywords))

    # ── penalty: red herring services blamed ─────────────────────────────────
    # Each red-herring service found in the answer deducts 0.3.
    # Score is clamped to 0 — can never go negative.
    for red_herring in ground_truth.red_herrings:
        if red_herring.lower() in answer_lower:
            score = max(0.0, score - 0.3)

    return round(min(score, 1.0), 4)