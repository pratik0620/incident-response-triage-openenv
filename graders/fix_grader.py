"""
Fix Grader — used when the agent takes a 'propose_fix' action.

Scoring breakdown (max 1.0):
  score = (number of fix-keywords found in answer) / (total fix-keywords)

Key design decisions:
  - Only words longer than 4 characters are used as keywords.
    Short words ("the", "and", "with", "set") are filler that appear
    in almost any answer and would inflate scores artificially.
  - Matching is case-insensitive substring-based.
  - Fully deterministic — no LLM calls.
  - Partial credit: finding 3 out of 5 fix concepts scores 0.6.

Note: In composite_grader, this score is weighted 0.6 and combined with
      0.4 × root_cause_score when action_type == 'propose_fix', because
      a fix that correctly names the cause AND the remedy is worth more
      than one that only names the remedy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GroundTruth

# Minimum character length for a word to count as a meaningful keyword.
# Words <= this length are considered filler and are skipped.
_MIN_KEYWORD_LENGTH = 4


def grade_fix(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Score whether the proposed fix is correct.

    Args:
        answer:       The agent's free-text fix proposal.
        ground_truth: A GroundTruth instance for the current scenario.

    Returns:
        float in [0.0, 1.0] — higher is better.
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()

    # Extract meaningful keywords from the expected fix string.
    # We split on whitespace and filter out short/filler words.
    fix_keywords = [
        word
        for word in ground_truth.correct_fix.lower().split()
        if len(word) > _MIN_KEYWORD_LENGTH
    ]

    # Avoid giving fix credit for words that only describe the root cause.
    raw_cause = ground_truth.root_cause_type.lower().replace("_", " ")
    cause_keywords = {w for w in raw_cause.split() if w}
    fix_keywords = [kw for kw in fix_keywords if kw not in cause_keywords]

    # Guard: if the correct_fix has no meaningful keywords (very short phrase),
    # we cannot evaluate — return 0.0 to avoid dividing by zero.
    if not fix_keywords:
        return 0.0

    matched = sum(1 for kw in fix_keywords if kw in answer_lower)
    return round(matched / len(fix_keywords), 4)