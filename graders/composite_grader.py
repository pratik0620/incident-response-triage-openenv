"""
Composite Grader — the single entry point called by environment.py.

This module routes an agent's terminal action to the appropriate sub-grader,
then applies a step-efficiency bonus to reward faster diagnosis.

Action routing:
  "identify_cause"  → root_cause_grader only
  "propose_fix"     → 0.4 × root_cause_score + 0.6 × fix_score
                      (agent must name cause AND remedy to score full marks)
  "escalate"        → flat 0.2 (valid fallback, but never rewarded fully)
  anything else     → 0.0 (invalid action)

Hard task:
  When difficulty == "hard" and action_type == "identify_cause",
  the categorization grader is used instead of root_cause_grader alone,
  because the hard task requires failure-type classification too.

Step-efficiency bonus (up to +0.15):
  The bonus rewards agents that diagnose faster than the step budget allows.
  Formula:
      efficiency_ratio = 1.0 - (steps_used / max_steps)
      bonus            = 0.15 × max(0.0, efficiency_ratio)

  Examples:
    steps_used=5,  max_steps=20  → ratio=0.75 → bonus=0.1125
    steps_used=10, max_steps=20  → ratio=0.50 → bonus=0.075
    steps_used=20, max_steps=20  → ratio=0.0  → bonus=0.0

    This decays linearly from full bonus (instant solve) to zero bonus
    (used every step). An agent cannot earn the bonus by escalating — the
    flat 0.2 base for "escalate" is applied before adding the bonus, so
    the best possible escalate score is 0.2 + 0.15 = 0.35. Invalid actions
    do not receive any efficiency bonus.

Final score = min(base + efficiency_bonus, 1.0), rounded to 4 decimal places.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .categorization_grader import grade_categorization
from .fix_grader import grade_fix
from .root_cause_grader import grade_root_cause

if TYPE_CHECKING:
    from ..models import GroundTruth

# Weight of fix_grader score vs root_cause_grader score for "propose_fix".
_FIX_WEIGHT = 0.6
_CAUSE_WEIGHT = 0.4

# Fixed score for the "escalate" fallback action.
_ESCALATE_BASE = 0.2

# Maximum efficiency bonus achievable (added on top of base score).
_MAX_EFFICIENCY_BONUS = 0.15

# Difficulty level string for the hard task.
_HARD = "hard"

ActionType = Literal["identify_cause", "propose_fix", "escalate"]


def compute_final_score(
    action_type: str,
    answer: str,
    ground_truth: "GroundTruth",
    steps_used: int,
    max_steps: int,
    difficulty: str,
) -> float:
    """
    Master grader — called by environment.py at the end of each episode.

    Args:
        action_type:  The terminal action the agent chose.
                      One of: "identify_cause", "propose_fix", "escalate".
        answer:       The agent's free-text response string.
        ground_truth: The GroundTruth object for this scenario.
        steps_used:   How many steps the agent took before terminating.
        max_steps:    The step budget for this difficulty level.
        difficulty:   "easy", "medium", or "hard".

    Returns:
        float in [0.0, 1.0] — the episode's final score.
    """
    # ── 1. Base score from the appropriate sub-grader ─────────────────────────
    invalid_action = False

    if action_type == "identify_cause":
        if difficulty == _HARD:
            # Hard task: agent must also classify the failure category.
            base = grade_categorization(answer, ground_truth)
        else:
            # Easy / medium: root cause identification only.
            base = grade_root_cause(answer, ground_truth)

    elif action_type == "propose_fix":
        # Propose-fix requires naming the cause AND the remedy.
        # Both sub-graders are always run regardless of difficulty.
        cause_score = grade_root_cause(answer, ground_truth)
        if ground_truth.root_cause_service.lower() not in answer.lower():
            cause_score = 0.0
        fix_score = grade_fix(answer, ground_truth)
        base = _CAUSE_WEIGHT * cause_score + _FIX_WEIGHT * fix_score

    elif action_type == "escalate":
        # Escalating is a valid but weak fallback — agents that cannot
        # diagnose can still choose this rather than guessing wildly.
        base = _ESCALATE_BASE

    else:
        # Unknown / invalid action type — no score.
        base = 0.0
        invalid_action = True

    # ── 2. Step-efficiency bonus ──────────────────────────────────────────────
    # Protect against division by zero (max_steps should always be > 0).
    if max_steps > 0:
        efficiency_ratio = max(0.0, 1.0 - (steps_used / max_steps))
    else:
        efficiency_ratio = 0.0

    efficiency_bonus = _MAX_EFFICIENCY_BONUS * efficiency_ratio
    if invalid_action:
        efficiency_bonus = 0.0

    # ── 3. Combine and clamp ──────────────────────────────────────────────────
    final = min(base + efficiency_bonus, 1.0)
    return round(final, 4)