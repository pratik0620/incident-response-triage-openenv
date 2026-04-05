"""
Composite Grader — the single entry point called by environment.py.

Upgrade from v1 (keyword-only) to v2 (multi-signal evaluation):

v1 computed:
    final = base_from_one_grader + efficiency_bonus

v2 computes a weighted combination of six independent signals:
    A. Root Cause Score      — correct service + failure type
    B. Fix Quality Score     — tiered correctness + prevention
    C. Reasoning Score       — causal chain explanation quality
    D. Faithfulness Score    — grounding in actual log content
    E. Noise Handling Score  — red-herring filtering
    F. Efficiency Score      — step budget usage

Weights are selected dynamically per difficulty level (easy/medium/hard).
Conditional adjustment rules are applied after the weighted sum.

API CONTRACT (unchanged from v1):
    compute_final_score(action_type, answer, ground_truth,
                        steps_used, max_steps, difficulty,
                        log_content="") -> float

The log_content parameter is NEW and optional. When provided, Signal D
(faithfulness) scores how well the agent grounded its answer in logs.
When omitted, faithfulness returns a neutral 0.5.

Action routing:
    "identify_cause"  → Signals A, C, D, E, F
                        (fix quality = 0 — agent didn't propose a fix)
    "propose_fix"     → All six signals
    "escalate"        → Flat 0.2 base (no signal computation)
    anything else     → 0.0

Determinism guarantee:
    All scoring is rule-based keyword/pattern matching.
    No randomness. No external API calls. No ML inference.
    Same inputs always produce the same output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .categorization_grader import grade_categorization
from .signals import (
    efficiency_signal,
    faithfulness_signal,
    fix_quality_signal,
    noise_signal,
    reasoning_signal,
    root_cause_signal,
)
from .weights import apply_adjustment_rules, get_weights_for_action, weighted_sum

if TYPE_CHECKING:
    from ..models import GroundTruth

# Flat score for "escalate" — always a fallback, never a winning strategy.
_ESCALATE_BASE = 0.2

# Hard difficulty label
_HARD = "hard"


def compute_final_score(
    action_type: str,
    answer: str,
    ground_truth: "GroundTruth",
    steps_used: int,
    max_steps: int,
    difficulty: str,
    log_content: str = "",
) -> float:
    """
    Master grader — called by environment.py at the end of each episode.

    Args:
        action_type:  Terminal action the agent chose.
                      One of: "identify_cause", "propose_fix", "escalate".
        answer:       Agent's free-text response string.
        ground_truth: GroundTruth object for this scenario.
        steps_used:   Steps taken before terminating.
        max_steps:    Step budget for this difficulty level.
        difficulty:   "easy", "medium", or "hard".
        log_content:  (Optional) All log/alert text the agent had access to.
                      Used for Signal D (faithfulness). Pass "" to skip.

    Returns:
        float in [0.0, 1.0] — the episode's final score.
    """

    # ── Handle escalate and invalid actions immediately ───────────────────────
    if action_type == "escalate":
        return round(_ESCALATE_BASE, 4)

    if action_type not in ("identify_cause", "propose_fix"):
        return 0.0

    # ── Compute all six signals ───────────────────────────────────────────────

    # Signal A: Root Cause
    # Hard task uses categorization grader which adds category scoring on top.
    if difficulty == _HARD and action_type == "identify_cause":
        sig_a = grade_categorization(answer, ground_truth)
    else:
        sig_a = root_cause_signal(answer, ground_truth)

    # Signal B: Fix Quality
    # For "identify_cause", fix quality is 0 — agent wasn't asked to fix.
    # For "propose_fix", we score the fix quality.
    if action_type == "propose_fix":
        sig_b = fix_quality_signal(answer, ground_truth)
    else:
        sig_b = 0.0

    # Signal C: Reasoning
    sig_c = reasoning_signal(answer, ground_truth)

    # Signal D: Faithfulness
    sig_d = faithfulness_signal(answer, ground_truth, log_content)

    # Signal E: Noise Handling
    sig_e = noise_signal(answer, ground_truth)

    # Signal F: Efficiency
    sig_f = efficiency_signal(steps_used, max_steps)

    # ── Get difficulty-appropriate weights ────────────────────────────────────
    weights = get_weights_for_action(difficulty, action_type)

    # ── Weighted sum ──────────────────────────────────────────────────────────
    base = weighted_sum(
        root_cause=sig_a,
        fix_quality=sig_b,
        reasoning=sig_c,
        faithfulness=sig_d,
        noise=sig_e,
        efficiency=sig_f,
        weights=weights,
    )

    # ── Conditional adjustment rules ──────────────────────────────────────────
    final = apply_adjustment_rules(
        base_score=base,
        root_cause=sig_a,
        fix_quality=sig_b,
        reasoning=sig_c,
        noise=sig_e,
        difficulty=difficulty,
    )

    return round(max(0.0, min(final, 1.0)), 4)


def compute_score_breakdown(
    action_type: str,
    answer: str,
    ground_truth: "GroundTruth",
    steps_used: int,
    max_steps: int,
    difficulty: str,
    log_content: str = "",
) -> dict:
    """
    Same as compute_final_score but returns the full signal breakdown.

    Useful for debugging, logging, and the README baseline script.

    Returns:
        dict with keys:
            final_score, root_cause, fix_quality, reasoning,
            faithfulness, noise, efficiency, weights_used, difficulty
    """
    if action_type == "escalate":
        return {
            "final_score": _ESCALATE_BASE,
            "root_cause": 0.0,
            "fix_quality": 0.0,
            "reasoning": 0.0,
            "faithfulness": 0.0,
            "noise": 0.0,
            "efficiency": 0.0,
            "weights_used": "escalate (flat)",
            "difficulty": difficulty,
            "action_type": action_type,
        }

    if action_type not in ("identify_cause", "propose_fix"):
        return {
            "final_score": 0.0,
            "root_cause": 0.0,
            "fix_quality": 0.0,
            "reasoning": 0.0,
            "faithfulness": 0.0,
            "noise": 0.0,
            "efficiency": 0.0,
            "weights_used": "invalid action",
            "difficulty": difficulty,
            "action_type": action_type,
        }

    # Compute signals
    if difficulty == _HARD and action_type == "identify_cause":
        sig_a = grade_categorization(answer, ground_truth)
    else:
        sig_a = root_cause_signal(answer, ground_truth)

    sig_b = fix_quality_signal(answer, ground_truth) if action_type == "propose_fix" else 0.0
    sig_c = reasoning_signal(answer, ground_truth)
    sig_d = faithfulness_signal(answer, ground_truth, log_content)
    sig_e = noise_signal(answer, ground_truth)
    sig_f = efficiency_signal(steps_used, max_steps)

    weights = get_weights_for_action(difficulty, action_type)
    base = weighted_sum(sig_a, sig_b, sig_c, sig_d, sig_e, sig_f, weights)
    final = apply_adjustment_rules(base, sig_a, sig_b, sig_c, sig_e, difficulty)
    final = round(max(0.0, min(final, 1.0)), 4)

    return {
        "final_score":   final,
        "root_cause":    round(sig_a, 4),
        "fix_quality":   round(sig_b, 4),
        "reasoning":     round(sig_c, 4),
        "faithfulness":  round(sig_d, 4),
        "noise":         round(sig_e, 4),
        "efficiency":    round(sig_f, 4),
        "base_before_rules": round(base, 4),
        "weights_used":  difficulty,
        "difficulty":    difficulty,
        "action_type":   action_type,
    }