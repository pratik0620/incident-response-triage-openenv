"""
Dynamic Weight Engine and Conditional Adjustment Rules.

Weights are selected per difficulty level as specified in the design doc.
Adjustment rules are applied AFTER the weighted sum to catch scoring
patterns that would be unrealistic in real incident response.

All logic is deterministic and rule-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Weight Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalWeights:
    """
    Immutable weight configuration for one difficulty level.

    All weights must sum to 1.0. This is enforced at module load time.

    Signals:
        root_cause:  Signal A — correctness of diagnosis
        fix_quality: Signal B — quality/completeness of proposed fix
        reasoning:   Signal C — causal reasoning chain quality
        faithfulness:Signal D — grounding in provided logs
        noise:       Signal E — red-herring filtering ability
        efficiency:  Signal F — step budget efficiency
    """
    root_cause:   float
    fix_quality:  float
    reasoning:    float
    faithfulness: float
    noise:        float
    efficiency:   float

    def __post_init__(self) -> None:
        total = round(
            self.root_cause + self.fix_quality + self.reasoning
            + self.faithfulness + self.noise + self.efficiency,
            6,
        )
        assert abs(total - 1.0) < 1e-5, (
            f"SignalWeights must sum to 1.0, got {total}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty-Based Weight Tables
# ─────────────────────────────────────────────────────────────────────────────

# EASY: reward direct, correct diagnosis over explanation depth.
# Agent just needs to find the right answer efficiently.
WEIGHTS_EASY = SignalWeights(
    root_cause=0.35,
    fix_quality=0.30,
    reasoning=0.10,
    faithfulness=0.10,
    noise=0.10,
    efficiency=0.05,
)

# MEDIUM: agent must trace dependencies and demonstrate structured reasoning.
# Noise filtering starts to matter as scenarios have more red herrings.
WEIGHTS_MEDIUM = SignalWeights(
    root_cause=0.25,
    fix_quality=0.20,
    reasoning=0.25,
    faithfulness=0.15,
    noise=0.10,
    efficiency=0.05,
)

# HARD: correct reasoning and signal filtering dominate.
# Even a partially wrong root cause can score well if reasoning is sound.
# Speed matters least — thoroughness is the goal.
WEIGHTS_HARD = SignalWeights(
    root_cause=0.15,
    fix_quality=0.15,
    reasoning=0.30,
    faithfulness=0.20,
    noise=0.15,
    efficiency=0.05,
)

_WEIGHT_TABLE: Dict[str, SignalWeights] = {
    "easy":   WEIGHTS_EASY,
    "medium": WEIGHTS_MEDIUM,
    "hard":   WEIGHTS_HARD,
}


def _redistribute_fix_weight_for_identify(weights: SignalWeights) -> SignalWeights:
    """
    Move fix_quality weight into root_cause and reasoning for identify_cause.

    This keeps the total at 1.0 while avoiding a guaranteed 0.0 component
    when fix_quality is not evaluated.
    """
    if weights.fix_quality <= 0.0:
        return weights

    target_total = weights.root_cause + weights.reasoning
    if target_total <= 0.0:
        return weights

    shift = weights.fix_quality
    root_share = weights.root_cause / target_total
    reasoning_share = weights.reasoning / target_total

    return SignalWeights(
        root_cause=weights.root_cause + shift * root_share,
        fix_quality=0.0,
        reasoning=weights.reasoning + shift * reasoning_share,
        faithfulness=weights.faithfulness,
        noise=weights.noise,
        efficiency=weights.efficiency,
    )


def get_weights(difficulty: str) -> SignalWeights:
    """
    Return the SignalWeights for a given difficulty string.
    Unknown difficulty strings fall back to MEDIUM (safe default).
    """
    return _WEIGHT_TABLE.get(difficulty.lower(), WEIGHTS_MEDIUM)


def get_weights_for_action(difficulty: str, action_type: str) -> SignalWeights:
    """
    Return action-aware weights (e.g., identify_cause on easy difficulty).
    """
    base = get_weights(difficulty)
    if difficulty.lower() == "easy" and action_type == "identify_cause":
        return _redistribute_fix_weight_for_identify(base)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Sum
# ─────────────────────────────────────────────────────────────────────────────

def weighted_sum(
    root_cause: float,
    fix_quality: float,
    reasoning: float,
    faithfulness: float,
    noise: float,
    efficiency: float,
    weights: SignalWeights,
) -> float:
    """
    Compute the raw weighted sum of all six signals.
    Result is in [0.0, 1.0] before penalty adjustments.
    """
    return (
        weights.root_cause   * root_cause
        + weights.fix_quality  * fix_quality
        + weights.reasoning    * reasoning
        + weights.faithfulness * faithfulness
        + weights.noise        * noise
        + weights.efficiency   * efficiency
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Adjustment Rules
# ─────────────────────────────────────────────────────────────────────────────

# Rule 1 threshold: if root cause score is below this, apply diagnosis penalty.
_ROOT_CAUSE_THRESHOLD = 0.3

# Rule 1 penalty multiplier when root cause is very wrong.
_ROOT_CAUSE_PENALTY_FACTOR = 0.5

# Rule 2 gap threshold: fix_quality much higher than reasoning → guessing.
_FIX_REASONING_GAP = 0.5

# Rule 2 penalty when fix is high but reasoning is low.
_FIX_REASONING_PENALTY = 0.15

# Rule 4 threshold: many over-guess phrases → penalty.
# (Over-guessing is partially handled in noise_signal but we also
#  apply a macro penalty here if noise score is very low.)
_NOISE_LOW_THRESHOLD = 0.3
_NOISE_LOW_PENALTY = 0.10

# Hard bounds — platform requires strictly open interval (0, 1).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def apply_adjustment_rules(
    base_score: float,
    root_cause: float,
    fix_quality: float,
    reasoning: float,
    noise: float,
    difficulty: str,
) -> float:
    """
    Apply deterministic conditional adjustments to the weighted base score.

    Rule 1 — Root Cause Dependency:
        If root_cause < 0.3: multiply score by 0.5.
        Rationale: correct fix without correct diagnosis is luck, not skill.

    Rule 2 — Fix–Reasoning Consistency:
        If fix_quality - reasoning > 0.5: subtract 0.15.
        Rationale: high fix score + low reasoning score = copied answer.

    Rule 3 — Red Herring Penalty:
        Already fully applied inside noise_signal (Signal E).
        Here we apply an additional macro penalty if noise < 0.3.

    Rule 4 — Over-Guessing Penalty:
        Also partially in noise_signal. Additional -0.10 if noise < 0.3.

    Args:
        base_score:  Weighted sum before adjustments.
        root_cause:  Signal A score.
        fix_quality: Signal B score.
        reasoning:   Signal C score.
        noise:       Signal E score.
        difficulty:  "easy", "medium", or "hard".

    Returns:
        Adjusted score, clamped to (0.0, 1.0).
    """
    score = base_score

    # Rule 1: Root Cause Dependency
    if root_cause < _ROOT_CAUSE_THRESHOLD:
        score *= _ROOT_CAUSE_PENALTY_FACTOR

    # Rule 2: Fix–Reasoning Consistency
    # Only applied on medium/hard where reasoning matters.
    if difficulty in ("medium", "hard"):
        if (fix_quality - reasoning) > _FIX_REASONING_GAP:
            score -= _FIX_REASONING_PENALTY

    # Rules 3 & 4: Noise / over-guessing macro penalty
    if noise < _NOISE_LOW_THRESHOLD:
        score -= _NOISE_LOW_PENALTY

    if score <= _SCORE_MIN:
        return _SCORE_MIN
    if score >= _SCORE_MAX:
        return _SCORE_MAX
    return round(score, 4)