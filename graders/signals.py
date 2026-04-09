"""
Multi-Signal Grader — six independent scoring signals (A through F).

Every function in this file:
  - Takes only strings and GroundTruth fields as input
  - Is 100% deterministic (no randomness, no external calls)
  - Returns a float strictly in (0.0, 1.0) — never exactly 0.0 or 1.0
  - Has clear, auditable scoring logic

Signal overview:
  A. root_cause_signal   — did agent identify service + failure type?
  B. fix_quality_signal  — how good is the proposed fix? (tiered scoring)
  C. reasoning_signal    — does agent show causal chain logic?
  D. faithfulness_signal — is reasoning grounded in the provided logs?
  E. noise_signal        — did agent correctly ignore red herrings?
  F. efficiency_signal   — did agent solve within step budget efficiently?
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .synonyms import (
    LOG_SIGNAL_KEYWORDS,
    OVER_GUESS_PHRASES,
    REASONING_CHAIN_PATTERNS,
    SUPERFICIAL_FIX_PHRASES,
    get_cause_synonyms,
    get_fix_tier_score,
)

if TYPE_CHECKING:
    from ..models import GroundTruth

# Hard bounds — platform requires strictly open interval (0, 1).
# Every signal function must pass its return value through _clamp().
_SIG_MIN = 0.01
_SIG_MAX = 0.99


def _clamp(score: float) -> float:
    """Enforce strictly open (0, 1) on any signal score."""
    val = float(score)
    if val <= _SIG_MIN:
        return _SIG_MIN
    if val >= _SIG_MAX:
        return _SIG_MAX
    return round(val, 4)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL A — Root Cause Score
# ─────────────────────────────────────────────────────────────────────────────

def root_cause_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent identified the correct service and failure type.

    Uses synonym expansion so "heap overflow" matches "memory_leak".
    """
    if not answer or not answer.strip():
        return _clamp(0.0)

    answer_lower = answer.lower()
    score = 0.0

    service = ground_truth.root_cause_service.lower()
    service_parts = [p for p in service.replace("-", " ").replace("_", " ").split() if len(p) > 3]
    service_hit = service in answer_lower or any(p in answer_lower for p in service_parts)
    if service_hit:
        score += 0.5

    all_cause_terms = get_cause_synonyms(ground_truth.root_cause_type)
    phrase_hits = sum(1 for term in all_cause_terms if " " in term and term in answer_lower)
    word_hits   = sum(1 for term in all_cause_terms if " " not in term and term in answer_lower)

    if phrase_hits >= 1:
        score += 0.5
    elif word_hits >= 2:
        score += 0.4
    elif word_hits == 1:
        score += 0.25

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL B — Fix Quality Score
# ─────────────────────────────────────────────────────────────────────────────

def fix_quality_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate fix completeness using tiered keyword matching.

    Tiers (defined per cause type in synonyms.py):
        tier_1: superficial fix (restart, redeploy)      -> 0.3
        tier_2: correct functional fix                   -> 0.7
        tier_3: correct + preventive fix                 -> 1.0 (clamped to 0.99)
    """
    if not answer or not answer.strip():
        return _clamp(0.0)

    answer_lower = answer.lower()

    tier_score = get_fix_tier_score(
        answer_lower,
        ground_truth.root_cause_type,
        ground_truth.correct_fix,
    )

    if tier_score == 0.3:
        service = ground_truth.root_cause_service.lower()
        if service not in answer_lower:
            return _clamp(0.1)

    return _clamp(tier_score)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL C — Reasoning Score
# ─────────────────────────────────────────────────────────────────────────────

def reasoning_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent provides a logical causal reasoning chain.
    """
    if not answer or not answer.strip():
        return _clamp(0.0)

    answer_lower = answer.lower()

    chain_hits = sum(1 for p in REASONING_CHAIN_PATTERNS if p in answer_lower)

    has_causal_connector = any(c in answer_lower for c in [
        "because", "caused by", "triggered by", "due to",
        "leading to", "resulting in", "->", "->",
    ])
    has_evidence_phrase = any(e in answer_lower for e in [
        "the logs show", "log indicates", "based on the logs",
        "from the logs", "error rate", "metric shows", "alert shows",
        "spike", "the alert",
    ])
    has_conclusion_phrase = any(c in answer_lower for c in [
        "root cause is", "root cause:", "therefore", "this means",
        "diagnosed as", "conclusion",
    ])

    score = 0.0
    score += min(chain_hits * 0.08, 0.4)

    if has_causal_connector:
        score += 0.2
    if has_evidence_phrase:
        score += 0.25
    if has_conclusion_phrase:
        score += 0.15

    word_count = len(answer.split())
    if word_count < 15:
        score *= 0.4

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL D — Faithfulness Score
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_signal(
    answer: str,
    ground_truth: "GroundTruth",
    log_content: str,
) -> float:
    """
    Evaluate whether agent reasoning is grounded in actual log content.
    """
    if not answer or not answer.strip():
        return _clamp(0.0)

    if not log_content or not log_content.strip():
        return _clamp(0.5)

    answer_lower = answer.lower()
    log_lower    = log_content.lower()

    total_hits   = 0
    grounded     = 0
    hallucinated = 0

    for kw in LOG_SIGNAL_KEYWORDS:
        if kw in answer_lower:
            total_hits += 1
            if kw in log_lower:
                grounded += 1
            else:
                hallucinated += 1

    if total_hits == 0:
        return _clamp(0.4)

    grounded_ratio = grounded / total_hits
    score = 0.7 * grounded_ratio

    cause_synonyms  = get_cause_synonyms(ground_truth.root_cause_type)
    cause_in_logs   = any(term in log_lower   for term in cause_synonyms)
    cause_in_answer = any(term in answer_lower for term in cause_synonyms)
    if cause_in_logs and cause_in_answer:
        score += 0.3

    hallucination_penalty = min(hallucinated * 0.15, 0.4)
    score = max(0.0, score - hallucination_penalty)

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL E — Noise Handling Score
# ─────────────────────────────────────────────────────────────────────────────

def noise_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent correctly ignored red-herring services/signals.

    NOTE: score starts at 0.98 (not 1.0) to ensure it never reaches exactly
    1.0 even when the agent perfectly ignores all red herrings.
    """
    if not answer or not answer.strip():
        return _clamp(0.5)

    answer_lower = answer.lower()
    # Start below 1.0 so perfect performance returns 0.98 (safely clamped to 0.99)
    score = 0.98

    red_herrings = ground_truth.red_herrings or []

    for rh in red_herrings:
        rh_lower = rh.lower()
        if rh_lower in answer_lower:
            ruled_out = any(
                phrase in answer_lower
                for phrase in (
                    f"{rh_lower} is not",
                    f"not {rh_lower}",
                    f"ruled out {rh_lower}",
                    f"{rh_lower} does not",
                )
            )
            if ruled_out:
                score += 0.01
            else:
                score -= 0.3

    over_guess_count = sum(1 for p in OVER_GUESS_PHRASES if p in answer_lower)
    if over_guess_count >= 3:
        score -= 0.3
    elif over_guess_count >= 1:
        score -= 0.1

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL F — Efficiency Score
# ─────────────────────────────────────────────────────────────────────────────

def efficiency_signal(steps_used: int, max_steps: int) -> float:
    """
    Evaluate how efficiently the agent reached its conclusion.

    Scoring curve (non-linear — rewards decisiveness):
        <= 25% of budget used  -> 0.99
        <= 50% of budget used  -> 0.80
        <= 75% of budget used  -> 0.60
        <= 90% of budget used  -> 0.40
        100% of budget used    -> 0.20
        > budget (capped)      -> 0.01

    NOTE: 1.0 and 0.0 are never returned — all values pass through _clamp().
    """
    if max_steps <= 0:
        return _clamp(0.5)

    ratio = steps_used / max_steps

    if ratio <= 0.25:
        raw = 0.99
    elif ratio <= 0.50:
        raw = 0.80
    elif ratio <= 0.75:
        raw = 0.60
    elif ratio <= 0.90:
        raw = 0.40
    elif ratio <= 1.0:
        raw = 0.20
    else:
        raw = 0.01

    return _clamp(raw)