# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-Signal Grader — six independent scoring signals (A through F).

Every function in this file:
  - Takes only strings and GroundTruth fields as input
  - Is 100% deterministic (no randomness, no external calls)
  - Returns a float in [0.0, 1.0]
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


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL A — Root Cause Score
# ─────────────────────────────────────────────────────────────────────────────

def root_cause_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent identified the correct service and failure type.

    Scoring:
        Exact match (service + type fully identified)         → 1.0
        Service correct + partial cause keywords             → 0.5–0.9
        Only cause type correct, service wrong               → 0.3–0.5
        Completely wrong                                     → 0.0

    Uses synonym expansion so "heap overflow" matches "memory_leak".
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()
    score = 0.0

    # ── Service match (0.5 pts) ───────────────────────────────────────────────
    service = ground_truth.root_cause_service.lower()
    # Accept partial service name match (e.g. "payment" matches "payment-service")
    service_parts = [p for p in service.replace("-", " ").replace("_", " ").split() if len(p) > 3]
    service_hit = service in answer_lower or any(p in answer_lower for p in service_parts)
    if service_hit:
        score += 0.5

    # ── Cause type match (0.5 pts, proportional via synonym expansion) ────────
    all_cause_terms = get_cause_synonyms(ground_truth.root_cause_type)

    # Multi-word phrases score more than single words
    phrase_hits = sum(1 for term in all_cause_terms if " " in term and term in answer_lower)
    word_hits   = sum(1 for term in all_cause_terms if " " not in term and term in answer_lower)

    if phrase_hits >= 1:
        # At least one full phrase matched
        score += 0.5
    elif word_hits >= 2:
        # Multiple individual keywords matched
        score += 0.4
    elif word_hits == 1:
        # One keyword matched — partial credit
        score += 0.25

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL B — Fix Quality Score
# ─────────────────────────────────────────────────────────────────────────────

def fix_quality_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate fix completeness using tiered keyword matching.

    Tiers (defined per cause type in synonyms.py):
        tier_1: superficial fix (restart, redeploy)      → 0.3
        tier_2: correct functional fix                   → 0.7
        tier_3: correct + preventive fix                 → 1.0

    Penalty: if agent only gives a superficial fix AND root cause
             is not identified, score is capped at 0.2.
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()

    tier_score = get_fix_tier_score(
        answer_lower,
        ground_truth.root_cause_type,
        ground_truth.correct_fix,
    )

    # Penalty: superficial fix with no cause understanding
    if tier_score == 0.3:
        # Check if agent at least named the service
        service = ground_truth.root_cause_service.lower()
        if service not in answer_lower:
            return 0.1  # Pure restart guess with no diagnosis

    return round(tier_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL C — Reasoning Score
# ─────────────────────────────────────────────────────────────────────────────

def reasoning_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent provides a logical causal reasoning chain.

    Scoring:
        Strong causal chain with evidence phrases     → 0.8–1.0
        Some reasoning vocabulary, weak chain        → 0.4–0.7
        No reasoning structure                       → 0.0–0.3

    Pattern detection is purely string-based (no NLP library).
    """
    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()

    # Count reasoning chain markers
    chain_hits = sum(1 for p in REASONING_CHAIN_PATTERNS if p in answer_lower)

    # Check for explicit causal structure: "X → Y" or "X caused Y" or "X because Y"
    has_causal_connector = any(c in answer_lower for c in [
        "because", "caused by", "triggered by", "due to",
        "leading to", "resulting in", "→", "->",
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

    # Score based on combinations
    score = 0.0
    score += min(chain_hits * 0.08, 0.4)   # up to 0.4 from raw pattern hits

    if has_causal_connector:
        score += 0.2
    if has_evidence_phrase:
        score += 0.25
    if has_conclusion_phrase:
        score += 0.15

    # Length heuristic: very short answers can't have good reasoning
    word_count = len(answer.split())
    if word_count < 15:
        score *= 0.4  # heavy penalty for one-liners

    return round(min(score, 1.0), 4)


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

    High score: agent uses signals that ARE present in logs.
    Penalty:    agent introduces causes NOT present in logs (hallucination).

    Args:
        answer:       Agent's free-text response.
        ground_truth: Scenario ground truth.
        log_content:  All log/alert text the agent had access to (joined string).
                      If empty, this signal returns 0.5 (neutral — can't evaluate).

    Scoring:
        Grounded signals present in both logs and answer   → +score
        Agent claims cause not in logs                     → -penalty
    """
    if not answer or not answer.strip():
        return 0.0

    # If no logs provided to compare against, return neutral
    if not log_content or not log_content.strip():
        return 0.5

    answer_lower   = answer.lower()
    log_lower      = log_content.lower()

    score      = 0.0
    total_hits = 0
    grounded   = 0
    hallucinated = 0

    for kw in LOG_SIGNAL_KEYWORDS:
        if kw in answer_lower:
            total_hits += 1
            if kw in log_lower:
                grounded += 1
            else:
                hallucinated += 1

    if total_hits == 0:
        # Agent didn't reference any observable signals — neutral
        return 0.4

    # Grounded fraction
    grounded_ratio = grounded / total_hits
    score = 0.7 * grounded_ratio

    # Bonus: agent used log signals that directly correspond to root cause
    cause_synonyms = get_cause_synonyms(ground_truth.root_cause_type)
    cause_in_logs   = any(term in log_lower   for term in cause_synonyms)
    cause_in_answer = any(term in answer_lower for term in cause_synonyms)
    if cause_in_logs and cause_in_answer:
        score += 0.3  # Agent correctly connected log signal to root cause

    # Penalty: hallucinated causes (strong signals not in logs)
    hallucination_penalty = min(hallucinated * 0.15, 0.4)
    score = max(0.0, score - hallucination_penalty)

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL E — Noise Handling Score
# ─────────────────────────────────────────────────────────────────────────────

def noise_signal(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Evaluate whether agent correctly ignored red-herring services/signals.

    Scoring:
        No red herrings mentioned                    → 1.0
        1 red herring mentioned                      → 0.5 penalty
        2+ red herrings mentioned                    → heavy penalty
        Over-guessing phrases detected               → additional penalty

    Bonus: if agent explicitly says a red herring is NOT the cause,
           that shows good reasoning and is rewarded.
    """
    if not answer or not answer.strip():
        return 0.5  # neutral — we can't tell

    answer_lower = answer.lower()
    score = 1.0

    red_herrings = ground_truth.red_herrings or []

    for rh in red_herrings:
        rh_lower = rh.lower()
        if rh_lower in answer_lower:
            # Check if agent explicitly ruled it out
            # e.g. "frontend-service is not the cause"
            ruled_out = any(
                f"{rh_lower} is not" in answer_lower
                or f"not {rh_lower}" in answer_lower
                or f"ruled out {rh_lower}" in answer_lower
                or f"{rh_lower} does not" in answer_lower
                for _ in [1]  # single iteration trick for early-exit
            )
            if ruled_out:
                score += 0.05  # Small bonus for explicit ruling out
            else:
                score -= 0.3   # Agent incorrectly blamed this service

    # Over-guessing penalty
    over_guess_count = sum(1 for p in OVER_GUESS_PHRASES if p in answer_lower)
    if over_guess_count >= 3:
        score -= 0.3
    elif over_guess_count >= 1:
        score -= 0.1

    return round(max(0.0, min(score, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL F — Efficiency Score
# ─────────────────────────────────────────────────────────────────────────────

def efficiency_signal(steps_used: int, max_steps: int) -> float:
    """
    Evaluate how efficiently the agent reached its conclusion.

    Scoring curve (non-linear — rewards decisiveness):
        ≤ 25% of budget used   → 1.0
        ≤ 50% of budget used   → 0.8
        ≤ 75% of budget used   → 0.6
        ≤ 90% of budget used   → 0.4
        100% of budget used    → 0.2
        > budget (capped)      → 0.0

    A non-linear curve is used (vs the old linear bonus) because real
    incident responders who resolve issues in 2 of 20 steps are
    meaningfully better than those who need 10 of 20 steps.
    """
    if max_steps <= 0:
        return 0.5  # neutral — no budget defined

    ratio = steps_used / max_steps

    if ratio <= 0.0:
        return 1.0
    elif ratio <= 0.25:
        return 1.0
    elif ratio <= 0.50:
        return 0.8
    elif ratio <= 0.75:
        return 0.6
    elif ratio <= 0.90:
        return 0.4
    elif ratio <= 1.0:
        return 0.2
    else:
        return 0.0  # Went over budget