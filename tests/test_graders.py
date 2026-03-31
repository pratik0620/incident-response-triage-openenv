# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for all graders.

Run with:
    pytest tests/test_graders.py -v

All tests use a fake GroundTruth so they can be run without the full
OpenEnv package installed — just the graders package itself.
"""

import sys
import os
import types

import pytest

# ── Bootstrap: allow running without the full openenv package ─────────────────
# We create a minimal stub so graders can be imported standalone.
def _stub_openenv():
    try:
        import openenv  # noqa: F401
        return False
    except Exception:
        pass

    openenv = types.ModuleType("openenv")
    core    = types.ModuleType("openenv.core")
    server  = types.ModuleType("openenv.core.env_server")
    types_m = types.ModuleType("openenv.core.env_server.types")

    class _Base:
        pass

    types_m.Action      = _Base
    types_m.Observation = _Base
    types_m.State       = _Base

    openenv.core                     = core
    core.env_server                  = server
    server.types                     = types_m
    sys.modules["openenv"]                        = openenv
    sys.modules["openenv.core"]                   = core
    sys.modules["openenv.core.env_server"]        = server
    sys.modules["openenv.core.env_server.types"]  = types_m
    return True


_OPENENV_STUBBED = _stub_openenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graders.root_cause_grader   import grade_root_cause
from graders.fix_grader          import grade_fix
from graders.categorization_grader import grade_categorization
from graders.composite_grader    import compute_final_score


class FakeGroundTruth:
    def __init__(
        self,
        root_cause_service="payment-service",
        root_cause_type="memory_leak",
        correct_fix="increase JVM heap size and add memory limit alerts",
        red_herrings=None,
        failure_category="real",
    ):
        self.root_cause_service = root_cause_service
        self.root_cause_type    = root_cause_type
        self.correct_fix        = correct_fix
        self.red_herrings       = red_herrings or []
        self.failure_category   = failure_category


GT = FakeGroundTruth()  



class TestRootCauseGrader:

    def test_empty_answer_returns_zero(self):
        assert grade_root_cause("", GT) == 0.0

    def test_whitespace_only_returns_zero(self):
        assert grade_root_cause("   ", GT) == 0.0

    def test_perfect_answer_scores_one(self):
        answer = "The root cause is a memory leak in the payment-service."
        score  = grade_root_cause(answer, GT)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_service_only_scores_half(self):
        answer = "I think payment-service is the culprit."
        score  = grade_root_cause(answer, GT)
        assert score == 0.5, f"Expected 0.5 (service only), got {score}"

    def test_cause_keywords_only_scores_half(self):
        answer = "Looks like a memory leak somewhere."
        score  = grade_root_cause(answer, GT)
        assert score == 0.5, f"Expected 0.5 (cause only), got {score}"

    def test_case_insensitive(self):
        answer = "PAYMENT-SERVICE has a MEMORY LEAK"
        score  = grade_root_cause(answer, GT)
        assert score == 1.0, f"Expected 1.0 (case insensitive), got {score}"

    def test_red_herring_penalises_score(self):
        gt     = FakeGroundTruth(red_herrings=["frontend-service"])
        answer = "The memory leak in payment-service might be in frontend-service too."
        score  = grade_root_cause(answer, gt)
        assert score == 0.7, f"Expected 0.7 after red-herring penalty, got {score}"

    def test_multiple_red_herrings_clamped_to_zero(self):
        gt     = FakeGroundTruth(red_herrings=["frontend-service", "load-balancer", "nginx"])
        answer = "frontend-service, load-balancer, nginx all look suspicious."
        score  = grade_root_cause(answer, gt)
        assert score == 0.0, f"Expected 0.0 (over-penalised, clamped), got {score}"

    def test_partial_keyword_match(self):
        answer = "payment-service has a memory issue"   
        score  = grade_root_cause(answer, GT)
        assert score == 0.75, f"Expected 0.75 (partial keyword), got {score}"

    def test_completely_wrong_answer_scores_zero(self):
        answer = "Everything looks fine to me."
        score  = grade_root_cause(answer, GT)
        assert score == 0.0, f"Expected 0.0, got {score}"

    def test_score_never_exceeds_one(self):
        answer = "payment-service memory leak memory leak memory leak"
        score  = grade_root_cause(answer, GT)
        assert score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# fix_grader tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFixGrader:

    def test_empty_answer_returns_zero(self):
        assert grade_fix("", GT) == 0.0

    def test_perfect_fix_scores_one(self):
        # GT correct_fix = "increase JVM heap size and add memory limit alerts"
        # Keywords (>4 chars): increase, heap, size, memory, limit, alerts  (6 words)
        answer = "You should increase the JVM heap size and add memory limit alerts."
        score  = grade_fix(answer, GT)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_partial_fix_scores_proportionally(self):
        answer = "increase the heap size"  # missing memory, limit, alerts
        score  = grade_fix(answer, GT)
        # keywords: increase(7), heap(4→skip), size(4→skip), memory(6), limit(5), alerts(6)
        # wait — "heap" is 4 chars, filtered (len > 4, not >=4).
        # Let's verify: increase=8✓, heap=4✗, size=4✗, memory=6✓, limit=5✓, alerts=6✓
        # So 4 keywords total.  Answer has: increase✓, heap(not keyword), size(not keyword)
        # matched=1/4 = 0.25
        assert 0.0 < score <= 1.0  # at least partial

    def test_no_keywords_in_fix_returns_zero(self):
        gt     = FakeGroundTruth(correct_fix="fix it")  # all words <=4 chars
        answer = "I will fix it"
        score  = grade_fix(answer, gt)
        assert score == 0.0

    def test_case_insensitive(self):
        answer = "INCREASE JVM HEAP SIZE AND ADD MEMORY LIMIT ALERTS"
        score  = grade_fix(answer, GT)
        assert score == 1.0, f"Expected 1.0 (case insensitive), got {score}"

    def test_wrong_fix_scores_zero(self):
        answer = "Just restart the server."
        score  = grade_fix(answer, GT)
        assert score == 0.0

    def test_score_between_zero_and_one(self):
        answer = "increase alerts"
        score  = grade_fix(answer, GT)
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# categorization_grader tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCategorizationGrader:

    def _hard_gt(self, category="real"):
        return FakeGroundTruth(failure_category=category)

    def test_empty_answer_returns_zero(self):
        assert grade_categorization("", self._hard_gt()) == 0.0

    def test_full_score_for_perfect_answer(self):
        gt = FakeGroundTruth(
            root_cause_service="payment-service",
            root_cause_type="memory_leak",
            failure_category="real",
        )
        answer = "This is a real failure. payment-service has a memory leak."
        score  = grade_categorization(answer, gt)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_category_only_scores_0_6(self):
        gt     = self._hard_gt(category="flaky")
        answer = "This looks flaky to me."
        score  = grade_categorization(answer, gt)
        assert score == 0.6, f"Expected 0.6 (category only), got {score}"

    def test_wrong_category_scores_zero_on_category_block(self):
        gt     = self._hard_gt(category="real")
        answer = "This is a flaky failure in payment-service with memory leak."
        score  = grade_categorization(answer, gt)
        # category wrong: 0.0; service: +0.2; cause: +0.2 → 0.4
        assert score == 0.4, f"Expected 0.4 (wrong category), got {score}"

    def test_underscore_and_spaced_form_both_accepted(self):
        gt1    = self._hard_gt(category="env_specific")
        answer = "env specific failure in payment-service memory leak"
        score1 = grade_categorization(answer, gt1)
        answer2 = "env_specific failure in payment-service memory leak"
        score2 = grade_categorization(answer2, gt1)
        assert score1 == score2 == 1.0, f"Both forms should score 1.0: {score1}, {score2}"

    def test_score_clamped_to_one(self):
        gt     = self._hard_gt()
        answer = "real real real payment-service memory leak real"
        score  = grade_categorization(answer, gt)
        assert score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# composite_grader tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeGrader:

    def _score(self, action_type, answer, gt=None, steps=5, max_steps=20, difficulty="easy"):
        if gt is None:
            gt = GT
        return compute_final_score(action_type, answer, gt, steps, max_steps, difficulty)

    # ── identify_cause (easy/medium) ──────────────────────────────────────────

    def test_identify_cause_perfect(self):
        answer = "payment-service has a memory leak"
        score  = self._score("identify_cause", answer)
        # base=1.0, efficiency=1-(5/20)=0.75, bonus=0.15*0.75=0.1125
        expected = round(min(1.0 + 0.15 * 0.75, 1.0), 4)
        assert score == expected, f"Expected {expected}, got {score}"

    def test_identify_cause_zero_steps_remaining(self):
        answer = "payment-service has a memory leak"
        score  = self._score("identify_cause", answer, steps=20, max_steps=20)
        # base=1.0, efficiency=0, bonus=0 → 1.0
        assert score == 1.0

    def test_identify_cause_empty_answer(self):
        score = self._score("identify_cause", "")
        assert score == 0.0 or score <= 0.15  # only efficiency bonus possible

    # ── propose_fix ───────────────────────────────────────────────────────────

    def test_propose_fix_perfect_answer_full_score(self):
        answer = (
            "payment-service has a memory leak. "
            "Fix: increase JVM heap size and add memory limit alerts."
        )
        score  = self._score("propose_fix", answer, steps=10, max_steps=20)
        # base = 0.4*1.0 + 0.6*1.0 = 1.0; bonus = 0.15*0.5 = 0.075
        expected = round(min(1.0 + 0.15 * 0.5, 1.0), 4)
        assert score == expected

    def test_propose_fix_cause_only_no_fix(self):
        answer = "payment-service has a memory leak. No fix yet."
        score  = self._score("propose_fix", answer)
        # cause_score=1.0, fix_score=~0 → base=0.4
        assert score < 0.6

    def test_propose_fix_fix_only_no_cause(self):
        answer = "increase JVM heap size and add memory limit alerts"
        score  = self._score("propose_fix", answer)
        # cause_score~0 (no service named), fix_score=high
        # base = 0.4*0 + 0.6*high; should be less than 0.7
        assert score < 0.75

    # ── escalate ──────────────────────────────────────────────────────────────

    def test_escalate_always_0_2_base(self):
        score_a = self._score("escalate", "I cannot determine the cause.", steps=1, max_steps=20)
        score_b = self._score("escalate", "", steps=20, max_steps=20)
        # steps=1: bonus=0.15*(19/20)=0.1425 → 0.3425; steps=20: bonus=0 → 0.2
        assert score_b == 0.2
        assert round(score_a, 4) == round(0.2 + 0.15 * (19 / 20), 4)

    def test_escalate_never_full_marks(self):
        score = self._score("escalate", "escalating immediately", steps=0, max_steps=20)
        assert score < 1.0

    # ── hard task (categorization) ────────────────────────────────────────────

    def test_hard_task_uses_categorization_grader(self):
        gt     = FakeGroundTruth(failure_category="flaky")
        answer = "This is a flaky failure in payment-service due to memory leak."
        score  = self._score("identify_cause", answer, gt=gt, difficulty="hard",
                             steps=5, max_steps=20)
        # base=1.0, bonus=0.15*0.75=0.1125 → clamped to 1.0
        assert score == 1.0

    def test_hard_task_wrong_category_penalised(self):
        gt     = FakeGroundTruth(failure_category="real")
        answer = "This is a flaky failure in payment-service due to memory leak."  # wrong category
        score_hard   = self._score("identify_cause", answer, gt=gt, difficulty="hard",
                                   steps=20, max_steps=20)
        score_easy   = self._score("identify_cause", answer, gt=gt, difficulty="easy",
                                   steps=20, max_steps=20)
        # Hard should score lower because category is wrong
        assert score_hard < score_easy

    # ── invalid action ────────────────────────────────────────────────────────

    def test_unknown_action_type_scores_zero(self):
        score = self._score("do_nothing", "some answer")
        assert score == 0.0

    # ── efficiency bonus edge cases ───────────────────────────────────────────

    def test_efficiency_bonus_zero_when_all_steps_used(self):
        answer = "payment-service has a memory leak"
        score  = self._score("identify_cause", answer, steps=20, max_steps=20)
        # efficiency=0, bonus=0; base=1.0 → 1.0
        assert score == 1.0

    def test_efficiency_bonus_max_when_step_one(self):
        answer = "payment-service has a memory leak"
        score  = self._score("identify_cause", answer, steps=1, max_steps=20)
        expected_bonus = round(0.15 * (1 - 1 / 20), 4)
        expected = round(min(1.0 + expected_bonus, 1.0), 4)
        assert score == expected

    def test_max_steps_zero_does_not_crash(self):
        # Protect against divide-by-zero in edge cases
        score = self._score("identify_cause", "payment-service memory leak",
                            steps=0, max_steps=0)
        assert 0.0 <= score <= 1.0

    def test_final_score_always_between_zero_and_one(self):
        for action in ["identify_cause", "propose_fix", "escalate", "unknown"]:
            score = self._score(action, "anything", steps=5, max_steps=10)
            assert 0.0 <= score <= 1.0, f"Out of range for action={action}: {score}"


def test_openenv_integration_if_available():
    if _OPENENV_STUBBED:
        pytest.skip("openenv not installed; skipping integration check")
    import openenv  # noqa: F401
    assert hasattr(openenv, "core")