# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders package — Multi-Signal Evaluation Framework v2.

Primary entry point for environment.py:
    compute_final_score(action_type, answer, ground_truth,
                        steps_used, max_steps, difficulty,
                        log_content="") -> float

Debug/logging entry point:
    compute_score_breakdown(...) -> dict  (all six signal scores)

Individual graders (backward compatible):
    grade_root_cause(answer, ground_truth) -> float
    grade_fix(answer, ground_truth) -> float
    grade_categorization(answer, ground_truth) -> float
"""

from .composite_grader import compute_final_score, compute_score_breakdown
from .root_cause_grader import grade_root_cause
from .fix_grader import grade_fix
from .categorization_grader import grade_categorization

__all__ = [
    "compute_final_score",
    "compute_score_breakdown",
    "grade_root_cause",
    "grade_fix",
    "grade_categorization",
]