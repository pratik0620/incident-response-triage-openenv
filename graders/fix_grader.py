# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fix Grader — standalone wrapper around Signal B.

Kept as a separate file for backward compatibility with environment.py
and any existing tests that import grade_fix directly.

Upgrade from v1:
  - Now uses tiered scoring (0.3 / 0.7 / 1.0) instead of flat keyword overlap.
  - "Restart the service" now correctly scores low (~0.3) not high.
  - Cause keywords are no longer filtered from fix keywords — that caused
    false negatives when a good fix naturally restates the cause.
  - Scores still in [0.0, 1.0], fully deterministic.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .signals import fix_quality_signal

if TYPE_CHECKING:
    from ..models import GroundTruth


def grade_fix(answer: str, ground_truth: "GroundTruth") -> float:
    """
    Score whether the proposed fix is correct and complete.

    Args:
        answer:       The agent's free-text fix proposal.
        ground_truth: A GroundTruth instance for the current scenario.

    Returns:
        float in [0.0, 1.0] — higher is better.
    """
    return fix_quality_signal(answer, ground_truth)