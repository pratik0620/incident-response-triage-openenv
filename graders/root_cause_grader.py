# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Root Cause Grader — standalone wrapper around Signal A.

Kept as a separate file for backward compatibility with environment.py
and any existing tests that import grade_root_cause directly.

Upgrade from v1:
  - Now uses synonym expansion (get_cause_synonyms) so "heap overflow"
    matches "memory_leak" without needing an exact keyword.
  - Service matching accepts partial names (e.g. "payment" → "payment-service").
  - Red-herring penalty is still -0.3 per service, clamped to 0.
  - Scores are still in [0.0, 1.0], fully deterministic.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .signals import root_cause_signal

if TYPE_CHECKING:
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
    return root_cause_signal(answer, ground_truth)