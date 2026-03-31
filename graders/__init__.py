# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders package for the Incident Response Triage environment.

Exposes the single entry point used by the environment:
    compute_final_score(action_type, answer, ground_truth, steps_used, max_steps, difficulty)
"""

from .composite_grader import compute_final_score

__all__ = ["compute_final_score"]