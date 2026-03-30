# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Incident Response Triage Env Environment.

These models define the action and observation payloads used by the environment.
"""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

try:
    from .server.scenario_models import Alert, LogEntry, MetricSnapshot
except ImportError:
    from server.scenario_models import Alert, LogEntry, MetricSnapshot


class IncidentResponseTriageAction(Action):
    """Action payload submitted by an agent to triage an incident."""

    root_cause_service: str = Field(..., description="Predicted root-cause service")
    root_cause_type: str = Field(..., description="Predicted root-cause type")
    fix_command: str = Field(..., description="Proposed remediation command")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Agent confidence in the triage decision"
    )


class IncidentResponseTriageObservation(Observation):
    """Observation payload returned by the incident response triage environment."""

    logs: list[LogEntry] = Field(default_factory=list)
    metrics: list[MetricSnapshot] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
