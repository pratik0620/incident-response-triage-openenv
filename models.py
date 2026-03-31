# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Incident Response Triage Environment.

Three Pydantic models live here:

  GroundTruth                — the hidden answer for each scenario.
                               Held server-side; never sent to the agent.

  IncidentResponseTriageAction — what the agent sends on each step.
                               Replaces the old echo-only "message" action.

  IncidentResponseTriageObservation — what the environment sends back.
                               Includes the incident context, available tools,
                               current step count, and (at episode end) reward.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


FailureCategory = Literal["real", "flaky", "env_specific"]

ActionType = Literal["identify_cause", "propose_fix", "escalate", "query"]

class GroundTruth(BaseModel):
    """
    Hidden ground-truth answer for a single incident scenario.

    This object is created when a scenario is loaded and is NEVER exposed
    to the agent through observations.  The graders receive it directly
    from the environment's internal state.

    Fields:
        root_cause_service  — name of the service that is the root cause.
                              e.g. "payment-service", "auth-db", "redis-cache"

        root_cause_type     — category of the failure, using snake_case.
                              e.g. "memory_leak", "connection_timeout",
                                   "config_error", "disk_full"

        correct_fix         — a short human-readable description of the fix.
                              Grader extracts keywords (>4 chars) from this.
                              e.g. "increase JVM heap size to 4 GB and add
                                   memory limit alerts"

        red_herrings        — list of service names that show symptoms but are
                              NOT the root cause.  Naming these in the answer
                              triggers a -0.3 penalty per service.
                              e.g. ["frontend-service", "load-balancer"]

        failure_category    — used by the hard task only.
                              One of: "real", "flaky", "env_specific"
    """

    root_cause_service: str = Field(
        ...,
        description="Name of the service that is the true root cause.",
    )
    root_cause_type: str = Field(
        ...,
        description="Snake_case category of the failure (e.g. 'memory_leak').",
    )
    correct_fix: str = Field(
        ...,
        description="Human-readable description of the correct remediation.",
    )
    red_herrings: List[str] = Field(
        default_factory=list,
        description="Service names that show symptoms but are NOT root causes.",
    )
    failure_category: FailureCategory = Field(
        default="real",
        description="Hard-task only: 'real', 'flaky', or 'env_specific'.",
    )


class IncidentResponseTriageAction(Action):
    """
    Action sent by the agent to the environment on each step.

    action_type choices:
      "query"           — non-terminal: ask for more information.
                          answer field contains the query string.
                          Environment responds with relevant log/metric data.

      "identify_cause"  — TERMINAL: agent declares the root cause.
                          answer field contains the free-text diagnosis.
                          Episode ends; grader scores the answer.

      "propose_fix"     — TERMINAL: agent declares root cause AND proposed fix.
                          answer field must mention both for full marks.
                          Episode ends; grader scores the answer.

      "escalate"        — TERMINAL: agent admits it cannot diagnose.
                          Scores a flat 0.2 regardless of answer content.
                          Episode ends immediately.
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "Type of action: 'query' (non-terminal) or "
            "'identify_cause' / 'propose_fix' / 'escalate' (terminal)."
        ),
    )
    answer: str = Field(
        ...,
        description=(
            "For terminal actions: the agent's diagnosis / fix / escalation reason. "
            "For 'query': the question the agent wants answered."
        ),
    )



class IncidentResponseTriageObservation(Observation):
    """
    Observation returned by the environment after each step or reset.

    At episode start (reset):
      - incident_title, incident_description, available_tools are populated.
      - logs, metrics, alerts contain the initial context.
      - done=False, reward=0.0.

    After a 'query' action:
      - query_response contains the answer to the agent's question.
      - step_count increments.
      - done=False.

    After a terminal action (identify_cause / propose_fix / escalate):
      - done=True.
      - reward contains the final episode score (0.0–1.0).
      - feedback explains what was correct / incorrect (for debugging).
    """

    incident_title: str = Field(
        default="",
        description="Short title of the incident (e.g. 'Payment service 500 errors spike').",
    )
    incident_description: str = Field(
        default="",
        description="Narrative description of the incident for the agent.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description=(
            "Tools the agent can query. "
            "e.g. ['logs', 'metrics', 'alerts', 'runbook', 'deployment_history']"
        ),
    )

    logs: List[str] = Field(
        default_factory=list,
        description="Log lines relevant to the incident.",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key metrics snapshot, e.g. {'error_rate': 0.42, 'latency_p99': 3200}.",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Active alert strings from the monitoring system.",
    )

    query_response: str = Field(
        default="",
        description="Answer to the agent's most recent 'query' action.",
    )

    step_count: int = Field(
        default=0,
        description="How many steps the agent has taken so far this episode.",
    )
    max_steps: int = Field(
        default=20,
        description="Total step budget for this episode.",
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: 'easy', 'medium', or 'hard'.",
    )

    feedback: str = Field(
        default="",
        description=(
            "Human-readable feedback explaining the score. "
            "Only present when done=True."
        ),
    )