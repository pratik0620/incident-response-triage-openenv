# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incident Response Triage Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import (
    IncidentResponseTriageAction,
    IncidentResponseTriageObservation,
    IncidentResponseTriageState,
)


class IncidentResponseTriageEnv(
    EnvClient[
        IncidentResponseTriageAction,
        IncidentResponseTriageObservation,
        IncidentResponseTriageState,
    ]
):
    """
    Client for the Incident Response Triage Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with IncidentResponseTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.step)
        ...
        ...     result = client.step(
        ...         IncidentResponseTriageAction(
        ...             action_type="identify_cause",
        ...             reasoning="Payment-service is timing out under load.",
        ...             answer="payment-service db connection pool exhaustion",
        ...         )
        ...     )
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = IncidentResponseTriageEnv.from_docker_image("incident_response_triage_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         IncidentResponseTriageAction(
        ...             action_type="propose_fix",
        ...             reasoning="Restart payment-service to clear pool exhaustion.",
        ...             answer="kubectl rollout restart deploy/payment-service",
        ...         )
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: IncidentResponseTriageAction) -> Dict:
        """
        Convert IncidentResponseTriageAction to JSON payload for step message.

        Args:
            action: IncidentResponseTriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "service": action.service,
            "reasoning": action.reasoning,
            "answer": action.answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[IncidentResponseTriageObservation]:
        """
        Parse server response into StepResult[IncidentResponseTriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with IncidentResponseTriageObservation
        """
        obs_data = payload.get("observation", {})
        observation = IncidentResponseTriageObservation(
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 0),
            logs=obs_data.get("logs", []),
            metrics=obs_data.get("metrics", []),
            alerts=obs_data.get("alerts", []),
            previous_actions=obs_data.get("previous_actions", []),
            task_description=obs_data.get("task_description", ""),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            final_score=obs_data.get("final_score"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> IncidentResponseTriageState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step data
        """
        return IncidentResponseTriageState(
            episode_id=payload.get("episode_id"),
            step=payload.get("step", 0),
            max_steps=payload.get("max_steps", 0),
            previous_actions=payload.get("previous_actions", []),
            done=payload.get("done", False),
            final_score=payload.get("final_score"),
        )
