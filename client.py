# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incident Response Triage Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IncidentResponseTriageAction, IncidentResponseTriageObservation


class IncidentResponseTriageEnv(
    EnvClient[IncidentResponseTriageAction, IncidentResponseTriageObservation, State]
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
        ...     print(result.observation.step_count)
        ...
        ...     result = client.step(
        ...         IncidentResponseTriageAction(
        ...             root_cause_service="payment-service",
        ...             root_cause_type="db_connection_pool_exhaustion",
        ...             fix_command="kubectl rollout restart deploy/payment-service",
        ...             confidence=0.9,
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
        ...             root_cause_service="payment-service",
        ...             root_cause_type="db_connection_pool_exhaustion",
        ...             fix_command="kubectl rollout restart deploy/payment-service",
        ...             confidence=0.8,
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
            "root_cause_service": action.root_cause_service,
            "root_cause_type": action.root_cause_type,
            "fix_command": action.fix_command,
            "confidence": action.confidence,
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
            logs=obs_data.get("logs", []),
            metrics=obs_data.get("metrics", []),
            alerts=obs_data.get("alerts", []),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
