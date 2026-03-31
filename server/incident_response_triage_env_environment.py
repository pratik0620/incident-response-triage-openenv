from __future__ import annotations
from uuid import uuid4
import json
import os
import random


from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


try:
    from ..models import IncidentResponseTriageAction, IncidentResponseTriageObservation, IncidentResponseTriageState
except ImportError:
    from models import IncidentResponseTriageAction, IncidentResponseTriageObservation, IncidentResponseTriageState

from scenarios.schema import Scenario
from server.scenario_loader import load_random_scenario

class IncidentResponseTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True


    def __init__(self):
        """Initialize the incident_response_triage_env environment."""
        self._state = IncidentResponseTriageState(episode_id=str(uuid4()), step=0)
        self.scenarios: list[Scenario] = []
        self.current_scenario: Scenario | None = None


    def reset(self, difficulty: str = "easy") -> IncidentResponseTriageObservation:
        self.current_scenario = load_random_scenario(difficulty)

        self._state = IncidentResponseTriageState(
            episode_id=str(uuid4()),
            step=0,
            max_steps=self.current_scenario.max_steps,
            previous_actions=[],
            done=False,
        )

        return self._build_observation(reward=0.0, done=False)


    def step(self, action: IncidentResponseTriageAction) -> IncidentResponseTriageObservation:  # type: ignore[override]
        if self._state.done:
            return self._build_observation(reward=0.0, done=True, msg="Episode ended.")

        action_type = (action.action_type or "").strip().lower()

        self._state.step += 1
        self._state.previous_actions.append(action_type)

        if action_type in ["identify_cause", "propose_fix", "escalate"]:
            return self._handle_terminal(action_type, action)

        reward = -0.02

        if self._state.step >= self._state.max_steps:
            return self._force_end()

        return self._build_observation(reward=reward, done=False)


    def _handle_terminal(self, action_type: str, action):
        """Temporary function until graders are implemented"""

        if action_type == "escalate":
            score = 0.2
        else:
            score = 0.5

        self._state.done = True
        self._state.final_score = score

        return self._build_observation(
            reward=score,
            done=True,
            final_score=score,
        )


    def _force_end(self):
        self._state.done = True
        return self._build_observation(
            reward=0.0,
            done=True,
            msg="Max steps crossed. Episode ended."
        )


    def _build_observation(self, reward, done, msg=None, final_score=None):
        if not self.current_scenario:
            raise ValueError("Call reset() before step().")
        scenario = self.current_scenario

        return IncidentResponseTriageObservation(
            step=self._state.step,
            max_steps=scenario.max_steps,
            logs=scenario.logs,
            metrics=scenario.metrics,
            alerts=scenario.alerts,
            previous_actions=self._state.previous_actions,
            task_description=scenario.description,
            reward=reward,
            done=done,
            final_score=final_score,
        )


    @property
    def state(self) -> State:
        return self._state


if __name__ == "__main__":
    env = IncidentResponseTriageEnvironment()

    # reset() returns the initial observation.
    observation = env.reset()
    print(
        "reset -> "
        f"logs={len(observation.logs)}, metrics={len(observation.metrics)}, alerts={len(observation.alerts)}"
    )

    # Make the test deterministic for the hardcoded correct action below.
    env.current_scenario = load_scenario("scenario_001")

    correct_action = IncidentResponseTriageAction(
        root_cause_service="payment-service",
        root_cause_type="db_connection_pool_exhaustion",
        fix_command="kubectl rollout restart deploy/payment-service",
        confidence=0.95,
    )
    # step() follows Gym-style return format: (observation, reward, done, info).
    observation, reward, done, info = env.step(correct_action)
    print(f"correct action reward={reward}, done={done}, breakdown={info['breakdown']}")
    assert reward >= 1

    observation = env.reset()
    env.current_scenario = load_scenario("scenario_001")

    incorrect_action = IncidentResponseTriageAction(
        root_cause_service="auth-service",
        root_cause_type="token_issuer_outage",
        fix_command="kubectl rollout restart deploy/auth-service",
        confidence=0.15,
    )
    # Unpack tuple outputs instead of using attribute access on a result object.
    observation, reward, done, info = env.step(incorrect_action)
    print(f"incorrect action reward={reward}, done={done}, breakdown={info['breakdown']}")
    assert reward == -3
