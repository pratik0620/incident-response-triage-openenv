from __future__ import annotations
from uuid import uuid4


from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


try:
    from ..models import (
        IncidentResponseTriageAction,
        IncidentResponseTriageObservation,
        IncidentResponseTriageState,
    )
    from ..graders.composite_grader import compute_final_score
except ImportError:
    from models import (
        IncidentResponseTriageAction,
        IncidentResponseTriageObservation,
        IncidentResponseTriageState,
    )
    from graders.composite_grader import compute_final_score

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
        if not self.current_scenario:
            raise ValueError("Call reset() before step().")

        answer_text = " ".join(
            part for part in [action.answer, action.reasoning] if part
        ).strip()
        score = compute_final_score(
            action_type=action_type,
            answer=answer_text,
            ground_truth=self.current_scenario.ground_truth,
            steps_used=self._state.step,
            max_steps=self.current_scenario.max_steps,
            difficulty=self.current_scenario.difficulty,
        )

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
