# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


    @staticmethod
    def _normalize_score(score: float) -> float:
        """
        Enforce strictly open (0, 1) range for final_score.
        """
        if score is None:
            return 0.01

        val = float(score)
        if val <= 0.0:
            return 0.01
        if val >= 1.0:
            return 0.99
        
        rounded = round(val, 4)
        if rounded <= 0.0:
            return 0.01
        if rounded >= 1.0:
            return 0.99
        return rounded

    def reset(
        self,
        difficulty: str = "easy",
        task_id: str | None = None,
    ) -> IncidentResponseTriageObservation:
        """Reset the environment for a new episode.

        Args:
            difficulty: Scenario difficulty — "easy", "medium", or "hard".
            task_id:    OpenEnv task identifier (mirrors difficulty for this
                        environment).  When provided, it overrides `difficulty`
                        so that the HuggingFace platform's task-based resets
                        (``{"task_id": "easy"}``) resolve correctly.
        """
        # task_id mirrors difficulty in this environment (easy/medium/hard)
        effective_difficulty = task_id if task_id is not None else difficulty
        self.current_scenario = load_random_scenario(effective_difficulty)

        self._state = IncidentResponseTriageState(
            episode_id=str(uuid4()),
            step=0,
            max_steps=self.current_scenario.max_steps,
            previous_actions=[],
            done=False,
        )

        return self._build_observation(reward=0.0001, done=False)


    def step(self, action: IncidentResponseTriageAction) -> IncidentResponseTriageObservation:  # type: ignore[override]
        if self.current_scenario is None:
            self.reset(difficulty="easy")

        if self._state.done:
            return self._build_observation(
                reward=self._normalize_score(self._state.final_score or 0.0001),
                done=True,
                msg="Episode ended.",
                final_score=self._normalize_score(self._state.final_score or 0.01),
            )

        action_type = (action.action_type or "").strip().lower()

        self._state.step += 1
        self._state.previous_actions.append(action_type)

        if action_type in ["identify_cause", "propose_fix", "escalate"]:
            return self._handle_terminal(action_type, action)

        reward = 0.0001

        if self._state.step >= self._state.max_steps:
            return self._force_end()

        return self._build_observation(reward=reward, done=False)


    def _handle_terminal(self, action_type: str, action):
        if not self.current_scenario:
            raise ValueError("Call reset() before step().")

        answer_text = " ".join(
            part for part in [action.answer, action.reasoning] if part
        ).strip()

        log_content = " ".join(
            f"{l.service} {l.level} {l.message}"
            for l in self.current_scenario.logs
        )
        log_content += " " + " ".join(
            f"{a.service} {a.message}"
            for a in self.current_scenario.alerts
        )

        raw_score = compute_final_score(
            action_type=action_type,
            answer=answer_text,
            ground_truth=self.current_scenario.ground_truth,
            steps_used=self._state.step,
            max_steps=self.current_scenario.max_steps,
            difficulty=self.current_scenario.difficulty,
            log_content=log_content,
        )
        score = self._normalize_score(raw_score)

        self._state.done = True
        self._state.final_score = score

        return self._build_observation(
            reward=score,
            done=True,
            final_score=score,
        )


    def _force_end(self):
        self._state.done = True
        # Score as escalate — agent ran out of steps without diagnosing
        raw_score = compute_final_score(
            action_type="escalate",
            answer="",
            ground_truth=self.current_scenario.ground_truth,
            steps_used=self._state.max_steps,
            max_steps=self._state.max_steps,
            difficulty=self.current_scenario.difficulty,
            log_content="",
        )
        score = self._normalize_score(raw_score)
        self._state.final_score = score
        return self._build_observation(
            reward=score,
            done=True,
            final_score=score,
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
