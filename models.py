from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Literal, Optional
from scenarios.schema import Alert, LogEntry, MetricSnapshot, GroundTruth

__all__ = [
    "IncidentResponseTriageAction",
    "IncidentResponseTriageObservation",
    "IncidentResponseTriageState",
    "GroundTruth",
]

class IncidentResponseTriageAction(Action):
    """
    The actions an agent is allowed to take when checking an incident
    """
    action_type: Literal[
        "read_logs", "check_metrics", "identify_cause", "propose_fix", "escalate"
    ]
    service: Optional[str] = None
    reasoning: str
    answer: Optional[str] = None


class IncidentResponseTriageObservation(Observation):
    """
    The current state of the environment/incident visible or known to the agent
    """
    step: int
    max_steps: int
    logs: list[LogEntry]
    metrics: list[MetricSnapshot]
    alerts: list[Alert]
    previous_actions: list[str] = Field(default_factory=list)
    task_description: str
    reward: float = 0.0
    done: bool = False
    final_score : Optional[float] = None


class IncidentResponseTriageState(State):
    """
    The internal state of an environment for a single episode.

    Attributes:
        step (int):
            Current step number in the episode.

        max_steps (int):
            Maximum number of steps allowed before the episode terminates.

        previous_actions (list[str]):
            History of actions taken by the agent in order.

        done (bool):
            Indicates whether the episode has ended.

        final_score (Optional[float]):
            Final evaluation score assigned at episode termination (0.0–1.0).
    """
    step: int = 0
    max_steps: int = 0
    previous_actions: list[str] = Field(default_factory=list)
    done: bool = False
    final_score: Optional[float] = None
