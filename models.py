from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Literal, Optional
from scenarios.schema import Alert, LogEntry, MetricSnapshot

class IncidentResponseTriageAction(Action):
    action_type: Literal[
        "read_logs", "check_metrics", "identify_cause", "propose_fix", "escalate"
    ]
    service: Optional[str] = None
    reasoning: str
    answer: Optional[str] = None


class IncidentResponseTriageObservation(Observation):
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

