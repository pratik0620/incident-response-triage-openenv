from typing import Literal

from pydantic import BaseModel


class LogEntry(BaseModel):
    timestamp: str
    service: str
    message: str
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]


class MetricSnapshot(BaseModel):
    service: str
    cpu_pct: float
    mem_pct: float
    error_rate: float
    latency_p99_ms: float


class Alert(BaseModel):
    # Renamed from action_id to better represent incoming alert identifiers.
    alert_id: str
    service: str
    severity: Literal["P1", "P2", "P3", "P4"]
    message: str
    fired_at: str


class GroundTruth(BaseModel):
    root_cause_service: str
    root_cause_type: str
    correct_fix: str
    failure_category: Literal["real", "intermittent", "env_specific"]
    severity: Literal["P1", "P2", "P3", "P4"]
    red_herrings: list[str]


class Scenario(BaseModel):
    scenario_id: str
    # Renamed from difficult to difficulty for schema clarity.
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    logs: list[LogEntry]
    metrics: list[MetricSnapshot]
    alerts: list[Alert]
    ground_truth: GroundTruth
    max_steps: int
