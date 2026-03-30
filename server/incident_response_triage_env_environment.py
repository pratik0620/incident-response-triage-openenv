from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IncidentResponseTriageAction, IncidentResponseTriageObservation
    from .scenario_loader import load_random_scenario, load_scenario
    from .scenario_models import Scenario
except ImportError:
    from models import IncidentResponseTriageAction, IncidentResponseTriageObservation
    from server.scenario_loader import load_random_scenario, load_scenario
    from server.scenario_models import Scenario


class IncidentResponseTriageEnvironment(Environment):
    """Environment that serves incident triage scenarios and deterministic grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.step_count: int = 0
        self.current_scenario: Scenario | None = None
        self._last_reward: float = 0.0

    def reset(self) -> IncidentResponseTriageObservation:
        """Load a random easy scenario and return the initial observation."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.step_count = 0
        self._last_reward = 0.0

        # Keep reset selection random while delegating to the centralized loader.
        self.current_scenario = load_random_scenario("easy")

        return IncidentResponseTriageObservation(
            logs=self.current_scenario.logs,
            metrics=self.current_scenario.metrics,
            alerts=self.current_scenario.alerts,
            step_count=self.step_count,
            done=False,
            reward=0.0,
            metadata={
                "scenario_id": self.current_scenario.scenario_id,
                "difficulty": self.current_scenario.difficulty,
            },
        )

    def step(self, action: IncidentResponseTriageAction) -> tuple[IncidentResponseTriageObservation, int, bool, dict]:  # type: ignore[override]
        """Apply one triage action and return Gym-style (observation, reward, done, info)."""
        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset() before step().")

        # Count this action as one environment step.
        self.step_count += 1
        self._state.step_count = self.step_count

        truth = self.current_scenario.ground_truth

        # Support both requested field names and existing model names.
        predicted_service = str(
            getattr(action, "predicted_service", getattr(action, "root_cause_service", ""))
        )
        predicted_severity = str(getattr(action, "predicted_severity", ""))
        predicted_fix = str(
            getattr(action, "predicted_fix", getattr(action, "fix_command", ""))
        )

        # Deterministic exact-match checks (case-insensitive).
        service_correct = predicted_service.strip().lower() == truth.root_cause_service.strip().lower()
        severity_correct = predicted_severity.strip().lower() == truth.severity.strip().lower()
        fix_correct = predicted_fix.strip().lower() == truth.correct_fix.strip().lower()

        # Integer reward model:
        # +1 service, +1 severity, +2 fix, and -1 for each wrong prediction.
        reward = 0
        reward += 1 if service_correct else -1
        reward += 1 if severity_correct else -1
        reward += 2 if fix_correct else -1

        breakdown = {
            "service_correct": service_correct,
            "severity_correct": severity_correct,
            "fix_correct": fix_correct,
            "total_reward": reward,
        }

        # Episode ends on a fully correct triage or when max steps are reached.
        done = reward == 4 or self.step_count >= self.current_scenario.max_steps
        self._last_reward = reward

        info = {
            "breakdown": breakdown,
            "correct_answers": {
                "root_cause_service": truth.root_cause_service,
                "severity": truth.severity,
                "correct_fix": truth.correct_fix,
            },
        }

        observation = IncidentResponseTriageObservation(
            logs=self.current_scenario.logs,
            metrics=self.current_scenario.metrics,
            alerts=self.current_scenario.alerts,
            step_count=self.step_count,
            done=done,
            reward=float(reward),
            metadata={
                "scenario_id": self.current_scenario.scenario_id,
                "max_steps": self.current_scenario.max_steps,
                "graded": True,
                "info": info,
            },
        )

        return observation, reward, done, info

    def _grade(self, action: IncidentResponseTriageAction) -> float:
        """Deterministically score action fields against scenario ground truth."""
        if self.current_scenario is None:
            return 0.0

        truth = self.current_scenario.ground_truth
        reward = 0.0

        if action.root_cause_service.strip().lower() == truth.root_cause_service.strip().lower():
            reward += 1.0

        if action.root_cause_type.strip().lower() == truth.root_cause_type.strip().lower():
            reward += 0.5

        return min(reward, 1.5)

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
