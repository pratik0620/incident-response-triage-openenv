import json
from random import choice
from pathlib import Path

from scenarios.schema import Scenario


# Scenarios are now organized by difficulty under scenarios/easy.
SCENARIOS_BASE_DIR = Path(__file__).resolve().parents[1] / "scenarios"


def load_scenario(scenario_id: str) -> Scenario:
    """Load a scenario by ID from scenarios/{difficulty}/ directories."""
    for difficulty in ["easy", "medium", "hard"]:
        scenario_path = SCENARIOS_BASE_DIR / difficulty / f"{scenario_id}.json"

        if scenario_path.exists():
            with scenario_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return Scenario.model_validate(payload)

    raise FileNotFoundError(f"Scenario not found: {scenario_id}")


def get_all_scenarios() -> list[Scenario]:
    """Load and return all Scenario objects from scenarios/{difficulty}/ directories."""
    scenarios = []
    # Find every JSON scenario file in a deterministic order.
    for difficulty in ["easy", "medium", "hard"]:
        folder = SCENARIOS_BASE_DIR / difficulty

        if not folder.exists():
            continue

        for file in sorted(folder.glob("*.json")):
            with file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            scenarios.append(Scenario.model_validate(payload))

    return scenarios


def filter_scenarios_by_difficulty(
    scenarios: list[Scenario], difficulty: str
) -> list[Scenario]:
    """Return only scenarios that match the requested difficulty level."""
    return [scenario for scenario in scenarios if scenario.difficulty == difficulty]


def load_random_scenario(difficulty: str = "easy") -> Scenario:
    """Load one random scenario for the requested difficulty."""
    # Load the full scenario catalog first.
    scenarios = get_all_scenarios()

    # Keep only scenarios that match the requested difficulty.
    filtered = filter_scenarios_by_difficulty(scenarios, difficulty)
    if not filtered:
        raise ValueError(
            "No scenarios found for difficulty "
            f"'{difficulty}'. Available difficulties: easy, medium, hard."
        )

    # Sample one scenario for RL episode initialization.
    return choice(filtered)
