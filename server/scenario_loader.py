import json
from random import choice
from pathlib import Path

from .scenario_models import Scenario


# Scenarios are now organized by difficulty under scenarios/easy.
SCENARIOS_EASY_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "easy"


def load_scenario(scenario_id: str) -> Scenario:
    """Load a scenario by ID from the easy scenarios directory."""
    scenario_path = SCENARIOS_EASY_DIR / f"{scenario_id}.json"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return Scenario.model_validate(payload)


def get_all_scenarios() -> list[Scenario]:
    """Load and return all Scenario objects from scenarios/easy."""

    # Find every JSON scenario file in a deterministic order.
    scenario_files = sorted(SCENARIOS_EASY_DIR.glob("*.json"))
    if not scenario_files:
        raise FileNotFoundError(
            f"No scenario JSON files found in: {SCENARIOS_EASY_DIR}"
        )

    scenarios: list[Scenario] = []
    for scenario_file in scenario_files:
        # Read each JSON file and validate it against the Scenario schema.
        with scenario_file.open("r", encoding="utf-8") as f:
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
