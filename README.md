---
title: Incident Response Triage Env
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - incident-response
  - devops
  - sre
  - reinforcement-learning
---

# Incident Response Triage Environment
An OpenEnv reinforcement learning environment where AI agents diagnose real-world production incidents. The agent reads logs, metrics, and alerts from a simulated microservice system and must identify the root cause, propose a fix, and do so efficiently within a step budget.

## Environment overview

Real SRE teams respond to production incidents by reading logs and metrics, forming a hypothesis, and proposing a fix. This environment simulates that workflow. An agent takes investigative actions (read_logs, check_metrics) and terminal actions (identify_cause, propose_fix, escalate). Every episode is a unique incident scenario drawn from a pool of 15 realistic scenarios across 3 difficulty levels.

## Action space

| Action | Type | Description |
|---|---|---|
| `read_logs` | Investigative | Fetch log entries for a service. Costs -0.02 reward per step. |
| `check_metrics` | Investigative | Fetch CPU, memory, error rate, latency for a service. Costs -0.02 reward per step. |
| `identify_cause` | Terminal | Declare the root cause service and failure type. Ends the episode. |
| `propose_fix` | Terminal | Declare root cause AND propose a concrete fix. Ends the episode. |
| `escalate` | Terminal | Hand off when unable to diagnose. Returns flat 0.2 reward. |

## Observation space

Each step returns:
- `logs` — timestamped log entries with service, level, and message
- `metrics` — per-service CPU%, memory%, error rate, p99 latency
- `alerts` — fired alerts with severity (P1–P4) and message
- `step` / `max_steps` — current step and budget
- `previous_actions` — history of actions taken
- `task_description` — plain-English description of the incident
- `reward` — reward for the current step
- `done` / `final_score` — episode termination and final 0.0–1.0 score

## Tasks

| Difficulty | Scenarios | max_steps | Description |
|---|---|---|---|
| Easy | 5 | 10 | Single service, obvious root cause, no red herrings |
| Medium | 5 | 15 | Cascading failures across 2–3 services, 1 red herring |
| Hard | 5 | 20 | Noisy multi-service logs, 2+ red herrings, must classify failure type |

## Reward function

Scores are computed by a 6-signal grader:

| Signal | Weight (Easy / Medium / Hard) | Description |
|---|---|---|
| A — Root cause | 0.35 / 0.25 / 0.15 | Correct service and failure type identified |
| B — Fix quality | 0.30 / 0.20 / 0.15 | Tiered fix scoring: restart (0.3), functional fix (0.7), fix + prevention (1.0) |
| C — Reasoning | 0.10 / 0.25 / 0.30 | Causal chain explanation quality |
| D — Faithfulness | 0.10 / 0.15 / 0.20 | Reasoning grounded in actual log content |
| E — Noise handling | 0.10 / 0.10 / 0.15 | Red herring services correctly ignored |
| F — Efficiency | 0.05 / 0.05 / 0.05 | Step budget usage (non-linear: faster = higher score) |

Investigative steps cost -0.02 reward each. Escalating returns a flat 0.2.

## Running the environment

The environment is deployed as a Hugging Face Space.

Base URL: https://pratik234567-incident-response-triage-env.hf.space

Example reset request: curl -X POST https://pratik234567-incident-response-triage-env.hf.space/reset