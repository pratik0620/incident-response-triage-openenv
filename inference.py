from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

def _ensure_package_loaded() -> None:
    try:
        import incident_response_triage_env
    except ModuleNotFoundError:
        import importlib.util
        import sys
        import types

        root = Path(__file__).resolve().parent
        pkg_name = "incident_response_triage_env"
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root)]
        sys.modules[pkg_name] = pkg

        for mod, filename in (("models", "models.py"), ("client", "client.py")):
            fq = f"{pkg_name}.{mod}"
            spec = importlib.util.spec_from_file_location(fq, root / filename)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {fq} from {root / filename}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[fq] = module
            spec.loader.exec_module(module)
            setattr(pkg, mod, module)


_ensure_package_loaded()
from incident_response_triage_env.client import IncidentResponseTriageEnv
from incident_response_triage_env.models import IncidentResponseTriageAction


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASK_NAME = os.getenv("INCIDENT_TRIAGE_TASK", "incident-triage")
BENCHMARK = os.getenv("INCIDENT_TRIAGE_BENCHMARK", "incident_response_triage_env")
DIFFICULTY = os.getenv("INCIDENT_TRIAGE_DIFFICULTY", "easy")

TEMPERATURE = float(os.getenv("INCIDENT_TRIAGE_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("INCIDENT_TRIAGE_MAX_TOKENS", "1024"))
MAX_EPISODE_STEPS = int(os.getenv("INCIDENT_TRIAGE_MAX_EPISODE_STEPS", "64"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))

VALID_ACTIONS = frozenset(
    {"read_logs", "check_metrics", "identify_cause", "propose_fix", "escalate"}
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an on-call SRE triaging a microservice incident. You choose one action per turn.

    Allowed action_type values:
    - read_logs — gather signal (small step cost; episode continues)
    - check_metrics — gather signal (small step cost; episode continues)
    - identify_cause — terminal: you must state the root cause (service + failure) in "answer"
    - propose_fix — terminal: "answer" must include both root cause and a concrete fix/command
    - escalate — terminal: hand off when you cannot diagnose; lower reward

    Respond with a single JSON object only (no markdown), keys:
      "action_type" (string, required)
      "reasoning" (string, required)
      "answer" (string or null) — required for identify_cause / propose_fix
      "service" (string or null) — optional focus service

    For hard difficulty, root-cause answers must also mention the failure category when using
    identify_cause (real, flaky, intermittent, env_specific).
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = "null" if error is None else error.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _log_entry_line(entry: Any) -> str:
    if isinstance(entry, dict):
        ts = entry.get("timestamp", "")
        svc = entry.get("service", "")
        lvl = entry.get("level", "")
        msg = entry.get("message", "")
    else:
        ts = getattr(entry, "timestamp", "")
        svc = getattr(entry, "service", "")
        lvl = getattr(entry, "level", "")
        msg = getattr(entry, "message", "")
    return f"  [{ts}] {svc} {lvl}: {msg}"


def _metric_line(m: Any) -> str:
    if isinstance(m, dict):
        return (
            f"  {m.get('service')}: cpu={m.get('cpu_pct')}% mem={m.get('mem_pct')}% "
            f"err_rate={m.get('error_rate')} p99_ms={m.get('latency_p99_ms')}"
        )
    return (
        f"  {m.service}: cpu={m.cpu_pct}% mem={m.mem_pct}% "
        f"err_rate={m.error_rate} p99_ms={m.latency_p99_ms}"
    )


def _alert_line(a: Any) -> str:
    if isinstance(a, dict):
        return (
            f"  {a.get('alert_id')} {a.get('severity')} {a.get('service')}: "
            f"{a.get('message')}"
        )
    return f"  {a.alert_id} {a.severity} {a.service}: {a.message}"


def build_observation_prompt(observation: Any) -> str:
    logs = "\n".join(_log_entry_line(x) for x in observation.logs[:40])
    metrics = "\n".join(_metric_line(x) for x in observation.metrics[:30])
    alerts = "\n".join(_alert_line(x) for x in observation.alerts[:20])
    prev = observation.previous_actions[-12:] if observation.previous_actions else []
    prev_s = ", ".join(prev) if prev else "(none)"

    return textwrap.dedent(
        f"""
        Task description:
        {observation.task_description}

        Environment step: {observation.step} / max_steps: {observation.max_steps}
        Episode done: {observation.done}

        Recent actions (types only): {prev_s}

        Logs:
        {logs}

        Metrics:
        {metrics}

        Alerts:
        {alerts}

        Respond ONLY with a valid JSON object. Do not include explanations, markdown, or extra text.
        """
    ).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    return json.loads(text)


def parse_model_to_action(raw: str) -> IncidentResponseTriageAction:
    try:
        data = _extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return IncidentResponseTriageAction(
            action_type="read_logs",
            reasoning=f"Model output not valid JSON; probing logs. ({exc})",
            answer=None,
            service=None,
        )

    action_type = str(data.get("action_type", "")).strip().lower()
    if action_type not in VALID_ACTIONS:
        return IncidentResponseTriageAction(
            action_type="read_logs",
            reasoning=f"Invalid action_type {action_type!r}; defaulting to read_logs.",
            answer=None,
            service=None,
        )

    reasoning = str(data.get("reasoning", "")).strip() or "no reasoning provided"
    answer = data.get("answer")
    if answer is not None:
        answer = str(answer).strip() or None
    service = data.get("service")
    if service is not None:
        service = str(service).strip() or None

    return IncidentResponseTriageAction(
        action_type=action_type,  # type: ignore[arg-type]
        reasoning=reasoning,
        answer=answer,
        service=service,
    )


def action_to_log_line(action: IncidentResponseTriageAction) -> str:
    """Single-line action summary for [STEP] (no embedded newlines)."""
    parts = [
        f"{action.action_type}",
        f"reasoning={action.reasoning[:400]}",
    ]
    if action.service:
        parts.append(f"service={action.service}")
    if action.answer:
        parts.append(f"answer={action.answer[:500]}")
    return "|".join(parts).replace("\n", " ").replace("\r", " ")


def get_model_action(client: OpenAI, user_prompt: str) -> Tuple[IncidentResponseTriageAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        raw = json.dumps(
            {
                "action_type": "read_logs",
                "reasoning": f"LLM error: {exc}",
                "answer": None,
            }
        )

    action = parse_model_to_action(raw)
    return action, raw.replace("\n", " ")[:800]


async def main() -> None:
    if not API_KEY:
        print(
            "[DEBUG] HF_TOKEN or API_KEY is not set; LLM calls will fail.",
            flush=True,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "")

    env: Optional[IncidentResponseTriageEnv] = None
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if LOCAL_IMAGE_NAME:
            env = await IncidentResponseTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = IncidentResponseTriageEnv(base_url=ENV_BASE_URL)
            await env.connect()

        result = await env.reset(difficulty=DIFFICULTY)
        obs = result.observation

        for _ in range(MAX_EPISODE_STEPS):
            if obs.done:
                break

            user_prompt = build_observation_prompt(obs)
            action, _raw_snippet = get_model_action(client, user_prompt)
            line = action_to_log_line(action)

            try:
                result = await env.step(action)
            except Exception as exc:
                rewards.append(0.0)
                steps_taken = obs.step + 1
                log_step(
                    step=steps_taken,
                    action=line,
                    reward=0.00,
                    done=False,
                    error=str(exc),
                )
                break

            reward = float(result.reward if result.reward is not None else 0.0)
            done = bool(result.done)
            obs = result.observation

            rewards.append(reward)
            steps_taken = obs.step
            log_step(
                step=obs.step,
                action=line,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                fs = getattr(obs, "final_score", None)
                if fs is not None and float(fs) >= SUCCESS_SCORE_THRESHOLD:
                    success = True
                break

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
