"""
Inference Script — incident_response_triage_env
================================================
STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

ENV VARS:
    API_BASE_URL       LLM endpoint  (default: HuggingFace router)
    MODEL_NAME         Model id       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN / API_KEY Auth key
    IMAGE_NAME         Docker image name for the environment
    ENV_BASE_URL       Fallback if no docker image (default: http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Tuple

from openai import OpenAI


def _ensure_package_loaded() -> None:
    try:
        import incident_response_triage_env  # noqa: F401
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

from incident_response_triage_env.client import IncidentResponseTriageEnv  # noqa: E402
from incident_response_triage_env.models import IncidentResponseTriageAction  # noqa: E402


#==========Configuration (all overridable via environment variables)==========#
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASK_NAME = os.getenv("INCIDENT_TRIAGE_TASK", "incident-triage")
BENCHMARK = os.getenv("INCIDENT_TRIAGE_BENCHMARK", "incident_response_triage_env")
DIFFICULTY = os.getenv("INCIDENT_TRIAGE_DIFFICULTY", "easy")

TEMPERATURE = float(os.getenv("INCIDENT_TRIAGE_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("INCIDENT_TRIAGE_MAX_TOKENS", "1024"))
MAX_EPISODE_STEPS = int(os.getenv("INCIDENT_TRIAGE_MAX_EPISODE_STEPS", "64"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))

# Platform validator requires scores strictly between 0 and 1
# (not 0.0 and not 1.0). Use a safe open-interval clamp.
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def clamp_score(score: float) -> float:
    """Clamp a score to the open interval (0, 1) as required by the platform."""
    val = float(score)
    if val <= _SCORE_MIN:
        return _SCORE_MIN
    if val >= _SCORE_MAX:
        return _SCORE_MAX
    return round(val, 4)

async def run_episode(difficulty: str) -> float:
    env: Optional[IncidentResponseTriageEnv] = None
    final_score: float = _SCORE_MIN

    try:
        env = IncidentResponseTriageEnv(base_url=ENV_BASE_URL)
        await env.connect()

        result = await env.reset(difficulty=difficulty)
        obs = result.observation

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        for _ in range(MAX_EPISODE_STEPS):
            if obs.done:
                break

            user_prompt = build_observation_prompt(obs)
            action, _ = get_model_action(client, user_prompt)

            result = await env.step(action)
            obs = result.observation

            if result.done:
                fs = getattr(obs, "final_score", None)
                if fs is not None:
                    final_score = clamp_score(fs)
                break

    except Exception as e:
        print(f"[ERROR] Episode failed: {e}")

    finally:
        if env:
            try:
                await env.close()
            except:
                pass

    return clamp_score(final_score)


def run_sync_episode(difficulty: str) -> float:
    return asyncio.run(run_episode(difficulty))


# ── Grading Logic ────────────────────────────────────────────────────────────

def _grading_logic(difficulty: str) -> float:
    """Run a full episode for the given difficulty and return a clamped score."""
    score = run_sync_episode(difficulty)
    return clamp_score(score)


VALID_ACTIONS = frozenset(
    {"read_logs", "check_metrics", "identify_cause", "propose_fix", "escalate"}
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an on-call SRE triaging a microservice incident. You choose one action per turn.

    ALLOWED ACTIONS:
    - read_logs      — Fetch log entries to gather evidence. (episode continues)
    - check_metrics  — Fetch CPU/Mem/Error metrics. (episode continues)
    
    TERMINAL ACTIONS (Choose wisely):
    - propose_fix    — (RECOMMENDED for High Score) Provide the root cause AND a concrete fix.
                       Use this when you are confident in your diagnosis.
    - identify_cause — (DIAGNOSIS ONLY) State the root cause. 
                       Use this for Easy tasks or when you know the cause but lack a clear fix.
    - escalate       — (SAFETY HAND-OFF) Use this if you have exhausted most of your budget 
                       without a clear hypothesis, or if you find yourself repeating 
                       actions without gaining new insights.

    Respond with a single JSON object only (no markdown), keys:
      "action_type"  (string, required)
      "reasoning"    (string, required)
      "answer"       (string or null) — required for identify_cause / propose_fix
      "service"      (string or null) — optional focus service

    --- CRITICAL GUIDELINES ---
    1. PROPOSE RESOLUTION: On Medium and Hard tasks, you MUST aim for 'propose_fix'. Simply identifying the cause will yield partial credit.
    2. CAUSAL CHAIN: Use phrases like "the logs show...", "due to...", "resulting in..." in your reasoning.
    3. EVIDENCE GROUNDING: Do not hallucinate. Only state what is supported by Logs, Metrics, or Alerts.
    4. ESCALATE IF STUCK: If you have exhausted 80% of your step budget without a clear hypothesis, use 'escalate'.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ").replace("\r", " ") if error else "null"
    print(
        f"[STEP]  step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(difficulty: str, final_score: float, steps_taken: int) -> None:
    final_score = max(_SCORE_MIN, min(_SCORE_MAX, float(final_score)))
    print(f"[END] task={difficulty} score={final_score:.4f} steps={steps_taken}", flush=True)


def _log_entry_line(entry: Any) -> str:
    if isinstance(entry, dict):
        ts, svc, lvl, msg = entry.get("timestamp",""), entry.get("service",""), entry.get("level",""), entry.get("message","")
    else:
        ts, svc, lvl, msg = getattr(entry,"timestamp",""), getattr(entry,"service",""), getattr(entry,"level",""), getattr(entry,"message","")
    return f"  [{ts}] {svc} {lvl}: {msg}"


def _metric_line(m: Any) -> str:
    if isinstance(m, dict):
        return f"  {m.get('service')}: cpu={m.get('cpu_pct')}% mem={m.get('mem_pct')}% err_rate={m.get('error_rate')} p99_ms={m.get('latency_p99_ms')}"
    return f"  {m.service}: cpu={m.cpu_pct}% mem={m.mem_pct}% err_rate={m.error_rate} p99_ms={m.latency_p99_ms}"


def _alert_line(a: Any) -> str:
    if isinstance(a, dict):
        return f"  {a.get('alert_id')} {a.get('severity')} {a.get('service')}: {a.get('message')}"
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

        Current Goal: Gather enough signal to use 'propose_fix'. If you are stuck or budget is nearly exhausted, use 'escalate'.
        Respond ONLY with a valid JSON object.
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
            reasoning=f"Model output not valid JSON; defaulting to read_logs. ({exc})",
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


def action_to_log_str(action: IncidentResponseTriageAction) -> str:
    """Single-line action summary for [STEP] — no embedded newlines."""
    parts = [f"{action.action_type}", f"reasoning={action.reasoning[:200]}"]
    if action.service:
        parts.append(f"service={action.service}")
    if action.answer:
        parts.append(f"answer={action.answer[:300]}")
    return "|".join(parts).replace("\n", " ").replace("\r", " ")


def get_model_action(client: OpenAI, user_prompt: str) -> Tuple[IncidentResponseTriageAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        raw = json.dumps({
            "action_type": "read_logs",
            "reasoning": f"LLM error: {exc}",
            "answer": None,
        })

    action = parse_model_to_action(raw)
    return action, raw.replace("\n", " ")[:800]


async def run_single_episode(client: OpenAI, difficulty: str) -> None:
    """Run one episode for the given difficulty and emit START/STEP/END logs."""

    env: Optional[IncidentResponseTriageEnv] = None
    step_rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score: float = _SCORE_MIN

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    try:
        if IMAGE_NAME:
            env = await IncidentResponseTriageEnv.from_docker_image(IMAGE_NAME)
        else:
            env = IncidentResponseTriageEnv(base_url=ENV_BASE_URL)
            await env.connect()

        result = await env.reset(difficulty=difficulty)
        obs = result.observation

        for _ in range(MAX_EPISODE_STEPS):
            if obs.done:
                break

            user_prompt = build_observation_prompt(obs)
            action, _raw = get_model_action(client, user_prompt)
            action_log_str = action_to_log_str(action)

            try:
                result = await env.step(action)
            except Exception as exc:
                steps_taken = obs.step + 1
                log_step(step=steps_taken, action=action_log_str,
                         reward=_SCORE_MIN, done=False, error=str(exc))
                break

            raw_reward = float(result.reward if result.reward is not None else 0.0)
            clamped_reward = clamp_score(raw_reward)

            done = bool(result.done)
            obs = result.observation
            step_rewards.append(clamped_reward)
            steps_taken = obs.step

            log_step(step=obs.step, action=action_log_str, reward=clamped_reward, done=done, error=None)

            if done:
                raw_fs = getattr(obs, "final_score", None)
                if raw_fs is not None:
                    final_score = clamp_score(float(raw_fs))
                    if final_score >= SUCCESS_SCORE_THRESHOLD:
                        success = True
                break

    except Exception as exc:
        print(f"[DEBUG] Episode error ({difficulty}): {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

    log_end(difficulty=difficulty, final_score=final_score, steps_taken=steps_taken)


async def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN (or OPENAI_API_KEY) environment variable is required.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "")

    difficulties = ["easy", "medium", "hard"]

    for diff in difficulties:
        await run_single_episode(client, diff)


if __name__ == "__main__":
    asyncio.run(main())