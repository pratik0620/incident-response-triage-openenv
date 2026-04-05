"""
Synonym and keyword maps used across all graders.

ALL data here is hardcoded — no external calls, no randomness.
This is the single file to update when adding new incident types.

Design rule: every map key is lowercase. Matchers always .lower() inputs
before lookup. This guarantees case-insensitive determinism.
"""

from __future__ import annotations
from typing import Dict, List, Set

# ─────────────────────────────────────────────────────────────────────────────
# CAUSE TYPE SYNONYMS
# Maps a canonical root_cause_type (snake_case) to a list of equivalent
# phrases an agent might write instead.
# ─────────────────────────────────────────────────────────────────────────────
CAUSE_SYNONYMS: Dict[str, List[str]] = {
    "memory_leak": [
        "memory leak", "heap overflow", "oom", "out of memory",
        "memory exhaustion", "gc pressure", "heap exhaustion",
        "unbounded memory", "memory growth",
    ],
    "connection_timeout": [
        "connection timeout", "request timeout", "latency spike",
        "connection refused", "timeout error", "deadline exceeded",
        "connection pool exhausted", "db pool exhausted",
        "pool exhausted", "connection limit",
    ],
    "connection_pool_exhausted": [
        "connection pool exhausted", "pool exhausted", "db pool",
        "database connections exceeded", "connection limit reached",
        "connection timeout", "connection refused", "max connections",
    ],
    "disk_full": [
        "disk full", "disk space", "storage full", "no space left",
        "filesystem full", "disk exhaustion", "storage exhausted",
        "inode exhaustion",
    ],
    "config_error": [
        "config error", "configuration error", "misconfiguration",
        "wrong config", "bad config", "config mismatch",
        "environment variable", "missing config", "invalid config",
    ],
    "cpu_throttling": [
        "cpu throttling", "cpu limit", "cpu spike", "cpu pressure",
        "throttled", "cpu saturation", "cpu bound",
    ],
    "crash_loop": [
        "crash loop", "crashloop", "restart loop", "oom kill",
        "container crash", "pod crash", "crash loopbackoff",
        "repeated restart",
    ],
    "network_partition": [
        "network partition", "network split", "split brain",
        "network failure", "network issue", "connectivity issue",
        "packet loss", "network latency",
    ],
    "dependency_failure": [
        "dependency failure", "downstream failure", "upstream failure",
        "service dependency", "third party failure", "external service",
    ],
    "race_condition": [
        "race condition", "concurrency issue", "thread safety",
        "data race", "concurrent access", "locking issue",
    ],
    "deadlock": [
        "deadlock", "dead lock", "lock contention", "thread deadlock",
        "database deadlock",
    ],
    "certificate_expired": [
        "certificate expired", "cert expired", "tls error", "ssl error",
        "certificate invalid", "certificate error",
    ],
    "rate_limiting": [
        "rate limit", "rate limiting", "throttling", "429",
        "too many requests", "quota exceeded",
    ],
    "flaky_test": [
        "flaky", "flaky test", "intermittent", "non-deterministic",
        "timing issue", "race in test",
    ],
    "cache_warmup_failed": [
        "cache miss", "cache cold", "warm-up failed", "cache warmup",
        "cold cache", "cache not populated",
    ],
    "stale_jwks_cache": [
        "stale key", "jwks", "jwt key", "key rotation", "stale cache",
        "token key", "signing key",
    ],
    "worker_rate_limit_misconfiguration": [
        "rate limit", "worker limit", "throughput limit", "misconfiguration",
        "wrong config", "rate config",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX KEYWORD TIERS
# Maps a canonical root_cause_type to tiers of fix keywords.
# tier_1: superficial fixes (~0.3 base)
# tier_2: correct functional fixes (~0.7 base)
# tier_3: correct + preventive fixes (1.0 base)
# ─────────────────────────────────────────────────────────────────────────────
FIX_TIERS: Dict[str, Dict[str, List[str]]] = {
    "memory_leak": {
        "tier_1": ["restart", "redeploy", "rollback"],
        # tier_2: functional fix keywords only — no alert/monitor/profil
        # 4 keywords, need max(1, 4//2)=2 hits
        "tier_2": ["heap", "jvm", "memory limit", "increase"],
        # tier_3: tier_2 keywords PLUS preventive keywords (alert, monitor, profil)
        # 4 keywords, need max(2, 4//2)=2 hits — but crucially requires alert OR monitor
        # We split into two separate checks in get_fix_tier_score for memory_leak
        # Here we encode as: tier_3 = preventive-only keywords (alert, monitor, profil)
        # get_fix_tier_score: if any tier_3 preventive keyword + any tier_2 keyword → 1.0
        "tier_3": ["alert", "monitor", "profil", "bounded"],
    },
    "connection_pool_exhausted": {
        "tier_1": ["restart", "redeploy"],
        "tier_2": ["pool size", "connection limit", "increase", "max connections", "timeout"],
        "tier_3": ["pool size", "increase", "timeout", "monitor", "alert", "connection limit",
                   "retry", "circuit breaker"],
    },
    "connection_timeout": {
        "tier_1": ["restart", "redeploy"],
        "tier_2": ["timeout", "increase", "retry", "connection", "limit"],
        "tier_3": ["circuit breaker", "retry", "timeout", "monitor", "alert", "backoff"],
    },
    "disk_full": {
        "tier_1": ["delete", "remove", "clean"],
        "tier_2": ["disk", "storage", "expand", "cleanup", "rotate", "prune"],
        "tier_3": ["alert", "monitor", "expand", "rotate logs", "disk quota",
                   "storage limit", "cleanup policy"],
    },
    "config_error": {
        "tier_1": ["restart", "redeploy"],
        "tier_2": ["config", "environment", "variable", "correct", "fix"],
        "tier_3": ["config validation", "schema", "environment", "test", "staging",
                   "canary", "validate config"],
    },
    "cpu_throttling": {
        "tier_1": ["restart", "redeploy"],
        "tier_2": ["cpu limit", "resource limit", "increase", "scale"],
        "tier_3": ["cpu limit", "increase", "hpa", "autoscal", "monitor", "alert", "profile"],
    },
    "crash_loop": {
        "tier_1": ["restart"],
        "tier_2": ["memory", "limit", "oom", "increase", "rollback"],
        "tier_3": ["memory limit", "increase", "liveness probe", "readiness", "monitor",
                   "alert", "rollback"],
    },
    "rate_limiting": {
        "tier_1": ["retry", "wait"],
        "tier_2": ["backoff", "rate limit", "quota", "reduce"],
        "tier_3": ["exponential backoff", "circuit breaker", "quota increase",
                   "rate limit", "monitor", "alert"],
    },
    "deadlock": {
        "tier_1": ["restart", "redeploy"],
        "tier_2": ["lock", "transaction", "timeout", "retry"],
        "tier_3": ["lock order", "timeout", "retry", "monitor", "deadlock detection",
                   "transaction", "alert"],
    },
    "cache_warmup_failed": {
        "tier_1": ["restart", "redeploy", "rollback"],
        "tier_2": ["warmup", "warm-up", "preload", "populate cache", "cache warm",
                   "seed cache"],
        "tier_3": ["warmup job", "startup check", "monitor", "alert",
                   "cache warmup"],
    },
    "stale_jwks_cache": {
        "tier_1": ["restart", "redeploy", "rollback"],
        "tier_2": ["jwks", "refresh", "invalidate cache", "cache ttl",
                   "key rotation", "fetch keys"],
        "tier_3": ["auto refresh", "background refresh", "monitor", "alert",
                   "shorten ttl"],
    },
    "worker_rate_limit_misconfiguration": {
        "tier_1": ["restart", "redeploy", "rollback"],
        "tier_2": ["rate limit", "worker limit", "throughput", "config",
                   "increase", "adjust"],
        "tier_3": ["config validation", "monitor", "alert", "canary",
                   "rate limit policy"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REASONING CHAIN PATTERNS
# Causal chain phrases that indicate structured reasoning.
# An agent that uses these patterns demonstrates log-to-cause linking.
# ─────────────────────────────────────────────────────────────────────────────
REASONING_CHAIN_PATTERNS: List[str] = [
    # Causal connectors
    "because", "therefore", "as a result", "this caused", "which led to",
    "leading to", "resulting in", "due to", "caused by", "triggered by",
    "indicates that", "shows that", "confirms that", "suggests that",
    # Evidence linking
    "the logs show", "log indicates", "based on the logs", "from the logs",
    "the metric shows", "error rate", "spike in", "increase in",
    "according to the alert", "the alert shows",
    # Chain vocabulary
    "→", "->", "then", "subsequently", "this means",
    "connection pool exhausted → ", "timeout → ", "oom → ",
    # Structured words
    "root cause is", "root cause:", "caused by", "the issue is",
    "diagnosed as", "failure mode",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOG SIGNAL KEYWORDS
# Keywords that, if present in logs AND in agent answer, score faithfulness.
# Keywords NOT in logs but in agent answer → hallucination penalty.
# ─────────────────────────────────────────────────────────────────────────────
LOG_SIGNAL_KEYWORDS: List[str] = [
    "connection pool", "timeout", "oom", "out of memory", "heap",
    "disk full", "no space", "config", "rate limit", "crash", "restart",
    "cpu", "throttl", "deadlock", "certificate", "network", "latency",
    "error rate", "5xx", "4xx", "503", "500", "429",
    "memory", "connection refused", "gc", "pod", "container",
    "exception", "stack overflow", "null pointer",
]

# ─────────────────────────────────────────────────────────────────────────────
# OVER-GUESSING INDICATORS
# Phrases that signal shotgun-style multi-hypothesis guessing.
# ─────────────────────────────────────────────────────────────────────────────
OVER_GUESS_PHRASES: List[str] = [
    "could be", "might be", "possibly", "perhaps", "maybe",
    "or it could", "another possibility", "alternatively",
    "one option is", "another option", "it's hard to say",
    "not sure", "unclear", "multiple issues", "several issues",
    "could also be",
]

# ─────────────────────────────────────────────────────────────────────────────
# SUPERFICIAL FIX INDICATORS
# Generic fixes that show no real understanding.
# ─────────────────────────────────────────────────────────────────────────────
SUPERFICIAL_FIX_PHRASES: List[str] = [
    "restart the service", "restart service", "redeploy", "rollback",
    "reboot", "kill the process", "restart the pod", "restart pod",
    "just restart",
]


def get_cause_synonyms(root_cause_type: str) -> Set[str]:
    """
    Return all synonym phrases for a given root_cause_type.
    Always includes the original type words as well.
    """
    canonical = root_cause_type.lower()
    synonyms: Set[str] = set()

    # Add canonical keywords
    synonyms.update(canonical.replace("_", " ").split())
    synonyms.add(canonical.replace("_", " "))

    # Add mapped synonyms
    for phrase in CAUSE_SYNONYMS.get(canonical, []):
        synonyms.add(phrase.lower())
        synonyms.update(phrase.lower().split())

    return synonyms


def get_fix_tier_score(answer_lower: str, root_cause_type: str, correct_fix: str) -> float:
    """
    Compute fix quality score using tiered keyword matching.

    Scoring tiers:
        0.0  — no meaningful fix keywords found
        0.3  — only superficial keywords (restart/redeploy/rollback)
        0.7  — tier_2 functional fix keywords present (correct fix, no prevention)
        1.0  — tier_2 functional PLUS at least one tier_3 preventive keyword

    tier_3 keywords are *additive* — they represent alert/monitor/profiling
    additions on top of the functional fix. An answer scores 1.0 only if it
    has both the fix AND the preventive component.

    Falls back to proportional keyword overlap on correct_fix if the
    root_cause_type is not in FIX_TIERS.
    """
    canonical = root_cause_type.lower()
    superficial_hit = any(p in answer_lower for p in SUPERFICIAL_FIX_PHRASES)

    if canonical in FIX_TIERS:
        tiers = FIX_TIERS[canonical]

        t2_hits   = sum(1 for kw in tiers["tier_2"] if kw in answer_lower)
        t3_hits   = sum(1 for kw in tiers["tier_3"] if kw in answer_lower)
        t2_needed = max(1, len(tiers["tier_2"]) // 2)

        # 1.0: functional fix PLUS at least one preventive keyword
        if t2_hits >= t2_needed and t3_hits >= 1:
            return 1.0

        # 0.7: functional fix only (correct but not preventive)
        if t2_hits >= t2_needed:
            return 0.7

        # 0.3: superficial only (restart/redeploy)
        if superficial_hit:
            return 0.3

        return 0.0

    else:
        # Fallback: proportional keyword overlap on correct_fix string
        fix_keywords = [w for w in correct_fix.lower().split() if len(w) > 4]
        if not fix_keywords:
            return 0.0
        matched = sum(1 for kw in fix_keywords if kw in answer_lower)
        ratio = matched / len(fix_keywords)
        if ratio >= 0.7:
            return 1.0
        elif ratio >= 0.4:
            return 0.7
        elif superficial_hit:
            return 0.3
        return round(ratio, 4)