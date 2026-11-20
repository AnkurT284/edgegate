"""Debugging agent harness — full observability for the 9-agent pipeline.

Covers:
  - per-agent trace (inputs/outputs, timing, model label, lifecycle)
  - isolated single-agent runs (iterate on one agent without running all 9)
  - replay of saved cases / signals (data/signals.json, examples/*.json)
  - state diffing between steps + between runs (regression checks)
  - mock (offline) and live BYOK modes — keys never logged or persisted

All numeric scores remain grounded in engines.py; this module only observes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import agents, engines
from .llm import BYOKConfig
from .schemas import AccountRules, AgentStep, EventInput, Lifecycle, MarketReaction, OptionsSnapshot

AGENT_NAMES: list[str] = [fn.__name__.replace("_agent", "") for fn in agents.PIPELINE]

_AGENT_FNS: dict[str, Any] = {
    fn.__name__.replace("_agent", ""): fn for fn in agents.PIPELINE
}

# Which state keys each agent is expected to read/write — for trace readability.
AGENT_IO_HINTS: dict[str, dict[str, list[str]]] = {
    "verifier": {"reads": ["event"], "writes": ["verified", "verification"]},
    "classifier": {"reads": ["event", "verified"], "writes": ["classification"]},
    "fundamental": {"reads": ["event"], "writes": ["fundamental_score", "fundamental"]},
    "market": {"reads": ["market"], "writes": ["market_analysis"]},
    "options": {"reads": ["options"], "writes": ["options_analysis"]},
    "expectation": {"reads": ["market", "options"], "writes": ["expectation"]},
    "second_order": {"reads": ["event"], "writes": ["second_order"]},
    "signal": {"reads": ["fundamental_score", "market_analysis", "options_analysis", "expectation", "account"],
               "writes": ["signal"]},
    "risk": {"reads": ["market", "options", "signal"], "writes": ["risk"]},
}


def list_agents() -> list[dict[str, Any]]:
    """Agent registry for CLI/API discovery."""
    order = []
    for i, fn in enumerate(agents.PIPELINE):
        name = fn.__name__.replace("_agent", "")
        order.append({
            "order": i,
            "name": name,
            "fn": fn.__name__,
            "reads": AGENT_IO_HINTS.get(name, {}).get("reads", []),
            "writes": AGENT_IO_HINTS.get(name, {}).get("writes", []),
        })
    return order


def _dump(model) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if isinstance(model, dict):
        return copy.deepcopy(model)
    return dict(model)


def build_initial_state(event: EventInput, market: MarketReaction,
                        options: OptionsSnapshot, account: AccountRules) -> dict[str, Any]:
    return {
        "event": _dump(event), "market": _dump(market),
        "options": _dump(options), "account": _dump(account),
    }


def state_hash(state: dict[str, Any]) -> str:
    blob = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def diff_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    """Shallow diff: which top-level keys were added/removed/changed."""
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    changed = []
    for k in before:
        if k in after:
            try:
                if json.dumps(before[k], sort_keys=True, default=str) != \
                   json.dumps(after[k], sort_keys=True, default=str):
                    changed.append(k)
            except Exception:
                if before[k] != after[k]:
                    changed.append(k)
    return {"added": added, "removed": removed, "changed": changed}


@dataclass
class DebugStep:
    agent: str
    lifecycle: str = ""
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    model: str = "mock"
    duration_ms: float = 0.0
    state_delta: dict[str, list[str]] = field(default_factory=dict)
    state_keys: list[str] = field(default_factory=list)
    error: str = ""
    input_hash: str = ""
    output_hash: str = ""


@dataclass
class DebugTrace:
    trace_id: str
    started_at: str
    llm_label: str
    mock_mode: bool
    steps: list[DebugStep] = field(default_factory=list)
    final_state_keys: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    verdict: str = ""
    event_score: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "llm": {"label": self.llm_label, "mock_mode": self.mock_mode},
            "total_ms": round(self.total_ms, 2),
            "verdict": self.verdict,
            "event_score": self.event_score,
            "errors": self.errors,
            "final_state_keys": self.final_state_keys,
            "steps": [asdict(s) for s in self.steps],
        }


async def run_agent_isolated(agent_name: str, state: dict[str, Any],
                             llm: BYOKConfig) -> tuple[dict[str, Any], DebugStep]:
    """Run ONE agent against a copy of `state`. Returns (new_state, DebugStep)."""
    if agent_name not in _AGENT_FNS:
        raise ValueError(f"unknown agent '{agent_name}'. Known: {sorted(_AGENT_FNS)}")
    fn = _AGENT_FNS[agent_name]
    working = copy.deepcopy(state)
    before_hash = state_hash(working)
    t0 = time.perf_counter()
    error = ""
    try:
        working, raw = await fn(working, llm)
    except Exception as e:  # never let one agent kill the harness
        error = f"{type(e).__name__}: {e}"
        raw = {"agent": agent_name, "lifecycle": Lifecycle.NO_TRADE.value,
               "summary": f"EXCEPTION: {error}", "data": {}, "model": llm.label}
    dt_ms = (time.perf_counter() - t0) * 1000.0
    delta = diff_states(state, working)
    step = DebugStep(
        agent=raw.get("agent", agent_name),
        lifecycle=str(raw.get("lifecycle", "")),
        summary=raw.get("summary", ""),
        data=raw.get("data", {}),
        model=raw.get("model", llm.label),
        duration_ms=round(dt_ms, 2),
        state_delta=delta,
        state_keys=sorted(working.keys()),
        error=error,
        input_hash=before_hash,
        output_hash=state_hash(working),
    )
    return working, step


async def run_traced(event: EventInput, market: MarketReaction,
                     options: OptionsSnapshot, account: AccountRules,
                     llm: BYOKConfig,
                     stop_after: Optional[str] = None,
                     start_from: Optional[str] = None,
                     preloaded_state: Optional[dict[str, Any]] = None,
                     capture_snapshots: bool = False,
                     snapshots: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], DebugTrace]:
    """Run the pipeline with per-agent timing + diffs.

    Args:
        stop_after: if set, halt after this agent (inclusive). Enables step-through.
        start_from: if set, skip all agents before this one (requires preloaded_state
            or a fresh initial state — useful with `snapshots` replay).
        preloaded_state: full state dict to start from instead of build_initial_state.
        capture_snapshots: if True, fill `snapshots` (dict agent_name -> deep-copied
            state AFTER that agent) for later single-agent debugging.
        snapshots: caller-provided dict to populate; created if None and capturing.
    Returns (final_state, DebugTrace).
    """
    names = [n for n in AGENT_NAMES]
    if start_from and start_from not in _AGENT_FNS:
        raise ValueError(f"unknown start_from '{start_from}'")
    if stop_after and stop_after not in _AGENT_FNS:
        raise ValueError(f"unknown stop_after '{stop_after}'")

    state: dict[str, Any] = copy.deepcopy(preloaded_state) if preloaded_state else \
        build_initial_state(event, market, options, account)
    trace = DebugTrace(
        trace_id=f"tr_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
        started_at=datetime.now(timezone.utc).isoformat(),
        llm_label=llm.label, mock_mode=llm.mock_mode,
    )
    if capture_snapshots and snapshots is None:
        snapshots = {}

    started = start_from is None
    t_all = time.perf_counter()
    for name in names:
        if not started:
            if name == start_from:
                started = True
            else:
                continue
        state, step = await run_agent_isolated(name, state, llm)
        trace.steps.append(step)
        if step.error:
            trace.errors.append(f"{name}: {step.error}")
        if capture_snapshots and snapshots is not None:
            snapshots[name] = copy.deepcopy(state)
        if stop_after and name == stop_after:
            break

    trace.total_ms = round((time.perf_counter() - t_all) * 1000.0, 2)
    trace.final_state_keys = sorted(state.keys())
    sig = state.get("signal", {}) if isinstance(state.get("signal"), dict) else {}
    trace.verdict = str(sig.get("status", "UNKNOWN"))
    try:
        trace.event_score = float(sig.get("event_score", 0) or 0)
    except Exception:
        trace.event_score = 0.0
    return state, trace


# --------------------------------------------------------------------------
# Case loading / replay / regression
# --------------------------------------------------------------------------

def load_case(path: str | Path) -> dict[str, Any]:
    """Load a case file shaped like examples/pb_fintech_input.json.

    Returns dict with keys event/market/options/account/llm (+ optional
    `expect` block: {status, event_score_min, ...} used by regression).
    """
    raw = json.loads(Path(path).read_text())
    # Support replay files that wrapped the case: {"input": {...}, ...}
    if "input" in raw and isinstance(raw["input"], dict) and "event" in raw["input"]:
        raw = raw["input"]
    # Support raw data/signals.json rows (they are outputs, not inputs) — reject clearly.
    if "steps" in raw and "event" not in raw:
        raise ValueError(
            f"{path} looks like a signal/output row (has 'steps'), not an input case. "
            "Replay outputs via load_signal_input() or re-run the original input JSON.")
    for k in ("event", "market", "options"):
        if k not in raw:
            raise ValueError(f"case {path} missing required key '{k}'")
    return raw


def coerce_case(raw: dict[str, Any]) -> tuple[EventInput, MarketReaction, OptionsSnapshot, AccountRules]:
    return (EventInput(**raw["event"]), MarketReaction(**raw["market"]),
            OptionsSnapshot(**raw["options"]), AccountRules(**raw.get("account", {})))


def load_signal_input(signal: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Best-effort: signals.json rows don't store inputs, so return None with a
    clear contract. Kept as a function so replay code has one place to evolve
    when input persistence is added (Phase 4 backtest log)."""
    return signal.get("input") if isinstance(signal.get("input"), dict) else None


def save_trace(trace: DebugTrace, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
    return p


def compare_traces(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two trace dicts (from DebugTrace.to_dict). Reports verdict/score
    drift + per-agent summary/data drift. Used for regression (mock-mode determinism)."""
    out: dict[str, Any] = {
        "verdict_a": a.get("verdict"), "verdict_b": b.get("verdict"),
        "verdict_match": a.get("verdict") == b.get("verdict"),
        "score_a": a.get("event_score"), "score_b": b.get("event_score"),
        "score_delta": round(float(a.get("event_score") or 0) - float(b.get("event_score") or 0), 4),
        "step_count_match": len(a.get("steps", [])) == len(b.get("steps", [])),
        "agent_diffs": [],
    }
    sa = {s["agent"]: s for s in a.get("steps", [])}
    sb = {s["agent"]: s for s in b.get("steps", [])}
    for name in sa:
        if name not in sb:
            out["agent_diffs"].append({"agent": name, "issue": "missing in B"})
            continue
        da = json.dumps(sa[name].get("data", {}), sort_keys=True, default=str)
        db = json.dumps(sb[name].get("data", {}), sort_keys=True, default=str)
        if da != db or sa[name].get("summary") != sb[name].get("summary"):
            out["agent_diffs"].append({
                "agent": name,
                "summary_a": sa[name].get("summary"), "summary_b": sb[name].get("summary"),
                "data_match": da == db,
                "model_a": sa[name].get("model"), "model_b": sb[name].get("model"),
            })
    out["match"] = out["verdict_match"] and not out["agent_diffs"]
    return out


async def run_regression(cases: list[str | Path], llm: BYOKConfig) -> list[dict[str, Any]]:
    """Run each case file in mock-friendly mode, collect verdicts + gate failures.

    Each case may include an `expect` block, e.g.
      {"expect": {"status": "NO_TRADE", "event_score_min": 50}}
    Results include pass/fail vs expectations (missing expect => reported, not failed).
    """
    results = []
    for c in cases:
        c = str(c)
        try:
            raw = load_case(c)
            event, market, options, account = coerce_case(raw)
            state, trace = await run_traced(event, market, options, account, llm)
            sig = state.get("signal", {})
            row: dict[str, Any] = {
                "case": c, "ok": True,
                "verdict": trace.verdict,
                "event_score": trace.event_score,
                "failed_gates": sig.get("failed", []),
                "total_ms": trace.total_ms,
                "errors": trace.errors,
            }
            exp = raw.get("expect", {})
            checks = {}
            if "status" in exp:
                checks["status"] = (trace.verdict == exp["status"])
            if "event_score_min" in exp:
                checks["event_score_min"] = (trace.event_score >= exp["event_score_min"])
            if "event_score_max" in exp:
                checks["event_score_max"] = (trace.event_score <= exp["event_score_max"])
            row["checks"] = checks
            row["pass"] = all(checks.values()) if checks else True
        except Exception as e:
            results.append({"case": c, "ok": False, "error": f"{type(e).__name__}: {e}", "pass": False})
            continue
        results.append(row)
    return results


def summarize_gate_failures(state: dict[str, Any]) -> list[str]:
    sig = state.get("signal", {})
    if isinstance(sig, dict):
        return list(sig.get("failed", []))
    return []
