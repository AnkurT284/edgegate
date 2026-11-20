#!/usr/bin/env python3
"""Debugging agent harness CLI.

Usage:
  python debug/harness.py list
  python debug/harness.py run --case examples/pb_fintech_input.json --agent verifier --verbose
  python debug/harness.py run --case examples/pb_fintech_input.json --all --verbose
  python debug/harness.py trace --case examples/pb_fintech_input.json --save debug/last_trace.json
  python debug/harness.py trace --case examples/pb_fintech_input.json --stop-after options
  python debug/harness.py replay --trace debug/last_trace.json --compare debug/baseline.json
  python debug/harness.py regression --dir examples --save debug/regression.json
  python debug/harness.py failures --limit 10

LLM mode:
  Mock (offline, default): no key needed.
  Live: --provider openrouter --model openai/gpt-4o-mini --api-key sk-... (or env
  LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL). Keys are never written to traces.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_fo import debug as dbg  # noqa: E402
from event_fo.llm import BYOKConfig  # noqa: E402


def llm_from_args(a) -> BYOKConfig:
    return BYOKConfig(
        provider=getattr(a, "provider", None) or os.getenv("LLM_PROVIDER", "openai"),
        model=getattr(a, "model", None) or os.getenv("LLM_MODEL", ""),
        api_key=getattr(a, "api_key", None) or os.getenv("LLM_API_KEY", ""),
        base_url=getattr(a, "base_url", None) or os.getenv("LLM_BASE_URL", ""),
    )


def cmd_list(_a):
    for ag in dbg.list_agents():
        print(f"{ag['order']}: {ag['name']:13s} reads={','.join(ag['reads'])} "
              f"writes={','.join(ag['writes'])}")


async def _run_case(case_path: str, llm: BYOKConfig, agent: str | None,
                    stop_after: str | None, verbose: bool, save: str | None,
                    snapshots: bool = False):
    raw = dbg.load_case(case_path)
    event, market, options, account = dbg.coerce_case(raw)
    if agent:
        state = dbg.build_initial_state(event, market, options, account)
        # run prereqs so isolated run sees realistic upstream state
        order = dbg.AGENT_NAMES
        idx = order.index(agent)
        for prior in order[:idx]:
            state, _ = await dbg.run_agent_isolated(prior, state, llm)
        state, step = await dbg.run_agent_isolated(agent, state, llm)
        print(f"[{agent}] {step.lifecycle} ({step.duration_ms}ms, model={step.model})")
        print(f"  summary: {step.summary}")
        print(f"  delta: +{step.state_delta.get('added')} ~{step.state_delta.get('changed')}")
        if verbose:
            print(json.dumps(step.data, indent=2, default=str)[:3000])
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            Path(save).write_text(json.dumps(
                {"case": case_path, "step": step.__dict__}, indent=2, default=str))
            print(f"saved -> {save}")
        return
    snap: dict = {}
    state, trace = await dbg.run_traced(
        event, market, options, account, llm,
        stop_after=stop_after, capture_snapshots=snapshots,
        snapshots=snap if snapshots else None)
    print(f"verdict={trace.verdict} score={trace.event_score} total={trace.total_ms}ms "
          f"llm={trace.llm_label} mock={trace.mock_mode}")
    for s in trace.steps:
        flag = " !" if s.error else ""
        print(f"  {s.agent:13s} {s.lifecycle:28s} {s.duration_ms:7.1f}ms "
              f"[{s.model}]{flag} :: {s.summary[:120]}")
        if verbose:
            print(f"      delta +{s.state_delta.get('added')} "
                  f"~{s.state_delta.get('changed')} -{s.state_delta.get('removed')}")
            d = json.dumps(s.data, default=str)
            print(f"      data: {d[:600]}")
    if trace.errors:
        print("ERRORS:")
        for e in trace.errors:
            print(f"  {e}")
    gates = dbg.summarize_gate_failures(state)
    if gates:
        print(f"failed gates: {gates}")
    if save:
        dbg.save_trace(trace, save)
        print(f"saved trace -> {save}")


def cmd_run(a):
    llm = llm_from_args(a)
    if a.agent and a.agent not in dbg._AGENT_FNS:
        print(f"unknown agent '{a.agent}'. Known: {sorted(dbg._AGENT_FNS)}", file=sys.stderr)
        sys.exit(2)
    asyncio.run(_run_case(a.case, llm, a.agent, a.stop_after, a.verbose, a.save))


def cmd_trace(a):
    llm = llm_from_args(a)
    asyncio.run(_run_case(a.case, llm, None, a.stop_after, a.verbose, a.save, snapshots=a.snapshots))


def cmd_replay(a):
    cur = json.loads(Path(a.trace).read_text())
    base = json.loads(Path(a.compare).read_text())
    # allow files that are {"trace": {...}} or raw trace dicts
    cur = cur.get("trace", cur)
    base = base.get("trace", base)
    diff = dbg.compare_traces(base, cur)
    print(json.dumps(diff, indent=2, default=str)[:4000])
    print(f"\nMATCH={diff['match']}")


def cmd_regression(a):
    llm = llm_from_args(a)
    cases = sorted(glob.glob(str(Path(a.dir) / "*.json")))
    # exclude non-case files gracefully inside run_regression error handling
    results = asyncio.run(dbg.run_regression(cases, llm))
    n_pass = sum(1 for r in results if r.get("pass"))
    for r in results:
        if r.get("ok"):
            print(f"{'PASS' if r.get('pass') else 'FAIL'} {r['case']}: "
                  f"{r['verdict']} score={r['event_score']} "
                  f"gates={r.get('failed_gates')} checks={r.get('checks')}")
        else:
            print(f"ERROR {r['case']}: {r.get('error')}")
    print(f"\n{n_pass}/{len(results)} passed")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(results, indent=2, default=str))
        print(f"saved -> {a.save}")
    if n_pass != len(results) and a.strict:
        sys.exit(1)


def cmd_failures(a):
    sys.path.insert(0, str(ROOT / "src"))
    from event_fo import store
    rows = store.list_signals()[:a.limit]
    shown = 0
    for r in rows:
        if r.get("status") != "NO_TRADE":
            continue
        shown += 1
        steps = r.get("steps", [])
        sig = next((s for s in steps if s.get("agent") == "signal"), {})
        print(f"{r.get('event_id')} {r.get('symbol')} score={r.get('event_score')} "
              f"strategy={r.get('suggested_strategy')} "
              f"gates={sig.get('data', {}).get('failed')}")
    if not shown:
        print("(no NO_TRADE signals stored yet — run examples/run_demo.py via API or add inputs)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Debug harness for the 9-agent F&O pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list agents in pipeline order")

    for name in ("run", "trace"):
        q = sub.add_parser(name, help=f"{name} a case file with tracing")
        q.add_argument("--case", required=True, help="path to input JSON (e.g. examples/pb_fintech_input.json)")
        q.add_argument("--save", default=None, help="save trace/step JSON here")
        q.add_argument("--stop-after", default=None, choices=dbg.AGENT_NAMES,
                       help="step-through: halt after this agent")
        q.add_argument("--verbose", action="store_true")
        q.add_argument("--provider", default=None)
        q.add_argument("--model", default=None)
        q.add_argument("--api-key", default=None)
        q.add_argument("--base-url", default=None)
        if name == "run":
            q.add_argument("--agent", default=None, choices=dbg.AGENT_NAMES,
                           help="run single agent in isolation (with prereqs)")
            q.add_argument("--all", dest="agent_all", action="store_true",
                           help="run full pipeline (same as trace)")
        else:
            q.add_argument("--snapshots", action="store_true",
                           help="capture per-agent state snapshots (for advanced replay)")

    r = sub.add_parser("replay", help="compare two saved traces")
    r.add_argument("--trace", required=True)
    r.add_argument("--compare", required=True)

    g = sub.add_parser("regression", help="run all case JSONs in a dir")
    g.add_argument("--dir", default="examples")
    g.add_argument("--save", default=None)
    g.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    g.add_argument("--provider", default=None)
    g.add_argument("--model", default=None)
    g.add_argument("--api-key", default=None)
    g.add_argument("--base-url", default=None)

    f = sub.add_parser("failures", help="list recent NO_TRADE signals with gate reasons")
    f.add_argument("--limit", type=int, default=20)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "list":
        cmd_list(a)
    elif a.cmd == "run":
        if getattr(a, "agent_all", False):
            a.agent = None
        cmd_run(a)
    elif a.cmd == "trace":
        cmd_trace(a)
    elif a.cmd == "replay":
        cmd_replay(a)
    elif a.cmd == "regression":
        cmd_regression(a)
    elif a.cmd == "failures":
        cmd_failures(a)


if __name__ == "__main__":
    main()
