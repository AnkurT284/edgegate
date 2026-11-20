import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from event_fo import debug as dbg
from event_fo.llm import BYOKConfig

CASE = Path(__file__).resolve().parents[1] / "examples" / "pb_fintech_input.json"
MOCK = BYOKConfig(provider="openai", api_key="")


def _case():
    raw = dbg.load_case(CASE)
    return dbg.coerce_case(raw)


def test_trace_runs_all_nine_mock():
    event, market, options, account = _case()
    state, trace = asyncio.run(dbg.run_traced(event, market, options, account, MOCK))
    assert len(trace.steps) == 9
    assert [s.agent for s in trace.steps] == dbg.AGENT_NAMES
    assert trace.verdict in ("TRADE_CANDIDATE", "NO_TRADE")
    assert trace.total_ms >= 0
    # PB Fintech fixture is designed to fail the capital gate
    assert trace.verdict == "NO_TRADE"
    assert any("Capital" in g or "Position" in g
               for g in dbg.summarize_gate_failures(state))


def test_single_agent_isolation():
    event, market, options, account = _case()
    state = dbg.build_initial_state(event, market, options, account)
    new_state, step = asyncio.run(dbg.run_agent_isolated("verifier", state, MOCK))
    assert step.agent == "verifier"
    assert "verified" in new_state
    assert step.duration_ms >= 0
    assert step.model == "mock"


def test_stop_after_step_through():
    event, market, options, account = _case()
    _, trace = asyncio.run(dbg.run_traced(event, market, options, account, MOCK,
                                    stop_after="fundamental"))
    assert [s.agent for s in trace.steps] == ["verifier", "classifier", "fundamental"]


def test_regression_and_compare():
    results = asyncio.run(dbg.run_regression([CASE], MOCK))
    assert len(results) == 1 and results[0]["ok"]
    event, market, options, account = _case()
    _, t1 = asyncio.run(dbg.run_traced(event, market, options, account, MOCK))
    _, t2 = asyncio.run(dbg.run_traced(event, market, options, account, MOCK))
    diff = dbg.compare_traces(t1.to_dict(), t2.to_dict())
    assert diff["match"] is True  # mock mode is deterministic


def test_unknown_agent_raises():
    import asyncio
    event, market, options, account = _case()
    state = dbg.build_initial_state(event, market, options, account)
    with pytest.raises(ValueError):
        asyncio.run(dbg.run_agent_isolated("nonexistent", state, MOCK))


def test_load_case_rejects_signal_rows(tmp_path):
    p = tmp_path / "sig.json"
    p.write_text(json.dumps({"steps": [], "status": "NO_TRADE"}))
    with pytest.raises(ValueError):
        dbg.load_case(p)
