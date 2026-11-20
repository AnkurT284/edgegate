"""Debug API router — /api/debug/* endpoints (BYOK-safe, keys never echoed)."""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel

from . import debug as dbg
from . import store
from .llm import BYOKConfig
from .orchestrator import run_pipeline
from .schemas import AccountRules, EventInput, MarketReaction, OptionsSnapshot

router = APIRouter(prefix="/api/debug", tags=["debug"])


class DebugCaseBody(BaseModel):
    event: EventInput
    market: MarketReaction
    options: OptionsSnapshot
    account: AccountRules = AccountRules()
    llm: Optional[dict[str, Any]] = None
    stop_after: Optional[str] = None
    start_from: Optional[str] = None


def _llm_from(body_llm: Optional[dict[str, Any]],
              h_provider: Optional[str], h_model: Optional[str],
              h_key: Optional[str], h_base: Optional[str]) -> BYOKConfig:
    b = body_llm or {}
    return BYOKConfig(
        provider=b.get("provider") or h_provider or os.getenv("LLM_PROVIDER", "openai"),
        model=b.get("model") or h_model or os.getenv("LLM_MODEL", ""),
        api_key=b.get("api_key") or h_key or os.getenv("LLM_API_KEY", ""),
        base_url=b.get("base_url") or h_base or os.getenv("LLM_BASE_URL", ""),
    )


def _headers_kw(h_provider, h_model, h_key, h_base):
    return dict(h_provider=h_provider, h_model=h_model, h_key=h_key, h_base=h_base)


@router.get("/agents")
async def debug_agents():
    return {"agents": dbg.list_agents()}


@router.post("/trace")
async def debug_trace(
    body: DebugCaseBody,
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_llm_api_key: Optional[str] = Header(None),
    x_llm_base_url: Optional[str] = Header(None),
):
    cfg = _llm_from(body.llm, x_llm_provider, x_llm_model, x_llm_api_key, x_llm_base_url)
    state, trace = await dbg.run_traced(
        body.event, body.market, body.options, body.account, cfg,
        stop_after=body.stop_after, start_from=body.start_from)
    # Also compute the canonical SignalOutput for parity with /api/analyze.
    out = await run_pipeline(body.event, body.market, body.options, body.account, cfg)
    payload = trace.to_dict()
    payload["signal"] = out.model_dump(mode="json")
    return payload


@router.post("/single/{agent_name}")
async def debug_single(
    agent_name: str,
    body: DebugCaseBody,
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_llm_api_key: Optional[str] = Header(None),
    x_llm_base_url: Optional[str] = Header(None),
):
    cfg = _llm_from(body.llm, x_llm_provider, x_llm_model, x_llm_api_key, x_llm_base_url)
    if agent_name not in dbg._AGENT_FNS:
        return {"ok": False, "error": f"unknown agent '{agent_name}'",
                "known": sorted(dbg._AGENT_FNS)}
    # Build prerequisite state: run pipeline up to (excluding) target agent so
    # isolated runs see realistic upstream keys — true single-step debugging
    # without manual fixture assembly.
    state = dbg.build_initial_state(body.event, body.market, body.options, body.account)
    order = dbg.AGENT_NAMES
    if agent_name in order:
        idx = order.index(agent_name)
        for prior in order[:idx]:
            state, _ = await dbg.run_agent_isolated(prior, state, cfg)
    new_state, step = await dbg.run_agent_isolated(agent_name, state, cfg)
    from dataclasses import asdict
    return {"ok": True, "agent": agent_name, "llm": cfg.label,
            "mock_mode": cfg.mock_mode,
            "prereqs_ran": order[:order.index(agent_name)] if agent_name in order else [],
            "step": asdict(step),
            "state_keys": sorted(new_state.keys())}


@router.get("/failures")
async def debug_failures(limit: int = 20):
    """Last NO_TRADE signals with gate reasons — the replay queue."""
    rows = store.list_signals()[:limit]
    out = []
    for r in rows:
        steps = r.get("steps", []) if isinstance(r, dict) else []
        sig_step = next((s for s in steps if s.get("agent") == "signal"), {})
        data = sig_step.get("data", {}) if isinstance(sig_step, dict) else {}
        out.append({
            "event_id": r.get("event_id"), "symbol": r.get("symbol"),
            "status": r.get("status"), "event_score": r.get("event_score"),
            "failed_gates": data.get("failed", []),
            "strategy": r.get("suggested_strategy"),
        })
    return {"failures": [x for x in out if x["status"] == "NO_TRADE"], "count": len(out)}


@router.post("/compare")
async def debug_compare(payload: dict[str, Any]):
    # POST {"a": <trace_dict>, "b": <trace_dict>}
    ta = payload.get("a", payload) if isinstance(payload, dict) else payload
    tb = payload.get("b") if isinstance(payload, dict) else None
    if not isinstance(ta, dict) or not isinstance(tb, dict):
        return {"ok": False, "error": "provide two trace dicts as {\"a\": ..., \"b\": ...}"}
    return {"ok": True, "diff": dbg.compare_traces(ta, tb)}
