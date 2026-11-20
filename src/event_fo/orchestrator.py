"""Orchestrator: runs the 9-agent lifecycle (KB Sec 5 + Sec 25).

DISCOVERED -> VERIFIED -> IMPACT_MAPPED -> MARKET_REACTION_MEASURED
-> OPTIONS_ANALYZED -> TRADE_CANDIDATE | NO_TRADE
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from . import agents, engines
from .llm import BYOKConfig
from .schemas import (AccountRules, AgentStep, EventInput, Lifecycle,
                      MarketReaction, OptionsSnapshot, SignalOutput)


def _dump(model) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return dict(model)


async def run_pipeline(event: EventInput, market: MarketReaction,
                       options: OptionsSnapshot,
                       account: AccountRules,
                       llm: BYOKConfig) -> SignalOutput:
    state: dict[str, Any] = {
        "event": _dump(event), "market": _dump(market),
        "options": _dump(options), "account": _dump(account),
    }
    steps: list[AgentStep] = []
    for fn in agents.PIPELINE:
        state, raw = await fn(state, llm)
        steps.append(AgentStep(agent=raw["agent"],
                               lifecycle=Lifecycle(raw["lifecycle"]),
                               summary=raw.get("summary", ""),
                               data=raw.get("data", {}),
                               model=raw.get("model", "mock")))
    sig = state.get("signal", {})
    risk = state.get("risk", {})
    oa = state.get("options_analysis", {})
    ex = state.get("expectation", {})
    status = sig.get("status", "NO_TRADE")
    level = engines.alert_level(status, sig.get("event_score", 0),
                                sig.get("gate_pass", False))
    direction = state.get("classification", {}).get("direction",
                                                    event.direction.value)
    o = _dump(options)
    strat = sig.get("strategy", "NONE")
    if "PUT" in strat:
        prem = [o.get("put_premium", 0) * 0.9, o.get("put_premium", 0) * 1.1]
    elif "CALL" in strat:
        prem = [o.get("call_premium", 0) * 0.9, o.get("call_premium", 0) * 1.1]
    else:
        prem = []
    spot = _dump(market).get("spot", 0)
    return SignalOutput(
        symbol=_dump(market).get("symbol", "XYZ"),
        event_id=event.event_id,
        status=status,
        alert_level=level,
        direction=direction,  # type: ignore[arg-type]
        fundamental_score=state.get("fundamental_score", 0),
        event_score=sig.get("event_score", 0),
        expectation_gap_pct=ex.get("gap_pp", 0),
        expected_move_abs=oa.get("expected_move_abs", 0),
        suggested_strategy=strat,
        entry_spot_range=[round(spot * 0.995, 2), round(spot * 1.005, 2)] if prem else [],
        premium_range=[round(p, 2) for p in prem],
        max_loss=risk.get("max_loss", 0),
        target_underlying=risk.get("target_underlying"),
        invalidation_price=risk.get("invalidation_price"),
        holding_period=risk.get("holding_period", "1-3 sessions"),
        reason=sig.get("narrative", ""),
        steps=steps,
        second_order=state.get("second_order", []),
    )
