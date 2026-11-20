"""Nine KB-grounded agents (Sec 25-26). Each agent:

- takes the shared pipeline `state` dict,
- optionally calls the user's BYOK model for reasoning,
- ALWAYS grounds numeric scores in `engines.py` (LLM never invents scores),
- returns a partial state update + AgentStep summary.

Golden Rule (Sec 38): a catalyst is not a trade. Agents surface evidence;
only the orchestrator's Sec-18 gate may emit TRADE_CANDIDATE.
"""
from __future__ import annotations

import json
from typing import Any

from . import engines
from .llm import BYOKConfig, chat_json, parse_llm_json
from .schemas import Lifecycle

KB_GUARDRAIL = (
    "You are an event-driven Indian equities/F&O research assistant. "
    "Rules: (1) PROPOSAL != FINAL RULE — track lifecycle stage. "
    "(2) Compare ACTUAL vs EXPECTATION, not vs history. "
    "(3) A Tier-D (social/unverified) source alone can never justify high confidence. "
    "(4) A dramatic headline alone is never a trade; require expectation gap + "
    "liquidity + defined risk. (5) Return ONLY a JSON object."
)


async def _llm_or_mock(cfg: BYOKConfig, system: str, user: str,
                       mock: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Returns (parsed_json, model_label). Falls back to mock on any failure."""
    if cfg.mock_mode:
        return mock, "mock"
    try:
        raw = await chat_json(cfg, KB_GUARDRAIL + "\n" + system, user)
        return parse_llm_json(raw), cfg.label
    except Exception as e:  # transport/auth/model errors -> graceful degrade
        fb = dict(mock)
        fb["_llm_error"] = f"{type(e).__name__}: {e}"
        return fb, "mock(fallback)"


# 1 — Verifier: DISCOVERED -> VERIFIED (Sec 7)
async def verifier_agent(state: dict, cfg: BYOKConfig):
    ev = state["event"]
    tier = ev["source"]["tier"]
    mock = {
        "verification_status": "CONFIRMED" if tier in ("A", "B") else "UNVERIFIED",
        "reliability_note": (
            "Tier A/B: usable with standard checks."
            if tier in ("A", "B") else
            "Tier C/D: triggers investigation only; cannot justify high confidence alone."
        ),
    }
    out, model = await _llm_or_mock(
        cfg,
        "You verify catalysts. Assess source reliability tier and lifecycle stage "
        "(RUMOUR/PROPOSAL/CONSULTATION/FINAL/IMPLEMENTATION). Respond with keys "
        "verification_status (CONFIRMED/UNVERIFIED), reliability_note.",
        json.dumps(ev)[:2000],
        mock,
    )
    verified = out.get("verification_status", mock["verification_status"]) == "CONFIRMED"
    state["verified"] = verified
    state["verification"] = out
    step = {"agent": "verifier", "lifecycle": Lifecycle.VERIFIED.value,
            "summary": f"Tier {tier} -> {'VERIFIED' if verified else 'UNVERIFIED'} "
                       f"({ev.get('lifecycle_stage')})", "data": out, "model": model}
    return state, step


# 2 — Classifier: event taxonomy Sec 4
async def classifier_agent(state: dict, cfg: BYOKConfig):
    ev = state["event"]
    mock = {"event_type": ev.get("event_type"), "sector": ev.get("affected_sector"),
            "direction": ev.get("direction"), "entities": ev.get("affected_entities", []),
            "confidence": 0.8 if state.get("verified") else 0.4}
    out, model = await _llm_or_mock(
        cfg,
        "Classify the event into REGULATORY/EARNINGS/CORPORATE_ACTION/LEGAL/"
        "GOVERNMENT/MACRO with affected entities, sector, direction "
        "(BULLISH/BEARISH/NEUTRAL) and confidence 0-1. Keys: event_type, entities, "
        "sector, direction, confidence.",
        json.dumps({k: ev.get(k) for k in ("title", "description", "event_type",
                                          "affected_entities", "affected_sector")})[:2000],
        mock,
    )
    state["classification"] = out
    step = {"agent": "classifier", "lifecycle": Lifecycle.VERIFIED.value,
            "summary": f"{out.get('event_type')} / {out.get('direction')} "
                       f"(conf {out.get('confidence')})", "data": out, "model": model}
    return state, step


# 3 — Fundamental impact (Sec 8-9). Scores computed deterministically.
async def fundamental_agent(state: dict, cfg: BYOKConfig):
    ev = state["event"]
    score = engines.fundamental_impact_score(
        ev.get("revenue_impact", 0), ev.get("margin_impact", 0),
        ev.get("eps_impact", 0), ev.get("cashflow_impact", 0),
        ev.get("duration_score", 0), ev.get("exposure_score", 0))
    mock = {"transmission": ["Revenue/margin channel per KB Sec 8"],
            "narrative": f"Engine score {score} ({engines.fundamental_band(score)}). "
                         "Proposal stage: earnings impact is scenario-based, not final."}
    out, model = await _llm_or_mock(
        cfg,
        "Explain the fundamental transmission chain NEWS->ECONOMIC->EARNINGS impact "
        "in 2-4 bullets and state duration (Temporary/1-2 quarters/Multi-year/"
        "Structural/Unknown). Keys: transmission (list), narrative, duration.",
        json.dumps({"title": ev.get("title"), "description": ev.get("description"),
                    "engine_score": score, "band": engines.fundamental_band(score)})[:2000],
        mock,
    )
    out["engine_score"] = score
    out["band"] = engines.fundamental_band(score)
    state["fundamental_score"] = score
    state["fundamental"] = out
    step = {"agent": "fundamental", "lifecycle": Lifecycle.IMPACT_MAPPED.value,
            "summary": f"Fundamental {score}/100 ({out['band']})", "data": out, "model": model}
    return state, step


# 4 — Market reaction (Sec 10)
async def market_agent(state: dict, cfg: BYOKConfig):
    m = state["market"]
    gap = engines.event_gap(m.get("open_price", 0), m.get("prev_close", 0))
    rel = engines.relative_move(m.get("stock_move_pct", 0), m.get("sector_move_pct", 0))
    sig_score = engines.market_reaction_score(abs(m.get("stock_move_pct", 0)),
                                              m.get("volume_multiple", 1))
    significant = abs(m.get("stock_move_pct", 0)) >= 5 and m.get("volume_multiple", 1) >= 1.5
    mock = {"narrative": (
        f"Stock {m.get('stock_move_pct')}% (gap {gap}%, rel-sector {rel}pp), "
        f"volume x{m.get('volume_multiple')}. "
        f"{'Significant reaction.' if significant else 'Reaction not yet significant.'}")}
    out, model = await _llm_or_mock(
        cfg,
        "Judge whether the price/volume reaction is significant and whether it looks "
        "like repricing vs noise. Keys: narrative.",
        json.dumps({"market": m, "gap_pct": gap, "relative_pp": rel,
                    "reaction_score": sig_score})[:2000],
        mock,
    )
    out.update({"gap_pct": gap, "relative_pp": rel,
                "reaction_score": sig_score, "significant": significant})
    state["market_analysis"] = out
    step = {"agent": "market", "lifecycle": Lifecycle.MARKET_REACTION_MEASURED.value,
            "summary": f"Move {m.get('stock_move_pct')}% x{m.get('volume_multiple')} "
                       f"-> {'significant' if significant else 'not significant'}",
            "data": out, "model": model}
    return state, step


# 5 — Options intelligence (Sec 12-14). IV crush warning built in.
async def options_agent(state: dict, cfg: BYOKConfig):
    o = state["options"]
    em = engines.expected_move(o.get("spot", 0), o.get("iv", 50), o.get("dte", 5))
    fair_abs = abs(o.get("spot", 0) * o.get("estimated_fundamental_move_pct", 0) / 100)
    opp = engines.options_opportunity_score(o.get("iv_percentile", 50), em, fair_abs)
    liq = engines.liquidity_score(o.get("bid_ask_spread_pct", 1), o.get("oi", 0))
    warn = ("IV elevated — directional gains can be erased by IV crush + theta. "
            "Prefer spreads or wait." if o.get("iv_percentile", 0) >= 75 else
            "IV moderate — outrights viable if thesis confirms.")
    mock = {"narrative": (
        f"Expected move Rs.{em} vs fundamental-implied Rs.{fair_abs}. "
        f"IV {o.get('iv')}% (p{round(o.get('iv_percentile', 0))}). {warn}")}
    out, model = await _llm_or_mock(
        cfg,
        "Assess option pricing: compare current IV vs percentile, expected (implied) "
        "move vs fundamental-implied move, and warn about IV crush + theta. "
        "Keys: narrative.",
        json.dumps({"options": o, "expected_move_abs": em,
                    "fair_move_abs": fair_abs, "opp_score": opp,
                    "liquidity_score": liq})[:2000],
        mock,
    )
    out.update({"expected_move_abs": em, "fair_move_abs": fair_abs,
                "opp_score": opp, "liquidity_score": liq,
                "liquidity_ok": liq >= 50})
    state["options_analysis"] = out
    step = {"agent": "options", "lifecycle": Lifecycle.OPTIONS_ANALYZED.value,
            "summary": f"ExpMove Rs.{em}, IVp{round(o.get('iv_percentile', 0))}, "
                       f"liq {liq}", "data": out, "model": model}
    return state, step


# 6 — Expectation gap (Sec 11) — the most important concept.
async def expectation_agent(state: dict, cfg: BYOKConfig):
    m, o = state["market"], state["options"]
    gap = engines.expectation_gap(m.get("stock_move_pct", 0),
                                  o.get("estimated_fundamental_move_pct", 0))
    if gap <= -5:
        interp = ("Market priced MORE damage than estimates justify — investigate "
                  "mean-reversion/volatility setups, not automatic puts.")
    elif gap >= 5:
        interp = ("Market priced LESS damage (or more upside) than estimates — "
                  "investigate remaining downside risk.")
    else:
        interp = "Market move roughly matches estimates — no clear dislocation yet."
    mock = {"narrative": f"Gap {gap}pp (actual {m.get('stock_move_pct')}% vs "
                         f"est. {o.get('estimated_fundamental_move_pct')}%). {interp}"}
    out, model = await _llm_or_mock(
        cfg,
        "Interpret EXPECTED_FUNDAMENTAL_MOVE vs ACTUAL_MARKET_MOVE. Never conclude "
        "'must revert' — only state which side looks under/over-priced and what new "
        "information would confirm it. Keys: narrative.",
        json.dumps({"actual": m.get("stock_move_pct"),
                    "estimated": o.get("estimated_fundamental_move_pct"),
                    "gap_pp": gap})[:1500],
        mock,
    )
    out.update({"gap_pp": gap, "material": abs(gap) >= 3, "interpretation": interp})
    state["expectation"] = out
    step = {"agent": "expectation", "lifecycle": Lifecycle.OPTIONS_ANALYZED.value,
            "summary": f"Expectation gap {gap}pp", "data": out, "model": model}
    return state, step


# 7 — Second-order / contagion (Sec 22)
async def second_order_agent(state: dict, cfg: BYOKConfig):
    ev = state["event"]
    mock = {"nodes": ["Competitors", "Suppliers/Customers", "Distributors/Brokers",
                      "Banks/Lenders", "Sector index"],
            "narrative": "Contagion map from primary entity outward."}
    out, model = await _llm_or_mock(
        cfg,
        "Build a second-order contagion map: competitors, suppliers, customers, "
        "distributors, banks/lenders, sector index, substitutes. Keys: nodes (list), "
        "narrative.",
        json.dumps({"entities": ev.get("affected_entities"),
                    "sector": ev.get("affected_sector"),
                    "type": ev.get("event_type")})[:1500],
        mock,
    )
    state["second_order"] = out.get("nodes", mock["nodes"])
    step = {"agent": "second_order", "lifecycle": Lifecycle.OPTIONS_ANALYZED.value,
            "summary": f"Contagion: {', '.join(state['second_order'][:4])}...",
            "data": out, "model": model}
    return state, step


# 8 — Signal: Sec 15/18/19/29. Only agent allowed to propose a strategy.
async def signal_agent(state: dict, cfg: BYOKConfig):
    ev, m, o = state["event"], state["market"], state["options"]
    fund = state.get("fundamental_score", 0)
    ma = state.get("market_analysis", {})
    oa = state.get("options_analysis", {})
    ex = state.get("expectation", {})
    es = engines.event_score(
        ev["source"]["tier"], fund, ev.get("surprise_score", 50),
        ma.get("reaction_score", 0),
        min(abs(ex.get("gap_pp", 0)) * 10, 100),
        oa.get("opp_score", 0), oa.get("liquidity_score", 0))
    strategy = engines.select_strategy(
        state.get("classification", {}).get("direction", ev.get("direction", "UNKNOWN")),
        o.get("iv_percentile", 50), o.get("dte", 5), abs(ex.get("gap_pp", 0)))
    acct = state.get("account", {"capital": 2000.0})
    prem = o.get("put_premium" if "BEAR" in str(
        state.get("classification", {}).get("direction", "")) else "call_premium", 0) or 0
    sizing = engines.size_for_account(prem, o.get("lot_size", 100),
                                      acct.get("capital", 2000.0))
    passed, failed = engines.qualify_trade(
        state.get("verified", False), fund, ma.get("significant", False),
        ex.get("material", False), oa.get("liquidity_ok", False),
        True, sizing["fits"])
    if not passed:
        strategy = "NONE"
    status = "TRADE_CANDIDATE" if passed else "NO_TRADE"
    mock = {"narrative": (f"Gate {'PASS' if passed else 'FAIL: ' + '; '.join(failed)}. "
                          f"EventScore {es}. Strategy {strategy}.")}
    out, model = await _llm_or_mock(
        cfg,
        "Synthesize a signal verdict. Respect the qualification gate result given to "
        "you — if it failed, verdict must be NO_TRADE/WATCH with reasons and next "
        "triggers. Never invent a trade. Keys: narrative, next_triggers (list).",
        json.dumps({"gate_pass": passed, "failed": failed, "event_score": es,
                    "strategy": strategy, "sizing": sizing})[:2000],
        mock,
    )
    out.update({"event_score": es, "strategy": strategy, "sizing": sizing,
                "gate_pass": passed, "failed": failed, "status": status})
    state["signal"] = out
    step = {"agent": "signal", "lifecycle": (Lifecycle.TRADE_CANDIDATE
                                            if passed else Lifecycle.NO_TRADE).value,
            "summary": f"{status} ({strategy}) — ES {es}", "data": out, "model": model}
    return state, step


# 9 — Risk: Sec 16-17/20-21. Invalidation mandatory.
async def risk_agent(state: dict, cfg: BYOKConfig):
    m, o, sig = state["market"], state["options"], state.get("signal", {})
    spot = m.get("spot", 0)
    direction = state.get("classification", {}).get("direction", "UNKNOWN")
    if sig.get("status") == "TRADE_CANDIDATE":
        inv = round(spot * 1.02, 2) if "BEAR" in str(direction) else round(spot * 0.98, 2)
        tgt = round(spot + (-1 if "BEAR" in str(direction) else 1)
                    * sig.get("sizing", {}).get("cost", 0), 2)
        mock = {"max_loss": sig.get("sizing", {}).get("cost", 0),
                "narrative": (f"Defined risk Rs.{sig.get('sizing', {}).get('cost', 0)} "
                              f"(1 lot). Invalidation at Rs.{inv}.")}
    else:
        inv, tgt = None, None
        mock = {"max_loss": 0.0, "narrative": "NO_TRADE — no capital at risk. "
                "Paper-watch only; invalidation = thesis inputs change."}
    out, model = await _llm_or_mock(
        cfg,
        "Define risk per Rs.2000 rules: max loss, invalidation price + reason, "
        "holding period 1-3 sessions, no averaging down. Keys: narrative, "
        "invalidation_reason, holding_period.",
        json.dumps({"status": sig.get("status"), "spot": spot,
                    "direction": direction, "sizing": sig.get("sizing")})[:1500],
        mock,
    )
    out.setdefault("max_loss", mock["max_loss"])
    out["invalidation_price"] = inv
    out["target_underlying"] = tgt
    state["risk"] = out
    step = {"agent": "risk", "lifecycle": state["signal"]["status"],
            "summary": out.get("narrative", "")[:160], "data": out, "model": model}
    return state, step


PIPELINE = [
    verifier_agent, classifier_agent, fundamental_agent, market_agent,
    options_agent, expectation_agent, second_order_agent, signal_agent,
    risk_agent,
]
