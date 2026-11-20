"""Deterministic engines from the KB — no LLM needed.

Implements:
- Sec 9:  FundamentalImpactScore
- Sec 10: Volume Multiple / Relative Move / Event Gap
- Sec 11: Expectation Gap
- Sec 14: Expected Move = Spot * IV * sqrt(DTE/365)
- Sec 15: Strategy selection helper
- Sec 16-17: Risk engine for the Rs.2000 experimental account
- Sec 18: Trade qualification gate
- Sec 19: EventScore
- Sec 29: Alert levels
"""
from __future__ import annotations

import math

TIER_RELIABILITY = {"A": 95.0, "B": 75.0, "C": 50.0, "D": 20.0}


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def fundamental_impact_score(revenue: float, margin: float, eps: float,
                             cashflow: float, duration: float,
                             exposure: float) -> float:
    """Sec 9 weighted score, each input 0-100."""
    s = (0.25 * revenue + 0.20 * margin + 0.20 * eps
         + 0.15 * cashflow + 0.10 * duration + 0.10 * exposure)
    return round(clamp(s), 2)


def fundamental_band(score: float) -> str:
    if score < 20:
        return "Minimal"
    if score < 40:
        return "Low"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "High"
    return "Severe"


def volume_multiple(current: float, avg20: float) -> float:
    if avg20 <= 0:
        return 1.0
    return round(current / avg20, 2)


def relative_move(stock_pct: float, sector_pct: float) -> float:
    return round(stock_pct - sector_pct, 2)


def event_gap(open_price: float, prev_close: float) -> float:
    if prev_close <= 0:
        return 0.0
    return round(open_price / prev_close - 1, 4) * 100


def expected_move(spot: float, iv_pct: float, dte: int) -> float:
    """Sec 14. Returns absolute expected move in price points."""
    iv = iv_pct / 100.0
    return round(abs(spot) * iv * math.sqrt(max(dte, 0) / 365.0), 2)


def expectation_gap(actual_market_move_pct: float,
                    estimated_fundamental_move_pct: float) -> float:
    """Signed gap: actual - estimated. Large negative = market priced more damage."""
    return round(actual_market_move_pct - estimated_fundamental_move_pct, 2)


def event_score(source_tier: str, fundamental: float, surprise: float,
                market_reaction: float, gap: float, options_opp: float,
                liquidity: float) -> float:
    """Sec 19. All inputs 0-100."""
    s = (0.20 * TIER_RELIABILITY.get(source_tier, 50.0)
         + 0.20 * fundamental + 0.15 * surprise + 0.15 * market_reaction
         + 0.10 * gap + 0.10 * options_opp + 0.10 * liquidity)
    return round(clamp(s), 2)


def market_reaction_score(abs_move_pct: float, volume_mult: float) -> float:
    """Heuristic 0-100: big move + high volume = significant reaction."""
    move_part = clamp(abs(abs_move_pct) * 5, 0, 70)  # 14% -> 70
    vol_part = clamp((volume_mult - 1) * 15, 0, 30)  # 3x -> 30
    return round(move_part + vol_part, 2)


def liquidity_score(bid_ask_spread_pct: float, oi: float) -> float:
    if bid_ask_spread_pct > 5 or oi <= 0:
        return 20.0
    if bid_ask_spread_pct > 2:
        return 55.0
    return 85.0


def options_opportunity_score(iv_percentile: float,
                              expected_move_abs: float,
                              fair_move_abs: float) -> float:
    """High when IV is not prohibitive and implied move < fair move estimate."""
    if expected_move_abs <= 0:
        return 30.0
    edge = (fair_move_abs - expected_move_abs) / expected_move_abs  # >0 = cheap
    edge_part = clamp(50 + edge * 100, 0, 70)
    iv_penalty = clamp((iv_percentile - 50) * 0.6, -30, 30)
    return round(clamp(edge_part - max(iv_penalty, 0) * 0.5 + 10), 2)


def qualify_trade(event_verified: bool, fundamental: float,
                  market_sig: bool, gap_material: bool,
                  liquidity_ok: bool, risk_defined: bool,
                  fits_capital: bool) -> tuple[bool, list[str]]:
    """Sec 18 gate. Returns (pass, failed_reasons)."""
    checks = {
        "Catalyst Verified": event_verified,
        "Fundamental Impact Material (>=40)": fundamental >= 40,
        "Market Reaction Significant": market_sig,
        "Expectation Gap Exists (|gap|>=3)": gap_material,
        "Options Liquidity Acceptable": liquidity_ok,
        "Risk Defined": risk_defined,
        "Position Fits Capital": fits_capital,
    }
    failed = [k for k, v in checks.items() if not v]
    return (len(failed) == 0, failed)


def select_strategy(direction: str, iv_percentile: float,
                    dte: int, gap_abs: float,
                    binary: bool = False) -> str:
    """Sec 15 helper. Conservative: spreads when IV hot, outrights when cool."""
    d = direction.upper()
    hot_iv = iv_percentile >= 75
    if binary and gap_abs >= 5:
        return "LONG_STRANGLE" if hot_iv else "LONG_STRADDLE"
    if d == "BULLISH":
        return "DEBIT_CALL_SPREAD" if hot_iv or dte <= 3 else "LONG_CALL"
    if d == "BEARISH":
        return "DEBIT_PUT_SPREAD" if hot_iv or dte <= 3 else "LONG_PUT"
    return "NONE"


def size_for_account(premium_per_share: float, lot_size: int,
                     capital: float = 2000.0,
                     max_loss: float = 500.0) -> dict:
    """Sec 16-17: one lot cost vs budget. Never recommend > max_single_loss."""
    if premium_per_share <= 0 or lot_size <= 0:
        return {"lots": 0, "cost": 0.0, "fits": False,
                "reason": "Invalid premium/lot"}
    cost = round(premium_per_share * lot_size, 2)
    fits = cost <= max_loss and cost <= capital
    return {"lots": 1 if fits else 0, "cost": cost, "fits": fits,
            "reason": ("Fits Rs.2000 rules" if fits
                       else f"One lot costs Rs.{cost} > Rs.{max_loss} budget -> NO_TRADE or paper trade")}


def alert_level(status: str, event_score_val: float,
                qualified: bool) -> int:
    """Sec 29: 1 INFO, 2 WATCH, 3 SETUP, 4 TRIGGERED, 5 INVALIDATED."""
    if status == "INVALIDATED":
        return 5
    if qualified:
        return 3
    if event_score_val >= 60:
        return 2
    return 1
