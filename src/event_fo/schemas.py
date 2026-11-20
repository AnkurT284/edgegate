"""Pydantic schemas mirroring the KB data models (Sec 6, 20, 28)."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    REGULATORY = "REGULATORY"
    EARNINGS = "EARNINGS"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    LEGAL = "LEGAL"
    GOVERNMENT = "GOVERNMENT"
    MACRO = "MACRO"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class SourceTier(str, Enum):
    A = "A"  # primary: RBI/SEBI/IRDAI/NSE/BSE/filings/courts
    B = "B"  # high-quality secondary
    C = "C"  # analyst/research
    D = "D"  # social/unverified


class Lifecycle(str, Enum):
    DISCOVERED = "DISCOVERED"
    VERIFIED = "VERIFIED"
    IMPACT_MAPPED = "IMPACT_MAPPED"
    MARKET_REACTION_MEASURED = "MARKET_REACTION_MEASURED"
    OPTIONS_ANALYZED = "OPTIONS_ANALYZED"
    TRADE_CANDIDATE = "TRADE_CANDIDATE"
    NO_TRADE = "NO_TRADE"
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"
    TARGET_REACHED = "TARGET_REACHED"
    EXPIRED = "EXPIRED"


class Source(BaseModel):
    publisher: str = "Unknown"
    url: str = ""
    source_type: str = "news"  # official | news | filing | social ...
    tier: SourceTier = SourceTier.B


class EventInput(BaseModel):
    event_id: str = "EVT-001"
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType = EventType.REGULATORY
    lifecycle_stage: str = "CONSULTATION"  # RUMOUR/PROPOSAL/CONSULTATION/FINAL/IMPLEMENTATION
    title: str = ""
    description: str = ""
    source: Source = Field(default_factory=Source)
    affected_entities: list[str] = Field(default_factory=list)
    affected_sector: str = ""
    direction: Direction = Direction.UNKNOWN
    # 0-100 analyst inputs for deterministic engines (Sec 8-9)
    revenue_impact: float = 0
    margin_impact: float = 0
    eps_impact: float = 0
    cashflow_impact: float = 0
    duration_score: float = 0
    exposure_score: float = 0
    surprise_score: float = 50  # 0-100, used in EventScore
    is_verified: bool = False


class MarketReaction(BaseModel):
    symbol: str = "XYZ"
    spot: float = 1000.0
    prev_close: float = 1000.0
    open_price: float = 1000.0
    stock_move_pct: float = 0.0
    sector_move_pct: float = 0.0
    index_move_pct: float = 0.0
    volume_multiple: float = 1.0
    delivery_pct: Optional[float] = None


class OptionsSnapshot(BaseModel):
    symbol: str = "XYZ"
    expiry: date = Field(default_factory=date.today)
    spot: float = 1000.0
    iv: float = 50.0  # percent, e.g. 109.5
    iv_percentile: float = 50.0
    lot_size: int = 100
    call_premium: float = 0.0
    put_premium: float = 0.0
    bid_ask_spread_pct: float = 1.0
    dte: int = 5
    oi: float = 0.0
    oi_change_pct: float = 0.0
    # user-supplied fundamental estimate of fair move (Sec 11), signed %
    estimated_fundamental_move_pct: float = -10.0


class AccountRules(BaseModel):
    capital: float = 2000.0
    normal_risk_min: float = 200.0
    normal_risk_max: float = 400.0
    max_single_loss: float = 500.0


class AgentStep(BaseModel):
    agent: str
    lifecycle: Lifecycle
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    model: str = "mock"  # which LLM produced it, or 'mock'/'engine'


class SignalOutput(BaseModel):
    symbol: str = "XYZ"
    event_id: str = "EVT-001"
    status: str = "NO_TRADE"  # TRADE_CANDIDATE | NO_TRADE | WATCH | SETUP
    alert_level: int = 2  # 1-5 per Sec 29
    direction: Direction = Direction.UNKNOWN
    fundamental_score: float = 0.0
    event_score: float = 0.0
    expectation_gap_pct: float = 0.0
    expected_move_abs: float = 0.0
    suggested_strategy: str = "NONE"
    entry_spot_range: list[float] = Field(default_factory=list)
    premium_range: list[float] = Field(default_factory=list)
    max_loss: float = 0.0
    target_underlying: Optional[float] = None
    invalidation_price: Optional[float] = None
    holding_period: str = "1-3 sessions"
    reason: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    second_order: list[str] = Field(default_factory=list)
