"""FastAPI surface (KB Sec 27). BYOK keys arrive per-request and are never stored.

Key transport (first match wins):
  1. JSON body `llm` block: {"provider","model","api_key","base_url"}
  2. Headers: X-LLM-Provider / X-LLM-Model / X-LLM-API-Key / X-LLM-Base-URL
  3. Env defaults: LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_BASE_URL
No key => mock mode (deterministic, offline).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import store
from .llm import BYOKConfig
from .orchestrator import run_pipeline
from .schemas import AccountRules, EventInput, MarketReaction, OptionsSnapshot

try:
    from .debug_api import router as debug_router
except Exception:  # debug module optional in minimal installs
    debug_router = None  # type: ignore[assignment]

app = FastAPI(title="EdgeGate — event-driven F&O intelligence (BYOK agents)",
              version="0.1.0")

if debug_router is not None:
    app.include_router(debug_router)


class AnalyzeBody(BaseModel):
    event: EventInput
    market: MarketReaction
    options: OptionsSnapshot
    account: AccountRules = AccountRules()
    llm: Optional[dict[str, Any]] = None


def _llm_from(req: AnalyzeBody,
              h_provider: Optional[str], h_model: Optional[str],
              h_key: Optional[str], h_base: Optional[str]) -> BYOKConfig:
    b = req.llm or {}
    return BYOKConfig(
        provider=b.get("provider") or h_provider or os.getenv("LLM_PROVIDER", "openai"),
        model=b.get("model") or h_model or os.getenv("LLM_MODEL", ""),
        api_key=b.get("api_key") or h_key or os.getenv("LLM_API_KEY", ""),
        base_url=b.get("base_url") or h_base or os.getenv("LLM_BASE_URL", ""),
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "edgegate", "mode": "BYOK"}


@app.post("/api/analyze")
async def analyze(
    body: AnalyzeBody,
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_llm_api_key: Optional[str] = Header(None),
    x_llm_base_url: Optional[str] = Header(None),
):
    cfg = _llm_from(body, x_llm_provider, x_llm_model, x_llm_api_key, x_llm_base_url)
    out = await run_pipeline(body.event, body.market, body.options, body.account, cfg)
    data = out.model_dump(mode="json")
    data["llm"] = {"provider": cfg.provider, "model": cfg.label}  # never echo key
    store.save_event(body.event.model_dump(mode="json"))
    store.save_signal(data)
    return JSONResponse(data)


@app.get("/api/events")
async def events():
    return {"events": store.list_events()}


@app.get("/api/signals")
async def signals():
    return {"signals": store.list_signals()}


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "ui.html").read_text()
    return HTMLResponse(html)
