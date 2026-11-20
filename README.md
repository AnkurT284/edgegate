<div align="center">

# ⚡ EdgeGate

**Event-driven F&O intelligence — nine agents, one gate.**

A catalyst is not a trade. EdgeGate runs market events through a nine-agent pipeline and only lets a setup through when all seven risk gates pass.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![BYOK](https://img.shields.io/badge/LLM-bring%20your%20own%20key-6E56CF)
![Mock mode](https://img.shields.io/badge/runs%20offline-no%20key%20needed-2F9E44)

</div>

---

Most event-driven trading tools stop at "there was news". EdgeGate answers the harder questions:
did the market already price it in, is the reaction real, is the move worth the risk, and can the
trade survive a ₹2,000 account?

## Why it exists

- **Catalysts are cheap; trades are expensive.** A verified catalyst still has to clear materiality,
  reaction size, expectation gap, liquidity, defined risk, and capital limits before it becomes a
  trade idea.
- **LLMs should not do arithmetic.** Every score, expected move, gap, and position size is computed
  by deterministic engines in `engines.py`. The model only writes the reasoning narrative — so a bad
  completion can change the wording, never the numbers.
- **No key, no problem.** With no API key the whole pipeline runs in mock mode, offline and
  deterministically, so demos and tests cost nothing.

## The pipeline

```
NEWS → VERIFIER → CLASSIFIER → FUNDAMENTAL → MARKET → OPTIONS
     → EXPECTATION GAP → SECOND-ORDER → SIGNAL → RISK
     → TRADE_CANDIDATE | NO_TRADE
```

Lifecycle: `DISCOVERED → VERIFIED → IMPACT_MAPPED → MARKET_REACTION_MEASURED → OPTIONS_ANALYZED → TRADE_CANDIDATE | NO_TRADE`

The final verdict is decided by one gate — **all seven** conditions must hold:

| # | Gate | Blocks when |
|---|------|-------------|
| 1 | Catalyst verified | Source tier too weak or event still a rumour |
| 2 | Fundamental material | Impact score below the materiality band |
| 3 | Reaction significant | Move and volume are noise-level |
| 4 | Expectation gap exists | Market has already priced the event |
| 5 | Liquidity OK | Spread too wide, OI too thin |
| 6 | Risk defined | No valid invalidation level |
| 7 | Fits capital | Lot size × premium exceeds the per-trade loss budget |

Fail any one of them and the system returns `NO_TRADE` **with the reasons attached**.

## The nine agents

| # | Agent | Responsibility | Grounding |
|---|-------|----------------|-----------|
| 1 | **Verifier** | Source tiering; tier-D alone never earns high confidence | Sec 7 |
| 2 | **Classifier** | Event taxonomy, lifecycle stage, direction | Sec 4 |
| 3 | **Fundamental** | Impact score from revenue / margin / EPS / cashflow / duration / exposure | Sec 8–9 |
| 4 | **Market** | Volume multiple, relative move vs sector, opening gap | Sec 10 |
| 5 | **Options** | IV percentile, expected move `S·IV·√(DTE/365)`, IV-crush warning | Sec 12–14 |
| 6 | **Expectation** | Actual-vs-estimated move gap — the core edge | Sec 11 |
| 7 | **Second-order** | Contagion graph across the affected sector | Sec 22 |
| 8 | **Signal** | Strategy selection, the seven-point gate, event score, alert level | Sec 15, 18, 19, 29 |
| 9 | **Risk** | ₹2,000 capital rules, trade schema, mandatory invalidation | Sec 16–17, 20 |

Section numbers reference the private knowledge base this system was built from; the code is
self-contained and the doc itself is not included in this repo.

## Quickstart

```bash
git clone https://github.com/AnkurT284/edgegate.git
cd edgegate

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline demo, no API key: the PB Fintech case resolves to NO_TRADE with reasons
python examples/run_demo.py

# Tests
pytest -q

# API + dashboard → http://127.0.0.1:8000
uvicorn event_fo.api:app --reload --app-dir src
```

## Bring your own key

Keys are **never stored server-side** and never logged. They arrive per request, live in memory for
one call, and are dropped — the dashboard keeps yours in browser `localStorage` only.

One OpenAI-compatible transport serves every provider:

| Provider | Default model | Notes |
|----------|---------------|-------|
| `openai` | `gpt-4o-mini` | |
| `openrouter` | `openai/gpt-4o-mini` | |
| `groq` | `llama-3.3-70b-versatile` | |
| `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | |
| `deepseek` | `deepseek-chat` | |
| `xai` | `grok-3-mini` | |
| `gemini` | `gemini-2.0-flash` | OpenAI-compatible endpoint |
| `ollama` | `llama3.1` | Local, no key |
| `custom` | — | Any `base_url` |

Three ways to supply a key, first match wins:

1. `llm` block in the request body — `{"provider","model","api_key","base_url"}`
2. Headers — `X-LLM-Provider`, `X-LLM-Model`, `X-LLM-API-Key`, `X-LLM-Base-URL`
3. Environment — `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` (see `.env.example`)

```bash
curl -X POST 127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -H 'X-LLM-Provider: openrouter' \
  -H 'X-LLM-Model: openai/gpt-4o-mini' \
  -H 'X-LLM-API-Key: sk-or-...' \
  -d @examples/pb_fintech_input.json | python3 -m json.tool
```

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /api/analyze` | Run the full nine-agent pipeline for one event + market + options snapshot |
| `GET /api/events` | Persisted event feed (`data/events.json`) |
| `GET /api/signals` | Persisted signals with full agent trace (`data/signals.json`) |
| `GET /api/health` | Liveness and mode |
| `GET /` | Dashboard: BYOK bar, catalyst/market/options forms, live agent trace |

Debug surface (`/api/debug/*`), for inspection rather than tuning:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/debug/agents` | Registered agents and their inputs/outputs |
| `POST /api/debug/trace` | Step-by-step trace with optional `stop_after` / `start_from` |
| `POST /api/debug/single/{agent}` | Run one agent with real upstream state |
| `GET /api/debug/failures` | Recent `NO_TRADE` rows with failed gates |
| `POST /api/debug/compare` | Diff two traces |

## CLI harness

```bash
python debug/harness.py list
python debug/harness.py run  --case examples/pb_fintech_input.json --agent verifier --verbose
python debug/harness.py run  --case examples/pb_fintech_input.json --all --verbose
python debug/harness.py trace --case examples/pb_fintech_input.json --save debug/last_trace.json
python debug/harness.py trace --case examples/pb_fintech_input.json --stop-after options
python debug/harness.py replay --trace debug/last_trace.json --compare debug/baseline.json
python debug/harness.py regression --dir examples
python debug/harness.py failures --limit 10
```

Mock mode is deterministic, so two traces of the same case compare clean — that is what makes the
regression workflow meaningful.

## Why the PB Fintech demo says NO_TRADE

The bundled case is a regulatory consultation on insurance distribution economics:

- fundamental ≈ 66/100 (High) — material, and it clears the first gates
- expectation gap −22pp — the market has already priced more damage than the −12% fundamental estimate
- IV at the 92nd percentile — expensive premium, spread-only logic applies
- one lot of 350 at ₹45 = **₹15,750**, far beyond the ₹500 max single loss

Verdict: **WATCH, not chase.** The system declining a tempting catalyst is the feature, not a failure.

## Project layout

```
src/event_fo/
  api.py           FastAPI app, BYOK resolution, static dashboard
  orchestrator.py  Pipeline runner → SignalOutput
  agents.py        The nine agents (LLM narrative + engine numbers)
  engines.py       All deterministic math: scores, gates, sizing
  llm.py           OpenAI-compatible BYOK transport, provider presets, mock fallback
  debug.py         Tracing, isolation, replay, regression, gate summaries
  debug_api.py     /api/debug/* router
  schemas.py       Pydantic contracts for every stage
  store.py         JSON-file event/signal persistence
  ui.html          Single-file dashboard
debug/harness.py   CLI for traces, replay, and regression runs
examples/          PB Fintech case + offline demo script
tests/             Engine math and debugging/trace behaviour
data/              Persisted events and signals (demo fixtures)
```

## Design principles

1. **Numbers from engines, words from models.** No score is ever parlayed from a completion.
2. **Fail closed.** Missing inputs degrade to `NO_TRADE`, never to a trade.
3. **Explain the refusal.** Every rejection lists the specific gate that failed.
4. **Keys are transient.** Never persisted, never logged, never echoed in responses.
5. **Debuggable by default.** Every step carries its lifecycle state, data, and model label.

## Roadmap

- [ ] Transcript and estimate-revision scrapers feeding the verifier
- [ ] Personal watchlists with Telegram/Slack webhooks on L3+ alerts
- [ ] Backtest log → signal library using the captured schema
- [ ] Live NSE option-chain poller into the options agent
- [ ] Swap the JSON store for Postgres

## Disclaimer

Research and decision-support tooling, **not financial advice**. F&O trading is high-risk. The
₹2,000 account framing exists to validate a process, not to generate income.
