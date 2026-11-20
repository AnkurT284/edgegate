"""Tiny JSON-file store for events + signals (MVP; swap for Postgres later)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE = Path(__file__).resolve().parents[2] / "data"
STORE.mkdir(exist_ok=True)
EVENTS = STORE / "events.json"
SIGNALS = STORE / "signals.json"


def _read(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _write(p: Path, rows: list[dict[str, Any]]) -> None:
    p.write_text(json.dumps(rows[-500:], indent=2))


def save_event(row: dict[str, Any]) -> None:
    rows = _read(EVENTS)
    rows.append(row)
    _write(EVENTS, rows)


def save_signal(row: dict[str, Any]) -> None:
    rows = _read(SIGNALS)
    rows.append(row)
    _write(SIGNALS, rows)


def list_events() -> list[dict[str, Any]]:
    return list(reversed(_read(EVENTS)))


def list_signals() -> list[dict[str, Any]]:
    return list(reversed(_read(SIGNALS)))
