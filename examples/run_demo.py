"""Offline demo: runs the full 9-agent pipeline in mock mode (no API key)."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from event_fo.llm import BYOKConfig
from event_fo.orchestrator import run_pipeline
from event_fo.schemas import AccountRules, EventInput, MarketReaction, OptionsSnapshot


async def main() -> None:
    raw = json.loads((Path(__file__).parent / "pb_fintech_input.json").read_text())
    out = await run_pipeline(
        EventInput(**raw["event"]), MarketReaction(**raw["market"]),
        OptionsSnapshot(**raw["options"]), AccountRules(**raw["account"]),
        BYOKConfig(provider="openai", api_key=""),  # mock mode
    )
    print(json.dumps(out.model_dump(mode="json"), indent=2)[:4000])
    print(f"\nVERDICT: {out.status} | strategy={out.suggested_strategy} "
          f"| fund={out.fundamental_score} event={out.event_score} "
          f"| gap={out.expectation_gap_pct}pp | maxloss=Rs.{out.max_loss}")


if __name__ == "__main__":
    asyncio.run(main())
