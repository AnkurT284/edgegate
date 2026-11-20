import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from event_fo import engines


def test_fundamental_score_pb_case():
    s = engines.fundamental_impact_score(70, 65, 70, 50, 60, 85)
    assert 60 <= s <= 75, s  # High band


def test_expected_move_math():
    assert engines.expected_move(1200, 109.5, 3) > 100  # hot IV, intuitive magnitude


def test_expectation_gap_sign():
    assert engines.expectation_gap(-34.0, -12.0) == -22.0


def test_capital_gate_blocks_oversize_lot():
    r = engines.size_for_account(45, 350)  # Rs.15,750 >> Rs.500 budget
    assert r["fits"] is False


def test_qualify_gate_requires_all_seven():
    ok, failed = engines.qualify_trade(True, 66, True, True, True, True, False)
    assert ok is False and any("Capital" in f for f in failed)
