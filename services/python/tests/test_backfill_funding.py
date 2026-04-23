import sys

sys.path.insert(0, ".")

from datetime import datetime

SAMPLE_GATEIO_RESPONSE = [
    {"t": 1704067200, "r": "0.000100", "T": "1704067200"},
    {"t": 1704096000, "r": "-0.000050", "T": "1704096000"},
]


def test_parse_gateio_funding():
    from scripts.backfill_funding_rates import parse_gateio_funding

    rows = parse_gateio_funding("SOL", SAMPLE_GATEIO_RESPONSE)
    assert len(rows) == 2
    assert rows[0]["asset"] == "SOL"
    assert rows[0]["funding_rate"] == 0.0001
    assert rows[0]["source"] == "gateio"
    assert isinstance(rows[0]["timestamp"], datetime)


def test_parse_gateio_funding_empty():
    from scripts.backfill_funding_rates import parse_gateio_funding

    rows = parse_gateio_funding("SOL", [])
    assert rows == []
