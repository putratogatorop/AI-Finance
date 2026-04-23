import csv
import io
import sys
import zipfile

sys.path.insert(0, ".")


def make_fake_zip(rows: list[list[str]]) -> bytes:
    """Create a fake Binance Vision zip with CSV data."""
    buf = io.BytesIO()
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    for row in rows:
        writer.writerow(row)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-15m-2024-01.csv", csv_buf.getvalue())
    return buf.getvalue()


def test_parse_binance_csv():
    from scripts.backfill_binance_15m import parse_binance_csv

    raw = (
        "1704067200000,42000.0,42100.0,41900.0,42050.0,123.45,"
        "1704068099999,5180000.0,1500,60.0,2520000.0,0\n"
    )
    rows = parse_binance_csv(raw)
    assert len(rows) == 1
    assert rows[0]["open"] == 42000.0
    assert rows[0]["close"] == 42050.0
    assert rows[0]["trades"] == 1500
    assert rows[0]["taker_buy_base"] == 60.0


def test_generate_months():
    from scripts.backfill_binance_15m import generate_months

    months = generate_months(2024, 1, 2024, 3)
    assert months == ["2024-01", "2024-02", "2024-03"]
