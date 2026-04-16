"""Shared DB config — reads DATABASE_URL env var with local-dev fallback."""
import os
from urllib.parse import urlparse

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)


def get_db_conn_kwargs() -> dict:
    """psycopg2.connect(**kwargs) style dict parsed from DB_URL."""
    p = urlparse(DB_URL)
    return dict(
        host=p.hostname,
        port=p.port or 5432,
        dbname=(p.path or "/").lstrip("/"),
        user=p.username,
        password=p.password,
    )
