import os
from unittest.mock import patch


def test_settings_loads_defaults():
    env = {
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpass",
        "POSTGRES_DB": "testdb",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "DATA_LAKE_PATH": "/tmp/data",
    }
    with patch.dict(os.environ, env, clear=False):
        from src.config import Settings

        settings = Settings()
        assert settings.POSTGRES_USER == "testuser"
        assert settings.POSTGRES_DB == "testdb"
        assert settings.DATA_LAKE_PATH == "/tmp/data"
        assert settings.POSITION_SIZE_IDR == 1_000_000
        assert settings.SCHEDULER_ENABLED is True


def test_settings_database_url():
    env = {
        "POSTGRES_USER": "u",
        "POSTGRES_PASSWORD": "p",
        "POSTGRES_DB": "d",
        "POSTGRES_HOST": "h",
        "POSTGRES_PORT": "5432",
        "DATA_LAKE_PATH": "/tmp/data",
    }
    with patch.dict(os.environ, env, clear=False):
        from src.config import Settings

        settings = Settings()
        assert settings.database_url == "postgresql://u:p@h:5432/d"
