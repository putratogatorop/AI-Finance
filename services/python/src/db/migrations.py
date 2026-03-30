import logging

from sqlalchemy import Engine

from src.db.models import Base

logger = logging.getLogger(__name__)


def run_migrations(engine: Engine) -> None:
    logger.info("Running database migrations...")
    Base.metadata.create_all(engine)
    logger.info("Database migrations complete.")
