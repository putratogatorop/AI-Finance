from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings


def create_db_engine(settings: Settings):
    return create_engine(settings.database_url, pool_pre_ping=True, pool_size=5)


def create_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
