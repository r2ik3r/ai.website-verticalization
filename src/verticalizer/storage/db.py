import os
import logging
from contextlib import contextmanager
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

DB_DSN = os.getenv("DB_DSN", "postgresql+psycopg2://user:pass@localhost:5432/verticalizer")
engine = sa.create_engine(DB_DSN, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("DB transaction rolled back due to error")
        raise
    finally:
        session.close()
