"""
Module 11: Database Engine & Session Management
Provides SQLAlchemy session factory with PostgreSQL connection support,
configurable connection pooling, and resilient local fallback.
"""

import os
import sys
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

load_dotenv()

# Read database URL from environment
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sales_forecast")

DEFAULT_PG_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)

Base = declarative_base()

def get_engine():
    """
    Attempts to initialize PostgreSQL engine. If PostgreSQL is unreachable
    in the active environment, falls back to local SQLite to ensure test and
    development autonomy.
    """
    try:
        if DATABASE_URL.startswith("postgresql"):
            pg_engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3}
            )
            # Test connection
            with pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[Database] Successfully connected to PostgreSQL instance.")
            return pg_engine
    except Exception as ex:
        print(f"[Database] PostgreSQL connection failed ({ex}). Switching to SQLite fallback for autonomous local runtime.")
    
    # Fallback SQLite engine
    sqlite_url = "sqlite:///./sales_forecast.db"
    sqlite_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False}
    )
    print(f"[Database] Active database: SQLite at '{sqlite_url}'")
    return sqlite_engine

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """FastAPI Dependency for database session lifecycle."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
