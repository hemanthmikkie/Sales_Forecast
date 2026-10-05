"""
Module 11: Database Engine & Session Management
Provides SQLAlchemy session factory supporting MySQL and PostgreSQL
with connection pooling, pre-ping liveness, and resilient local fallback.
"""

import os
import sys
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Read database parameters from environment (.env file — never hardcode credentials)
DB_USER     = os.getenv("DB_USER",     os.getenv("POSTGRES_USER",     "root"))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", ""))
DB_HOST     = os.getenv("DB_HOST",     os.getenv("POSTGRES_HOST",     "localhost"))
DB_PORT     = os.getenv("DB_PORT",     "3306")
DB_NAME     = os.getenv("DB_NAME",     os.getenv("POSTGRES_DB",       "sales_forecast"))

# Default to MySQL (active on user system) or PostgreSQL / SQLite fallback
DEFAULT_MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_MYSQL_URL)

Base = declarative_base()


def get_engine():
    """
    Attempts to initialize production engine (MySQL / PostgreSQL).
    Falls back gracefully to SQLite if the database server is unreachable.
    """
    # 1. Try primary configured DATABASE_URL (MySQL or PostgreSQL)
    try:
        if DATABASE_URL.startswith("mysql") or DATABASE_URL.startswith("postgresql"):
            target_engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3} if "mysql" in DATABASE_URL or "postgresql" in DATABASE_URL else {}
            )
            with target_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_type = "MySQL" if "mysql" in DATABASE_URL else "PostgreSQL"
            print(f"[Database] Successfully connected to {db_type} database ('{DB_NAME}').")
            return target_engine
    except Exception as ex:
        print(f"[Database] Primary database connection failed ({type(ex).__name__}).")

    # 2. Resilient SQLite fallback
    sqlite_url = "sqlite:///./sales_forecast.db"
    sqlite_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False}
    )
    print(f"[Database] Fallback active: SQLite at '{sqlite_url}'")
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
