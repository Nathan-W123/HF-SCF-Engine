"""
pytest configuration for backend tests.
Sets up a test FastAPI client with an in-memory SQLite database.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from main import app


TEST_DATABASE_URL = "sqlite:///./test_hf_scf.db"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session")
def db_engine():
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)
    # Clean up test DB file
    if os.path.exists("test_hf_scf.db"):
        os.unlink("test_hf_scf.db")


@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ─── H2O test geometry ───────────────────────────────────────────────────────
H2O_XYZ = """O  0.000000  0.000000  0.117176
H  0.000000  0.757001 -0.468704
H  0.000000 -0.757001 -0.468704"""
