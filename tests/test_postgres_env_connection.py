from pathlib import Path

import pytest

from nianetvae.storage.experiment_storage import (
    _load_dotenv_if_present,
    _postgres_params_from_env,
    psycopg2,
)


def test_postgres_connection_from_env_smoke() -> None:
    """
    Integration smoke test:
    - loads DB settings from .env via the same helper used in runtime
    - validates required env variables
    - verifies database connectivity with a simple SELECT 1
    """
    if not Path(".env").is_file():
        pytest.skip("No .env file found in project root; skipping Postgres connectivity test.")

    if psycopg2 is None:
        pytest.skip("psycopg2 is not installed; skipping Postgres connectivity test.")

    assert _load_dotenv_if_present(".env"), "Failed to load .env file."

    db_params, missing_env_vars = _postgres_params_from_env()
    assert not missing_env_vars, f"Missing required Postgres env vars: {missing_env_vars}"

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_params["host"],
            port=db_params["port"],
            dbname=db_params["dbname"],
            user=db_params["user"],
            password=db_params["password"],
            sslmode=db_params.get("sslmode", "disable"),
            connect_timeout=10,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        row = cur.fetchone()
        assert row is not None and row[0] == 1
    finally:
        if conn is not None:
            conn.close()
