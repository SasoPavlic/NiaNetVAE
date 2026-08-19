"""SQLite evidence ledger for shared-core architecture candidates."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class CandidateStore:
    def __init__(self, path: str | Path, table: str = "architecture_candidates_v1") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not table.replace("_", "").isalnum():
            raise ValueError(
                "SQLite candidate table must contain only letters, digits, and underscores."
            )
        self.table = table
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL,
                    search_contract_fingerprint TEXT NOT NULL,
                    architecture_hash TEXT NOT NULL,
                    genome_json TEXT NOT NULL,
                    architecture_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    obj_error REAL NOT NULL,
                    obj_pdm REAL NOT NULL,
                    obj_alarm_burden REAL NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(study_id, search_contract_fingerprint, architecture_hash)
                )
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_study "
                f"ON {self.table}(study_id, search_contract_fingerprint)"
            )

    def lookup(
        self,
        *,
        study_id: str,
        search_contract_fingerprint: str,
        architecture_hash: str,
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM {self.table} WHERE study_id=? AND search_contract_fingerprint=? "
                "AND architecture_hash=?",
                (study_id, search_contract_fingerprint, architecture_hash),
            ).fetchone()
        return _decode_row(row) if row is not None else None

    def insert(
        self,
        *,
        study_id: str,
        search_contract_fingerprint: str,
        architecture_hash: str,
        genome: tuple[float, ...],
        architecture: dict[str, Any],
        status: str,
        objectives: tuple[float, float, float],
        diagnostics: dict[str, Any],
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        values = (
            study_id,
            search_contract_fingerprint,
            architecture_hash,
            json.dumps(list(genome), separators=(",", ":")),
            json.dumps(architecture, sort_keys=True, separators=(",", ":")),
            str(status),
            float(objectives[0]),
            float(objectives[1]),
            float(objectives[2]),
            json.dumps(diagnostics, sort_keys=True, separators=(",", ":"), default=str),
            type(error).__name__ if error is not None else None,
            str(error) if error is not None else None,
            now,
        )
        with self._connection() as connection:
            connection.execute(
                f"""
                INSERT OR IGNORE INTO {self.table} (
                    study_id, search_contract_fingerprint, architecture_hash,
                    genome_json, architecture_json, status,
                    obj_error, obj_pdm, obj_alarm_burden, diagnostics_json,
                    error_type, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        cached = self.lookup(
            study_id=study_id,
            search_contract_fingerprint=search_contract_fingerprint,
            architecture_hash=architecture_hash,
        )
        if cached is None:
            raise RuntimeError("Candidate insert did not produce a readable SQLite record.")
        return cached

    def rows(self, *, study_id: str, search_contract_fingerprint: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            records = connection.execute(
                f"SELECT * FROM {self.table} WHERE study_id=? AND search_contract_fingerprint=? "
                "ORDER BY id",
                (study_id, search_contract_fingerprint),
            ).fetchall()
        return [_decode_row(row) for row in records]

    def frame(self, *, study_id: str, search_contract_fingerprint: str) -> pd.DataFrame:
        rows = self.rows(
            study_id=study_id,
            search_contract_fingerprint=search_contract_fingerprint,
        )
        flattened: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["genome"] = json.dumps(item.pop("genome"), separators=(",", ":"))
            item["architecture"] = json.dumps(
                item.pop("architecture"), sort_keys=True, separators=(",", ":")
            )
            diagnostics = item.pop("diagnostics")
            for key, value in diagnostics.items():
                item[f"diagnostic_{key}"] = value
            flattened.append(item)
        return pd.DataFrame(flattened)

    def checkpoint(self) -> Path:
        """Merge the WAL into the portable SQLite file before artifact hashing."""
        with self._connection() as connection:
            result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise RuntimeError(f"SQLite WAL checkpoint did not complete: {result}")
        return self.path


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["genome"] = tuple(float(value) for value in json.loads(payload.pop("genome_json")))
    payload["architecture"] = json.loads(payload.pop("architecture_json"))
    payload["diagnostics"] = json.loads(payload.pop("diagnostics_json"))
    return payload
