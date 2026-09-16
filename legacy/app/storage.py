"""SQLite persistence and nearest-neighbour search."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .fingerprint import FINGERPRINT_VERSION, classify_match, cosine_similarity


class VehicleStore:
    def __init__(self, path: str | Path, version=FINGERPRINT_VERSION):
        self.path = str(path)
        self.version = version
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS vehicles (
                        vehicle_id TEXT PRIMARY KEY,
                        fingerprint TEXT NOT NULL,
                        fingerprint_version TEXT NOT NULL DEFAULT 'baseline-v1',
                        sample_count INTEGER NOT NULL DEFAULT 1,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )"""
                )
                columns = {row[1] for row in connection.execute("PRAGMA table_info(vehicles)")}
                if "fingerprint_version" not in columns:
                    connection.execute(
                        "ALTER TABLE vehicles ADD COLUMN fingerprint_version TEXT NOT NULL DEFAULT 'baseline-v1'"
                    )
                if "sample_count" not in columns:
                    connection.execute(
                        "ALTER TABLE vehicles ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1"
                    )

    def upsert(self, vehicle_id: str, fingerprint: np.ndarray, metadata: dict) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as connection:
            with connection:
                existing = connection.execute(
                    "SELECT fingerprint, fingerprint_version, sample_count FROM vehicles WHERE vehicle_id = ?",
                    (vehicle_id,),
                ).fetchone()
                sample_count = 1
                stored_fingerprint = fingerprint
                if existing and existing["fingerprint_version"] == self.version:
                    previous = np.asarray(json.loads(existing["fingerprint"]), dtype=np.float32)
                    if previous.shape == fingerprint.shape:
                        sample_count = int(existing["sample_count"]) + 1
                        stored_fingerprint = previous * (sample_count - 1) + fingerprint
                        norm = float(np.linalg.norm(stored_fingerprint))
                        if norm > 1e-8:
                            stored_fingerprint = stored_fingerprint / norm
                connection.execute(
                    """INSERT INTO vehicles(
                           vehicle_id, fingerprint, fingerprint_version, sample_count, metadata, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(vehicle_id) DO UPDATE SET
                         fingerprint=excluded.fingerprint,
                         fingerprint_version=excluded.fingerprint_version,
                         sample_count=excluded.sample_count,
                         metadata=excluded.metadata,
                         created_at=excluded.created_at""",
                    (
                        vehicle_id,
                        json.dumps(stored_fingerprint.tolist()),
                        self.version,
                        sample_count,
                        json.dumps(metadata, ensure_ascii=False),
                        timestamp,
                    ),
                )
        return {
            "vehicle_id": vehicle_id,
            "metadata": metadata,
            "sample_count": sample_count,
            "created_at": timestamp,
        }

    def list(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT vehicle_id, metadata, sample_count, fingerprint_version, created_at
                   FROM vehicles ORDER BY created_at DESC"""
            ).fetchall()
        return [
            {
                "vehicle_id": row["vehicle_id"],
                "metadata": json.loads(row["metadata"]),
                "sample_count": row["sample_count"],
                "fingerprint_version": row["fingerprint_version"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM vehicles").fetchall()
        matches = []
        for row in rows:
            if row["fingerprint_version"] != self.version:
                continue
            stored = np.asarray(json.loads(row["fingerprint"]), dtype=np.float32)
            decision = classify_match(cosine_similarity(query, stored))
            matches.append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "score": decision.score,
                    "verdict": decision.verdict,
                    "metadata": json.loads(row["metadata"]),
                    "fingerprint_version": row["fingerprint_version"],
                }
            )
        return sorted(matches, key=lambda item: item["score"], reverse=True)[:top_k]

    def delete(self, vehicle_id: str) -> bool:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute("DELETE FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
        return cursor.rowcount > 0
