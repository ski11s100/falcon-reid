"""Хранилище галереи эмбеддингов.

Раздел 6 ТЗ требует реляционную, документо-ориентированную или векторную СУБД и
называет в качестве примеров PostgreSQL с pgvector, FAISS и Milvus. Основная
реализация — PostgreSQL + pgvector: это честная СУБД с транзакциями и
метаданными, а не отдельный индекс рядом с базой.

Хранилище спрятано за протоколом намеренно. Это позволяет:
  * поднимать сервис локально без Postgres (SQLite-реализация) при разработке;
  * подключить приближённый поиск (HNSW) той же ручкой — раздел 10 ТЗ просит
    продемонстрировать ANN на галерее порядка 10^6 объектов.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class GalleryItem:
    """Запись галереи: наблюдение ТС с его эмбеддингом и метаданными.

    thumbnail — небольшой JPEG кропа. Оператор сверяет машины глазами, и список
    из одних идентификаторов вида 3d7fd47923ac45d39fb159ecbec21e75 для принятия
    решения бесполезен. Раздел 10 ТЗ прямо говорит о продуктовой зрелости
    интерфейса.
    """

    image_id: str
    vehicle_id: str | None
    embedding: np.ndarray
    metadata: dict
    thumbnail: bytes | None = None


@dataclass(frozen=True)
class Match:
    image_id: str
    vehicle_id: str | None
    score: float
    metadata: dict
    thumbnail: bytes | None = None


class VectorRepository(Protocol):
    """Контракт хранилища. Реализации взаимозаменяемы для сервиса."""

    #: какой поиск фактически используется: "hnsw" или "exact"
    ann_index: str

    def initialise(self, dimension: int) -> None: ...
    def upsert(self, items: list[GalleryItem]) -> int: ...
    def search(self, embedding: np.ndarray, top_k: int) -> list[Match]: ...
    def embedding_of(self, image_id: str) -> np.ndarray | None: ...
    def delete(self, image_id: str) -> bool: ...
    def count(self) -> int: ...
    def list_vehicles(self, limit: int = 100) -> list[dict]: ...
    def close(self) -> None: ...


def _normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Нулевой эмбеддинг: сравнивать такой вектор бессмысленно")
    return vector / norm


class PgVectorRepository:
    """PostgreSQL + pgvector. Основное хранилище для поставки.

    Поиск идёт по оператору `<=>` (косинусное расстояние). Поскольку все векторы
    хранятся нормированными, косинусное сходство равно 1 - расстояние.
    """

    def __init__(self, dsn: str, table: str = "gallery"):
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._register_vector = register_vector
        self.table = table
        self.connection = psycopg.connect(dsn, autocommit=True)
        self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.connection)
        self._lock = threading.Lock()

    def _existing_dimension(self) -> int | None:
        """Размерность вектора в уже существующей таблице, если она есть."""
        row = self.connection.execute(
            """SELECT a.atttypmod FROM pg_attribute a
               JOIN pg_class c ON c.oid = a.attrelid
               WHERE c.relname = %s AND a.attname = 'embedding' AND a.attnum > 0""",
            (self.table,),
        ).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def initialise(self, dimension: int) -> None:
        # Размерность зашита в тип колонки, и CREATE TABLE IF NOT EXISTS тихо
        # пропустит создание при несовпадении. Дальше любая запись падала бы с
        # невнятной ошибкой 500. Такое бывает при смене модели или переходе на
        # ансамбль: 2048 превращается в 4096.
        existing = self._existing_dimension()
        if existing is not None and existing != dimension:
            raise RuntimeError(
                f"В базе лежит таблица {self.table} с вектором {existing} измерений, "
                f"а модель выдаёт {dimension}. Схема несовместима.\n"
                f"Если данные галереи не нужны, удалите таблицу:\n"
                f"    docker compose exec postgres "
                f"psql -U falcon -d falcon -c 'DROP TABLE {self.table}'\n"
                f"Либо поднимите базу заново: docker compose down -v"
            )

        with self._lock:
            self.connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    image_id   TEXT PRIMARY KEY,
                    vehicle_id TEXT,
                    embedding  vector({dimension}) NOT NULL,
                    metadata   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    thumbnail  BYTEA,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            # Таблица могла быть создана прежней версией без миниатюр:
            # CREATE TABLE IF NOT EXISTS новую колонку не добавляет.
            self.connection.execute(
                f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS thumbnail BYTEA")
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_vehicle_id ON {self.table} (vehicle_id)")

        self.ann_index = self._try_create_ann_index(dimension)

    def _try_create_ann_index(self, dimension: int) -> str:
        """Пытается построить HNSW-индекс, но не падает, если это невозможно.

        pgvector строит HNSW только для векторов не длиннее 2000 измерений.
        Наш эмбеддинг — 2048 (и 4096 у ансамбля), поэтому индекс не создаётся.
        Раньше исключение здесь убивало запуск сервиса: контейнер уходил в
        бесконечный перезапуск с сообщением про размерность, никак не связанным
        с тем, что видел пользователь.

        Отсутствие индекса не деградация. Измерения (docs/SCALABILITY.md)
        показали, что точный перебор быстрее приближённого поиска вплоть до
        сотни тысяч объектов, а демонстрационная галерея на порядки меньше.
        HNSW нужен от 10^6 объектов — но там и размерность придётся понижать:
        2048 измерений на миллион векторов это 8 ГБ только под данные.
        """
        try:
            with self._lock:
                self.connection.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table}_embedding_hnsw
                    ON {self.table} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            return "hnsw"
        except self._psycopg.errors.ProgramLimitExceeded:
            print(json.dumps({
                "ann_index": {
                    "status": "не создан",
                    "dimension": dimension,
                    "limit": 2000,
                    "reason": "pgvector строит HNSW только до 2000 измерений",
                    "effect": "поиск идёт точным перебором; на галерее такого "
                              "размера он и так быстрее (docs/SCALABILITY.md)",
                }
            }, ensure_ascii=False), flush=True)
            return "exact"

    def upsert(self, items: list[GalleryItem]) -> int:
        if not items:
            return 0
        rows = [(i.image_id, i.vehicle_id, _normalise(i.embedding), json.dumps(i.metadata),
                 i.thumbnail) for i in items]
        with self._lock, self.connection.cursor() as cursor:
            cursor.executemany(
                f"""INSERT INTO {self.table}
                        (image_id, vehicle_id, embedding, metadata, thumbnail)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO UPDATE
                    SET vehicle_id = EXCLUDED.vehicle_id,
                        embedding  = EXCLUDED.embedding,
                        metadata   = EXCLUDED.metadata,
                        thumbnail  = EXCLUDED.thumbnail""",
                rows,
            )
        return len(rows)

    def search(self, embedding: np.ndarray, top_k: int) -> list[Match]:
        vector = _normalise(embedding)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT image_id, vehicle_id, 1 - (embedding <=> %s) AS score,
                           metadata, thumbnail
                    FROM {self.table} ORDER BY embedding <=> %s LIMIT %s""",
                (vector, vector, top_k),
            )
            return [Match(r[0], r[1], float(r[2]), r[3] or {},
                          bytes(r[4]) if r[4] is not None else None)
                    for r in cursor.fetchall()]

    def embedding_of(self, image_id: str) -> np.ndarray | None:
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT embedding FROM {self.table} WHERE image_id = %s", (image_id,))
            row = cursor.fetchone()
        return np.asarray(row[0], dtype=np.float32) if row else None

    def delete(self, image_id: str) -> bool:
        with self._lock, self.connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE image_id = %s", (image_id,))
            return cursor.rowcount > 0

    def count(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {self.table}")
            return int(cursor.fetchone()[0])

    def list_vehicles(self, limit: int = 100) -> list[dict]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT g.vehicle_id, count(*) AS shots, max(g.created_at) AS last_seen,
                           (SELECT t.thumbnail FROM {self.table} t
                             WHERE t.vehicle_id = g.vehicle_id AND t.thumbnail IS NOT NULL
                             ORDER BY t.created_at DESC LIMIT 1),
                           (SELECT t.image_id FROM {self.table} t
                             WHERE t.vehicle_id = g.vehicle_id AND t.thumbnail IS NOT NULL
                             ORDER BY t.created_at DESC LIMIT 1)
                    FROM {self.table} g WHERE g.vehicle_id IS NOT NULL
                    GROUP BY g.vehicle_id ORDER BY last_seen DESC LIMIT %s""",
                (limit,),
            )
            return [{"vehicle_id": r[0], "shots": int(r[1]), "last_seen": r[2].isoformat(),
                     "thumbnail": bytes(r[3]) if r[3] is not None else None, "image_id": r[4]}
                    for r in cursor.fetchall()]

    def close(self) -> None:
        self.connection.close()


class SQLiteRepository:
    """Локальная реализация для разработки и тестов, без внешних сервисов.

    Точный перебор по всей галерее. Для конкурсных объёмов (750 объектов) это
    доли миллисекунды, для продакшена используется pgvector с HNSW.
    """

    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._dimension: int | None = None

    def initialise(self, dimension: int) -> None:
        self._dimension = dimension
        self.ann_index = "exact"
        with self._lock:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS gallery (
                    image_id   TEXT PRIMARY KEY,
                    vehicle_id TEXT,
                    embedding  BLOB NOT NULL,
                    metadata   TEXT NOT NULL DEFAULT '{}',
                    thumbnail  BLOB,
                    created_at REAL NOT NULL
                )
            """)
            columns = {row[1] for row in
                       self.connection.execute("PRAGMA table_info(gallery)").fetchall()}
            if "thumbnail" not in columns:
                self.connection.execute("ALTER TABLE gallery ADD COLUMN thumbnail BLOB")
            self.connection.commit()

    def upsert(self, items: list[GalleryItem]) -> int:
        import time
        rows = [(i.image_id, i.vehicle_id, _normalise(i.embedding).tobytes(),
                 json.dumps(i.metadata), i.thumbnail, time.time()) for i in items]
        with self._lock:
            self.connection.executemany(
                """INSERT INTO gallery
                       (image_id, vehicle_id, embedding, metadata, thumbnail, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET
                     vehicle_id=excluded.vehicle_id, embedding=excluded.embedding,
                     metadata=excluded.metadata, thumbnail=excluded.thumbnail,
                     created_at=excluded.created_at""",
                rows,
            )
            self.connection.commit()
        return len(rows)

    def search(self, embedding: np.ndarray, top_k: int) -> list[Match]:
        vector = _normalise(embedding)
        with self._lock:
            rows = self.connection.execute(
                "SELECT image_id, vehicle_id, embedding, metadata, thumbnail "
                "FROM gallery").fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Match(rows[i][0], rows[i][1], float(scores[i]), json.loads(rows[i][3]),
                      rows[i][4]) for i in order]

    def embedding_of(self, image_id: str) -> np.ndarray | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT embedding FROM gallery WHERE image_id = ?", (image_id,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).copy() if row else None

    def delete(self, image_id: str) -> bool:
        with self._lock:
            cursor = self.connection.execute("DELETE FROM gallery WHERE image_id = ?", (image_id,))
            self.connection.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        with self._lock:
            return int(self.connection.execute("SELECT count(*) FROM gallery").fetchone()[0])

    def list_vehicles(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self.connection.execute(
                """SELECT g.vehicle_id, count(*), max(g.created_at),
                          (SELECT t.thumbnail FROM gallery t
                            WHERE t.vehicle_id = g.vehicle_id AND t.thumbnail IS NOT NULL
                            ORDER BY t.created_at DESC LIMIT 1),
                          (SELECT t.image_id FROM gallery t
                            WHERE t.vehicle_id = g.vehicle_id AND t.thumbnail IS NOT NULL
                            ORDER BY t.created_at DESC LIMIT 1)
                   FROM gallery g WHERE g.vehicle_id IS NOT NULL GROUP BY g.vehicle_id
                   ORDER BY max(g.created_at) DESC LIMIT ?""", (limit,)).fetchall()
        return [{"vehicle_id": r[0], "shots": int(r[1]), "last_seen": r[2],
                 "thumbnail": r[3], "image_id": r[4]} for r in rows]

    def close(self) -> None:
        self.connection.close()


def build_repository(dsn: str | None, sqlite_path: Path | str | None = None) -> VectorRepository:
    """Выбирает хранилище: Postgres при заданном DSN, иначе локальный SQLite."""
    if dsn:
        return PgVectorRepository(dsn)
    return SQLiteRepository(sqlite_path or ":memory:")
