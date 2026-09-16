"""Конфигурация сервиса. Всё через переменные среды — требование контейнеризации."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FALCON_", env_file=".env", extra="ignore")

    # Модель
    checkpoint: Path | None = None
    device: str = "cuda"
    batch_size: int = 32
    input_size: int = 256
    flip_tta: bool = True
    half: bool = True

    # Хранилище. Пустой DSN означает локальный SQLite — удобно для разработки,
    # в docker-compose всегда задан Postgres.
    database_dsn: str | None = None
    sqlite_path: Path = Path("data/falcon.sqlite3")

    # Порог отказа. None означает «порог не откалиброван»: сервис показывает
    # кандидатов, но честно не подтверждает совпадение. Переносить порог между
    # моделями нельзя, поэтому значения по умолчанию здесь нет намеренно.
    match_threshold: float | None = None

    # HTTP
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 12

    @property
    def size(self) -> tuple[int, int]:
        return (self.input_size, self.input_size)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
