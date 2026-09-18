"""Конфигурация сервиса. Всё через переменные среды — требование контейнеризации."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FALCON_", env_file=".env", extra="ignore")

    # Модель. Список путей через запятую — тогда поднимается ансамбль, тот же,
    # что идёт в сдачу. Это важно для порога: он калибруется под конкретный
    # состав моделей и между ними не переносится.
    checkpoints: str | None = None
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
    # Порог ансамбля, откалиброванный на локальном сплите (runs/v3-ensemble).
    # Для одиночной модели он другой: переносить между моделями нельзя.
    match_threshold: float | None = None

    # HTTP
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 12
    # Пачка до 200 снимков: лимит на всё тело запроса регистрации пачкой.
    max_batch_mb: int = 96
    # Защита от «бомб распаковки»: PNG в несколько килобайт, разворачивающийся
    # в гигапиксельный кадр. 40 Мп с запасом покрывают любые камеры (4K — 8.3 Мп).
    max_image_megapixels: int = 40

    # Ключ API. Пусто — доступ открыт (удобно для проверки жюри одной командой).
    # В эксплуатации задаётся FALCON_API_KEY, и все методы, кроме /api/health,
    # требуют заголовок X-API-Key.
    api_key: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        return (self.input_size, self.input_size)

    @property
    def checkpoint_paths(self) -> list[Path]:
        """Веса для загрузки: список из FALCON_CHECKPOINTS либо один checkpoint."""
        if self.checkpoints:
            return [Path(p.strip()) for p in self.checkpoints.split(",") if p.strip()]
        return [self.checkpoint] if self.checkpoint else []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
