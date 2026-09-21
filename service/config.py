"""Конфигурация сервиса. Всё через переменные среды — требование контейнеризации."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: docker-compose передаёт незаданные переменные пустой
    # строкой (FALCON_RETENTION_DAYS: ${FALCON_RETENTION_DAYS:-}), и без этого
    # флага пустое значение не разобралось бы как число и сервис не стартовал.
    model_config = SettingsConfigDict(env_prefix="FALCON_", env_file=".env", extra="ignore",
                                      env_ignore_empty=True)

    # Модель. Список путей через запятую — тогда поднимается ансамбль, тот же,
    # что идёт в сдачу. Это важно для порога: он калибруется под конкретный
    # состав моделей и между ними не переносится.
    checkpoints: str | None = None
    checkpoint: Path | None = None
    device: str = "cuda"
    batch_size: int = 32
    input_size: int = 256
    # Отражение кадра (flip TTA) выключено, как в сдаче: у ансамбля с CLIP оно
    # даёт +0.003 mAP@10 ценой двойных вычислений (falcon/submit.py).
    flip_tta: bool = False
    half: bool = True
    # PCA-проекция склеенного вектора ансамбля (falcon/extract.Projection).
    # Порог калибруется в пространстве проекции, поэтому она задаётся вместе
    # с составом моделей и порогом — в .env, Dockerfile и docker-compose.
    projection: Path | None = None

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

    # Журнал обращений (service/audit.py): кто, когда и что искал. Пусто —
    # записи идут только в журнал контейнера; путь — ещё и в файл на томе.
    audit_log: Path | None = None
    # Срок хранения галереи в днях. Пусто — снимки не удаляются сами (так
    # нужно для показа с заранее наполненной галереей). В эксплуатации срок
    # задаёт регламент заказчика: 152-ФЗ требует хранить персональные данные
    # не дольше, чем этого требует цель обработки.
    retention_days: float | None = None

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
