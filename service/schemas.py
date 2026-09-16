"""Контракты API. Pydantic-модели попадают прямо в OpenAPI-схему.

Раздел 6 ТЗ требует формализовать все методы взаимодействия через OpenAPI, а
ответ 45 уточняет: сервис принимает изображение ВМЕСТЕ с координатами BBox и не
занимается детекцией сам.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """Рамка ТС в пикселях исходного кадра — формат из train.csv."""

    x: float = Field(description="Левая граница в пикселях исходного кадра")
    y: float = Field(description="Верхняя граница в пикселях исходного кадра")
    w: float = Field(gt=1, description="Ширина рамки, больше 1 пикселя")
    h: float = Field(gt=1, description="Высота рамки, больше 1 пикселя")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


class ImagePayload(BaseModel):
    image_base64: str = Field(description="Кадр в Base64 или Data URL (JPEG/PNG)")
    bbox: BBox | None = Field(
        default=None,
        description="Рамка ТС. Если не указана, изображение считается готовым кропом",
    )

    @field_validator("image_base64")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Поле image_base64 пустое")
        return value


class RegisterRequest(ImagePayload):
    image_id: str = Field(max_length=120, description="Уникальный идентификатор наблюдения")
    vehicle_id: str | None = Field(default=None, max_length=120,
                                   description="Известный идентификатор ТС, если он есть")
    metadata: dict = Field(default_factory=dict, description="Произвольные метаданные наблюдения")


class SearchRequest(ImagePayload):
    top_k: int = Field(default=10, ge=1, le=100, description="Сколько кандидатов вернуть")
    threshold: float | None = Field(
        default=None,
        description="Переопределить порог принятия решения для этого запроса",
    )


class QualityReport(BaseModel):
    """Оценка пригодности кадра. Раздел 4 ТЗ требует базовую валидацию входа."""

    width: int
    height: int
    sharpness: float = Field(description="Дисперсия градиента; низкая означает смаз")
    brightness: float = Field(description="Средняя яркость в диапазоне 0..1")
    warnings: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    image_id: str
    vehicle_id: str | None
    score: float = Field(description="Косинусное сходство, не калиброванная вероятность")
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Ответ поиска. Отказ выражается полем accepted, а не пустым списком."""

    accepted: bool = Field(description="Принято ли решение о совпадении")
    verdict: Literal["совпадение", "требуется проверка", "совпадений нет"]
    refusal_reason: str | None = Field(default=None)
    threshold: float | None = Field(default=None, description="Порог, применённый к этому запросу")
    matches: list[Candidate] = Field(default_factory=list,
                                     description="Подтверждённые совпадения, пусто при отказе")
    candidates: list[Candidate] = Field(default_factory=list,
                                        description="Полная выдача Top-K независимо от порога")
    quality: QualityReport
    elapsed_ms: float


class RegisterResponse(BaseModel):
    image_id: str
    vehicle_id: str | None
    embedding_dim: int
    gallery_size: int
    quality: QualityReport
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    embedding_dim: int
    device: str
    gallery_size: int
    threshold_calibrated: bool
    storage: str
