"""FastAPI-сервис ФАЛЬКОН: инференс, векторный поиск и интерфейс оператора.

Архитектура (раздел 6 ТЗ, микросервисное разделение ответственности):
  * этот процесс      — инференс ReID-модели и HTTP-контракт;
  * PostgreSQL+pgvector — хранение галереи эмбеддингов и метаданных;
  * статический клиент  — тонкий клиент в браузере, работает только через API.

Спецификация OpenAPI отдаётся по /openapi.json, интерактивный Swagger — по /docs.
"""

from __future__ import annotations

import base64
import binascii
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from falcon.extract import ExtractorConfig, FeatureExtractor
from falcon.model import VehicleReID

from .config import Settings, get_settings
from .repository import GalleryItem, VectorRepository, build_repository
from .schemas import (
    Candidate,
    HealthResponse,
    QualityReport,
    RegisterRequest,
    RegisterResponse,
    SearchRequest,
    SearchResponse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

DESCRIPTION = """
Сервис формирования устойчивого цифрового признака транспортного средства и
поиска того же автомобиля по разным камерам **без использования государственного
номера**.

На вход подаётся кадр и координаты рамки ТС. Детекция в задачу не входит:
координаты предоставляются внешней системой.

Сервис умеет отказываться от ответа. Если в галерее нет уверенного совпадения,
возвращается `accepted: false` — это штатное поведение, а не ошибка.
"""


class ServiceState:
    """Живые зависимости сервиса: модель и хранилище."""

    extractor: FeatureExtractor | None = None
    repository: VectorRepository | None = None
    model_name: str = "не загружена"
    storage_name: str = "не подключено"


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.checkpoint and Path(settings.checkpoint).is_file():
        state.extractor = FeatureExtractor(
            checkpoint=settings.checkpoint,
            config=ExtractorConfig(size=settings.size, batch_size=settings.batch_size,
                                   num_workers=0, device=settings.device,
                                   half=settings.half, flip_tta=settings.flip_tta),
        )
        state.model_name = f"{VehicleReID.ARCHITECTURE} ({Path(settings.checkpoint).name})"
    else:
        # Без обученного чекпоинта сервис поднимается, но честно сообщает об этом
        # в /api/health. Молча отдавать случайные эмбеддинги было бы хуже.
        state.model_name = "чекпоинт не задан (FALCON_CHECKPOINT)"

    state.repository = build_repository(settings.database_dsn, settings.sqlite_path)
    state.storage_name = "postgresql+pgvector" if settings.database_dsn else "sqlite"
    if state.extractor is not None:
        state.repository.initialise(state.extractor.feature_dim)

    yield

    if state.repository is not None:
        state.repository.close()


app = FastAPI(
    title="ФАЛЬКОН — цифровой признак транспортного средства",
    description=DESCRIPTION,
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)


def require_model() -> FeatureExtractor:
    if state.extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Укажите путь к весам в FALCON_CHECKPOINT.",
        )
    return state.extractor


def require_repository() -> VectorRepository:
    if state.repository is None:
        raise HTTPException(status_code=503, detail="Хранилище недоступно")
    return state.repository


def decode_image(payload: str, max_bytes: int) -> Image.Image:
    encoded = payload.split(",", 1)[1] if "," in payload else payload
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Некорректная строка Base64") from exc
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413,
                            detail=f"Изображение больше {max_bytes // (1024 * 1024)} МБ")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение") from exc


def crop_to_bbox(image: Image.Image, bbox) -> Image.Image:
    if bbox is None:
        return image
    x, y, w, h = bbox.as_tuple()
    if x >= image.width or y >= image.height or x + w <= 0 or y + h <= 0:
        raise HTTPException(status_code=400, detail="Рамка не пересекается с изображением")
    left = max(0, min(image.width - 1, int(round(x))))
    top = max(0, min(image.height - 1, int(round(y))))
    right = max(left + 1, min(image.width, int(round(x + w))))
    bottom = max(top + 1, min(image.height, int(round(y + h))))
    return image.crop((left, top, right, bottom))


def assess_quality(crop: Image.Image) -> QualityReport:
    """Базовая валидация входа, которую требует раздел 4 ТЗ."""
    array = np.asarray(crop.convert("L"), dtype=np.float32) / 255.0
    dy, dx = np.gradient(array)
    sharpness = float(np.var(np.hypot(dx, dy)))
    brightness = float(array.mean())

    warnings: list[str] = []
    if min(crop.width, crop.height) < 48:
        warnings.append("Очень маленькая рамка: различающих деталей может не остаться")
    if sharpness < 0.0015:
        warnings.append("Кадр выглядит смазанным или расфокусированным")
    if brightness < 0.12:
        warnings.append("Очень тёмный кадр")
    elif brightness > 0.9:
        warnings.append("Кадр пересвечен")
    return QualityReport(width=crop.width, height=crop.height,
                         sharpness=round(sharpness, 6), brightness=round(brightness, 4),
                         warnings=warnings)


def embed(extractor: FeatureExtractor, crop: Image.Image) -> np.ndarray:
    return extractor.encode_image(crop)


@app.get("/api/health", response_model=HealthResponse, tags=["Служебные"],
         summary="Состояние сервиса")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    repository = state.repository
    return HealthResponse(
        status="ok" if state.extractor is not None else "degraded",
        model=state.model_name,
        embedding_dim=state.extractor.feature_dim if state.extractor else 0,
        device=state.extractor.device.type if state.extractor else "нет",
        gallery_size=repository.count() if repository else 0,
        threshold_calibrated=settings.match_threshold is not None,
        storage=state.storage_name,
    )


@app.post("/api/gallery/register", response_model=RegisterResponse, status_code=201,
          tags=["Галерея"], summary="Добавить наблюдение в галерею")
def register(request: RegisterRequest, settings: Settings = Depends(get_settings),
             extractor: FeatureExtractor = Depends(require_model),
             repository: VectorRepository = Depends(require_repository)) -> RegisterResponse:
    started = time.perf_counter()
    image = decode_image(request.image_base64, settings.max_upload_mb * 1024 * 1024)
    crop = crop_to_bbox(image, request.bbox)
    quality = assess_quality(crop)
    embedding = embed(extractor, crop)

    repository.upsert([GalleryItem(
        image_id=request.image_id, vehicle_id=request.vehicle_id,
        embedding=embedding,
        metadata={**request.metadata, "crop": [crop.width, crop.height]},
    )])
    return RegisterResponse(
        image_id=request.image_id, vehicle_id=request.vehicle_id,
        embedding_dim=int(embedding.size), gallery_size=repository.count(),
        quality=quality, elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["Поиск"],
          summary="Найти похожие ТС или обоснованно отказаться")
def search(request: SearchRequest, settings: Settings = Depends(get_settings),
           extractor: FeatureExtractor = Depends(require_model),
           repository: VectorRepository = Depends(require_repository)) -> SearchResponse:
    started = time.perf_counter()
    image = decode_image(request.image_base64, settings.max_upload_mb * 1024 * 1024)
    crop = crop_to_bbox(image, request.bbox)
    quality = assess_quality(crop)
    embedding = embed(extractor, crop)

    found = repository.search(embedding, request.top_k)
    candidates = [Candidate(image_id=m.image_id, vehicle_id=m.vehicle_id,
                            score=round(m.score, 6), metadata=m.metadata) for m in found]

    threshold = request.threshold if request.threshold is not None else settings.match_threshold
    elapsed = round((time.perf_counter() - started) * 1000, 2)

    if threshold is None:
        # Порог не откалиброван — подтверждать совпадение нельзя. Показываем
        # кандидатов оператору, но решение за человеком.
        return SearchResponse(
            accepted=False, verdict="требуется проверка",
            refusal_reason="Порог принятия решения не откалиброван для текущей модели",
            threshold=None, matches=[], candidates=candidates, quality=quality, elapsed_ms=elapsed)

    if not candidates:
        return SearchResponse(
            accepted=False, verdict="совпадений нет",
            refusal_reason="Галерея пуста", threshold=threshold,
            matches=[], candidates=[], quality=quality, elapsed_ms=elapsed)

    accepted = candidates[0].score >= threshold
    return SearchResponse(
        accepted=accepted,
        verdict="совпадение" if accepted else "совпадений нет",
        refusal_reason=None if accepted else
            f"Лучший кандидат {candidates[0].score:.3f} ниже порога {threshold:.3f}",
        threshold=threshold,
        matches=[c for c in candidates if c.score >= threshold] if accepted else [],
        candidates=candidates, quality=quality, elapsed_ms=elapsed)


@app.get("/api/gallery", tags=["Галерея"], summary="Сводка по галерее")
def list_gallery(limit: int = 100,
                 repository: VectorRepository = Depends(require_repository)) -> dict:
    return {"size": repository.count(), "vehicles": repository.list_vehicles(limit)}


@app.delete("/api/gallery/{image_id}", tags=["Галерея"], summary="Удалить наблюдение")
def delete(image_id: str, repository: VectorRepository = Depends(require_repository)) -> dict:
    if not repository.delete(image_id):
        raise HTTPException(status_code=404, detail="Наблюдение не найдено")
    return {"deleted": image_id, "gallery_size": repository.count()}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


def run() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
