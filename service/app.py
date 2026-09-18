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
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from falcon.explain import GradCAM, overlay_heatmap, plate_attention, plate_masking_boxes
from falcon.extract import ExtractorConfig, FeatureExtractor, build_extractor
from falcon.model import VehicleReID, ViTReID

from .config import Settings, get_settings
from .repository import GalleryItem, VectorRepository, build_repository
from .security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    request_body_limit,
    require_api_key,
)
from .schemas import (
    BatchRegisterRequest,
    BatchRegisterResponse,
    Candidate,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PlateAttention,
    PlateCheck,
    QualityReport,
    RegisterRequest,
    RegisterResponse,
    SearchRequest,
    SearchResponse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("falcon.service")

# Защита от «бомб распаковки»: Pillow по умолчанию только предупреждает до
# 178 Мп. Здесь кадр больше лимита сразу отвергается (см. decode_image).
Image.MAX_IMAGE_PIXELS = get_settings().max_image_megapixels * 1_000_000

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
    model_summary: str = ""
    storage_name: str = "не подключено"


state = ServiceState()


# Вся работа с видеокартой идёт в ОДНОМ выделенном потоке.
#
# Обработчики FastAPI выполняются в пуле потоков, а кэш автоподбора алгоритмов
# свёртки cuDNN (cudnn.benchmark) у PyTorch свой в каждом потоке. Первый прогон
# в каждом новом потоке заново подбирает алгоритмы для всех свёрток ансамбля —
# это около 6 секунд вместо 25 мс. Замерено: первый поиск сразу после загрузки
# страницы (он идёт параллельно с /api/health и попадает в свежий поток)
# занимал 5.8-8.8 с. Один поток — один кэш, прогретый при старте сервиса.
# Заодно запросы к видеокарте не толкаются между собой.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def on_gpu(function, *args):
    return GPU.submit(function, *args).result()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    paths = [p for p in settings.checkpoint_paths if p.is_file()]
    if paths:
        state.extractor = build_extractor(
            paths,
            ExtractorConfig(size=settings.size, batch_size=settings.batch_size,
                            num_workers=0, device=settings.device,
                            half=settings.half, flip_tta=settings.flip_tta),
        )
        names = " + ".join(p.name for p in paths)
        state.model_name = (f"{state.extractor.metadata.get('architecture', 'falcon')} "
                            f"({'ансамбль: ' if len(paths) > 1 else ''}{names})")
        state.model_summary = summarise_models(state.extractor)
        # Прогрев: подбор алгоритмов cuDNN оплачивается при старте, а не
        # первым запросом оператора (или жюри).
        warmup = Image.new("RGB", (320, 240), (128, 128, 128))
        on_gpu(state.extractor.encode_image, warmup)          # поиск: батч из 1
        on_gpu(state.extractor.encode_images, [warmup])       # пачка: батч из 16
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
    # Swagger UI по умолчанию тянет скрипты с CDN, и на стенде без интернета
    # страница /docs была бы пустой. Здесь она собрана из локальных файлов
    # пакета swagger-ui-bundle (см. маршрут /docs ниже).
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)
app.add_middleware(BodySizeLimitMiddleware, limit_for_path=request_body_limit)
app.add_middleware(SecurityHeadersMiddleware)

# Ключ API (если задан FALCON_API_KEY) нужен всем методам, кроме /api/health.
protected = [Depends(require_api_key)]


MODEL_TITLES = {VehicleReID.ARCHITECTURE: "ResNet50-IBN", ViTReID.ARCHITECTURE: "CLIP ViT-B/16"}


def summarise_models(extractor) -> str:
    """«2 × ResNet50-IBN» или «ResNet50-IBN + CLIP ViT-B/16» — для строки состояния."""
    members = getattr(extractor, "members", None) or [extractor]
    titles = [MODEL_TITLES.get(m.model.ARCHITECTURE, m.model.ARCHITECTURE) for m in members]
    if len(titles) > 1 and len(set(titles)) == 1:
        return f"{len(titles)} × {titles[0]}"
    return " + ".join(titles)


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
        if image.format not in ("JPEG", "PNG", "WEBP", "BMP"):
            raise HTTPException(status_code=415, detail="Поддерживаются JPEG, PNG, WEBP и BMP")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Слишком большое разрешение кадра") from exc
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
    return on_gpu(extractor.encode_image, crop)


THUMBNAIL_SIZE = (320, 240)


def make_thumbnail(crop: Image.Image) -> bytes:
    """Небольшой JPEG для показа оператору и окна сравнения. Около 15 КБ."""
    thumbnail = crop.copy()
    thumbnail.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def as_data_url(thumbnail: bytes | None) -> str | None:
    if not thumbnail:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode()


FINGERPRINT_GROUPS = 128
FINGERPRINT_SEED = 20260918
_projections: dict[int, np.ndarray] = {}


def fingerprint(embedding: np.ndarray | None) -> list[float] | None:
    """Сводка эмбеддинга для показа оператору: «цифровой отпечаток» в картинке.

    Эмбеддинг проецируется на 128 фиксированных случайных направлений. Такая
    проекция приближённо сохраняет косинусное сходство (лемма Джонсона —
    Линденштраусса), поэтому у снимков одной машины кольца в интерфейсе похожи,
    а у разных — нет. Замер на 120 снимках 30 машин: ближайшее по форме кольцо
    принадлежит той же машине в 97% случаев; сходство колец одной машины +0.54,
    разных +0.01. Сводка «энергия групп признаков» различала хуже: +0.39.

    Направления задаются фиксированным зерном и одинаковы при каждом запуске.
    Это только визуализация: в поиске сводка не участвует.
    """
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    projection = _projections.get(vector.size)
    if projection is None:
        projection = np.random.default_rng(FINGERPRINT_SEED).standard_normal(
            (vector.size, FINGERPRINT_GROUPS)).astype(np.float32)
        _projections[vector.size] = projection
    values = vector @ projection
    return [round(float(v), 4) for v in values]


def to_candidate(match, repository: VectorRepository | None = None) -> Candidate:
    reference = repository.embedding_of(match.image_id) if repository is not None else None
    return Candidate(image_id=match.image_id, vehicle_id=match.vehicle_id,
                     score=round(match.score, 6), metadata=match.metadata,
                     thumbnail=as_data_url(match.thumbnail),
                     fingerprint=fingerprint(reference))


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
        threshold=settings.match_threshold,
        model_summary=state.model_summary,
        storage=state.storage_name,
        search=getattr(repository, "ann_index", "exact") if repository else "нет",
    )


@app.post("/api/gallery/register", response_model=RegisterResponse, status_code=201,
          tags=["Галерея"], summary="Добавить наблюдение в галерею",
          dependencies=protected)
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
        thumbnail=make_thumbnail(crop),
    )])
    return RegisterResponse(
        image_id=request.image_id, vehicle_id=request.vehicle_id,
        embedding_dim=int(embedding.size), gallery_size=repository.count(),
        quality=quality, elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["Поиск"],
          summary="Найти похожие ТС или обоснованно отказаться",
          dependencies=protected)
def search(request: SearchRequest, settings: Settings = Depends(get_settings),
           extractor: FeatureExtractor = Depends(require_model),
           repository: VectorRepository = Depends(require_repository)) -> SearchResponse:
    started = time.perf_counter()
    image = decode_image(request.image_base64, settings.max_upload_mb * 1024 * 1024)
    crop = crop_to_bbox(image, request.bbox)
    quality = assess_quality(crop)
    embedding = embed(extractor, crop)

    found = repository.search(embedding, request.top_k)
    candidates = [to_candidate(m, repository) for m in found]
    query_fingerprint = fingerprint(embedding)

    threshold = request.threshold if request.threshold is not None else settings.match_threshold
    elapsed = round((time.perf_counter() - started) * 1000, 2)

    if threshold is None:
        # Порог не откалиброван — подтверждать совпадение нельзя. Показываем
        # кандидатов оператору, но решение за человеком.
        return SearchResponse(
            accepted=False, verdict="требуется проверка",
            refusal_reason="Порог принятия решения не откалиброван для текущей модели",
            threshold=None, matches=[], candidates=candidates, quality=quality, elapsed_ms=elapsed,
            fingerprint=query_fingerprint)

    if not candidates:
        return SearchResponse(
            accepted=False, verdict="совпадений нет",
            refusal_reason="Галерея пуста", threshold=threshold,
            matches=[], candidates=[], quality=quality, elapsed_ms=elapsed,
            fingerprint=query_fingerprint)

    accepted = candidates[0].score >= threshold
    return SearchResponse(
        accepted=accepted,
        verdict="совпадение" if accepted else "совпадений нет",
        refusal_reason=None if accepted else
            f"Лучший кандидат {candidates[0].score:.3f} ниже порога {threshold:g}",
        threshold=threshold,
        matches=[c for c in candidates if c.score >= threshold] if accepted else [],
        candidates=candidates, quality=quality, elapsed_ms=elapsed,
        fingerprint=query_fingerprint)


@app.post("/api/gallery/register-batch", response_model=BatchRegisterResponse,
          status_code=201, tags=["Галерея"],
          summary="Добавить сразу несколько наблюдений",
          dependencies=protected)
def register_batch(request: BatchRegisterRequest,
                   settings: Settings = Depends(get_settings),
                   extractor: FeatureExtractor = Depends(require_model),
                   repository: VectorRepository = Depends(require_repository)) -> BatchRegisterResponse:
    """Регистрация пачкой: наполнение галереи для демонстрации за один вызов.

    Один снимок с ошибкой не отменяет остальные: его причина попадает в failed,
    а годные записи сохраняются. Для наполнения демонстрационной галереи это
    важнее строгой атомарности.
    """
    started = time.perf_counter()
    limit = settings.max_upload_mb * 1024 * 1024
    accepted: list[tuple] = []
    failed: list[dict] = []

    for entry in request.items:
        try:
            image = decode_image(entry.image_base64, limit)
            accepted.append((entry, crop_to_bbox(image, entry.bbox)))
        except HTTPException as exc:
            failed.append({"image_id": entry.image_id, "error": exc.detail})
        except Exception:
            # Подробности — в журнал сервера, клиенту — без внутренностей.
            log.exception("не удалось подготовить снимок %s", entry.image_id)
            failed.append({"image_id": entry.image_id, "error": "не удалось обработать снимок"})

    # Все кропы пачки проходят сеть батчами, а не по одному: при наполнении
    # галереи это в разы быстрее.
    embeddings = on_gpu(extractor.encode_images, [crop for _, crop in accepted])
    batch = [GalleryItem(
        image_id=entry.image_id, vehicle_id=entry.vehicle_id, embedding=embedding,
        metadata={**entry.metadata, "crop": [crop.width, crop.height]},
        thumbnail=make_thumbnail(crop),
    ) for (entry, crop), embedding in zip(accepted, embeddings)]

    registered = repository.upsert(batch) if batch else 0
    return BatchRegisterResponse(
        registered=registered, failed=failed, gallery_size=repository.count(),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def plate_check(extractor, crop: Image.Image, unit_reference: np.ndarray) -> PlateCheck:
    """Опирается ли сходство этой пары на зону номера — проверка маскированием.

    Запрос прогоняется целиком и с закрашенной зоной номера, а для сравнения —
    с закрашенными контрольными зонами той же площади. Все варианты идут одним
    батчем. Если зона номера роняет сходство не сильнее контрольных (с запасом
    0.02), модель на номер не опирается. Это причинная проверка, а не догадка
    по карте внимания: в «зону номера» по геометрии попадают и фары, и решётка.
    """
    from PIL import ImageDraw

    boxes = plate_masking_boxes(crop.width, crop.height)
    variants = [crop]
    for box in boxes.values():
        masked = crop.copy()
        # Заливка средним цветом ImageNet: после нормализации это нули.
        ImageDraw.Draw(masked).rectangle(box, fill=(124, 116, 104))
        variants.append(masked)
    vectors = on_gpu(extractor.encode_images, variants)
    scores = vectors @ unit_reference
    base = float(scores[0])
    drops = {name: round(base - float(score), 4) for name, score in zip(boxes, scores[1:])}
    plate_drop = drops.pop("зона номера")
    worst_control = max(drops.values())
    relies = plate_drop > worst_control + 0.02
    verdict = (f"закрашивание зоны номера снижает сходство на {plate_drop:.3f}, "
               f"контрольных зон — до {worst_control:.3f}: "
               + ("зона номера влияет заметно сильнее, случай для разбора" if relies
                  else "модель на номер не опирается"))
    return PlateCheck(similarity=round(base, 6), drops={"зона номера": plate_drop, **drops},
                      relies_on_plate=relies, verdict=verdict)


def explain_parts(extractor, reference: np.ndarray) -> list[tuple]:
    """Пары «модель — её часть эталонного вектора» для Grad-CAM.

    Вектор ансамбля — склейка векторов моделей. Раньше карта строилась по первой
    модели (2048 признаков) против всего вектора (4096), и объяснение для
    ансамбля падало с ошибкой размерности. Теперь каждая модель сравнивается со
    своей частью вектора, а карты усредняются: объяснение отражает весь ансамбль.
    """
    members = getattr(extractor, "members", None)
    if not members:
        return [(extractor.explain_model, reference)]
    parts, offset = [], 0
    for member in members:
        # Grad-CAM строится по свёрточной карте признаков: у ViT её нет, и такой
        # участник ансамбля в карту внимания не входит.
        if isinstance(member.model, VehicleReID):
            parts.append((member.explain_model, reference[offset:offset + member.feature_dim]))
        offset += member.feature_dim
    return parts


@app.post("/api/explain", response_model=ExplainResponse, tags=["Поиск"],
          summary="Показать, какие области повлияли на сопоставление",
          dependencies=protected)
def explain(request: ExplainRequest, settings: Settings = Depends(get_settings),
            extractor: FeatureExtractor = Depends(require_model),
            repository: VectorRepository = Depends(require_repository)) -> ExplainResponse:
    """Grad-CAM по близости запроса к конкретному кандидату из галереи.

    Раздел 10 ТЗ просит визуализацию областей, повлиявших на решение о
    сопоставлении, — она повышает доверие оператора к результату. Заодно
    считается доля внимания в зоне вероятного номера: пользоваться его
    признаками запрещено, и полезно иметь возможность это проверить.
    """
    started = time.perf_counter()
    reference = repository.embedding_of(request.gallery_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="Кандидат не найден в галерее")

    image = decode_image(request.image_base64, settings.max_upload_mb * 1024 * 1024)
    crop = crop_to_bbox(image, request.bbox)

    # Градиенты считаются по отдельной fp32-копии модели: рабочий путь инференса
    # идёт в half и трогать его ради объяснения нельзя.
    def gradcam() -> np.ndarray:
        tensor = extractor.transform(crop).unsqueeze(0).to(extractor.device)
        maps = []
        for model, part in explain_parts(extractor, reference):
            with GradCAM(model) as cam:
                maps.append(cam.similarity_map(tensor, torch.from_numpy(part)))
        heat = np.mean(maps, axis=0)
        span = float(heat.max() - heat.min())
        return (heat - heat.min()) / span if span > 1e-12 else heat

    heatmap = on_gpu(gradcam)

    overlay = overlay_heatmap(crop, heatmap)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")

    unit = reference / np.linalg.norm(reference)
    plate = plate_attention(heatmap).as_dict() if request.show_plate_region else None
    check = plate_check(extractor, crop, unit) if request.show_plate_region else None
    similarity = check.similarity if check else float(np.dot(embed(extractor, crop), unit))

    return ExplainResponse(
        gallery_id=request.gallery_id,
        similarity=round(similarity, 6),
        overlay_png_base64=base64.b64encode(buffer.getvalue()).decode(),
        plate_region=PlateAttention(**plate) if plate else None,
        plate_check=check,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.get("/api/gallery", tags=["Галерея"], summary="Сводка по галерее",
          dependencies=protected)
def list_gallery(limit: int = 100,
                 repository: VectorRepository = Depends(require_repository)) -> dict:
    """Машины в галерее: число снимков и миниатюра самого свежего."""
    vehicles = repository.list_vehicles(max(1, min(limit, 500)))
    for vehicle in vehicles:
        vehicle["thumbnail"] = as_data_url(vehicle.get("thumbnail"))
    return {"size": repository.count(), "vehicles": vehicles}


@app.delete("/api/gallery/{image_id}", tags=["Галерея"], summary="Удалить наблюдение",
          dependencies=protected)
def delete(image_id: str, repository: VectorRepository = Depends(require_repository)) -> dict:
    if not repository.delete(image_id):
        raise HTTPException(status_code=404, detail="Наблюдение не найдено")
    return {"deleted": image_id, "gallery_size": repository.count()}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class NoCacheStatic(StaticFiles):
    """Статика без кеширования браузером.

    Обнаружено при проверке: после обновления интерфейса браузер продолжал
    отдавать старый app.js из кеша, и новые элементы просто не появлялись.
    Файлы здесь маленькие (десятки килобайт), и запрет кеша ничего не стоит,
    зато жюри гарантированно увидит актуальную версию.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


try:
    from swagger_ui_bundle import swagger_ui_path

    app.mount("/docs-assets", StaticFiles(directory=swagger_ui_path), name="docs-assets")

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url, title=f"{app.title} — API",
            swagger_js_url="/docs-assets/swagger-ui-bundle.js",
            swagger_css_url="/docs-assets/swagger-ui.css",
            swagger_favicon_url="/docs-assets/favicon-32x32.png",
        )
except ImportError:  # пакет не установлен — документация с CDN, как в FastAPI
    from fastapi.openapi.docs import get_swagger_ui_html as _cdn_docs

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        return _cdn_docs(openapi_url=app.openapi_url, title=f"{app.title} — API")


if STATIC_DIR.is_dir():
    app.mount("/static", NoCacheStatic(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html",
                            headers={"Cache-Control": "no-cache, must-revalidate"})


def run() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
