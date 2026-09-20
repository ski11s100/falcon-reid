# ФАЛЬКОН — образ инференса.
#
# Ответ 39 организаторов разделяет два режима:
#   docker build — сеть разрешена, зависимости и веса можно скачивать;
#   docker run   — сети нет вообще, всё необходимое уже внутри образа.
# Поэтому веса копируются в образ на этапе сборки, а на старте ничего не
# скачивается: torch.hub при инференсе не вызывается.
#
# База — CUDA runtime, потому что замер производительности идёт на GPU
# (ответ 30: RTX A5000, CUDA 12.2, доступ в контейнер через --gpus all).
# Драйвер 12.2 совместим с рантаймом CUDA 12.6 благодаря minor version
# compatibility внутри мажорной версии 12.
#
# Ubuntu 24.04, а не 22.04: там штатный Python 3.12, тот же, на котором решение
# разрабатывалось и на котором измерены все метрики. В 22.04 пришлось бы тянуть
# python3.11 из стороннего репозитория, то есть расходиться с окружением
# разработки ради ничего.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Виртуальное окружение вместо установки в системный Python: свежие Ubuntu
# помечают системный интерпретатор как externally-managed и запрещают в него
# ставить пакеты (PEP 668). Venv заодно изолирует решение от системных
# библиотек и делает сборку воспроизводимой.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/falcon

# Зависимости ставятся до копирования кода: слои кешируются и не пересобираются
# при каждом изменении исходников. PyTorch (2.5 ГБ) — отдельным слоем: правка
# requirements.txt не должна заставлять качать его заново.
RUN pip install --upgrade pip \
    && pip install torch==2.14.0 torchvision==0.29.0 \
         --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY falcon ./falcon
COPY service ./service
COPY scripts ./scripts

# Веса внутри образа: три модели ансамбля. Суммарный размер весов ограничен
# 2 ГБ (раздел 7 ТЗ): две CLIP ViT-B/16 в fp16 (347 МБ) и ResNet50-IBN-a
# (52 МБ), все в fp16 — 399 МБ.
COPY models ./models

# По умолчанию — тот же ансамбль и порог, что в сдаче и в docker-compose:
# порог откалиброван под этот состав моделей и на одну модель не переносится.
ENV FALCON_CHECKPOINTS=/opt/falcon/models/clip-ours.pt,/opt/falcon/models/clip-veri-ours.pt,/opt/falcon/models/resnet-v2-veri.pt \
    FALCON_MATCH_THRESHOLD=0.605 \
    FALCON_PROJECTION=/opt/falcon/models/projection.pt \
    FALCON_FLIP_TTA=false \
    FALCON_DEVICE=cuda \
    FALCON_HOST=0.0.0.0 \
    FALCON_PORT=8000 \
    FALCON_SQLITE_PATH=/opt/falcon/data/falcon.sqlite3 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Непривилегированный пользователь для сервиса. docker-compose запускает api
# от него (user: 10001) с файловой системой только для чтения. Образ по
# умолчанию остаётся от root ради пакетной генерации сдачи: она пишет в
# каталог хоста, смонтированный в /data, а его владельца мы не знаем.
RUN useradd --system --uid 10001 --home-dir /tmp --shell /usr/sbin/nologin falcon \
    && mkdir -p /opt/falcon/data \
    && chown -R 10001:10001 /opt/falcon/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# По умолчанию поднимается демонстрационный сервис. Пакетная генерация файлов
# сдачи — отдельная команда, переопределяющая CMD:
#
#   docker run --rm --gpus all -v /путь/к/данным:/data falcon-api \
#       python scripts/run_submission.py /data --output /data/submission
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
