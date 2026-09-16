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

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libjpeg-turbo8 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /opt/falcon

# Зависимости ставятся до копирования кода: слой кешируется и не пересобирается
# при каждом изменении исходников.
COPY requirements.txt .
RUN python -m pip install --upgrade pip==24.3.1 \
    && python -m pip install torch==2.14.0 torchvision==0.29.0 \
         --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install -r requirements.txt

COPY falcon ./falcon
COPY service ./service
COPY scripts ./scripts

# Веса модели внутри образа. Суммарный размер весов ограничен 2 ГБ (раздел 7 ТЗ);
# ResNet50-IBN-a занимает около 107 МБ.
COPY models ./models

ENV FALCON_CHECKPOINT=/opt/falcon/models/best.pt \
    FALCON_DEVICE=cuda \
    FALCON_HOST=0.0.0.0 \
    FALCON_PORT=8000 \
    FALCON_SQLITE_PATH=/opt/falcon/data/falcon.sqlite3

RUN mkdir -p /opt/falcon/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# По умолчанию поднимается сервис. Пакетная генерация файлов сдачи — отдельная
# команда, она переопределяет CMD (см. docs/SUBMISSION.md):
#   docker run --rm --gpus all -v /data:/data falcon \
#       python scripts/run_submission.py /data --output /data/submission
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
