"""Формирование трёх файлов сдачи в точном формате организаторов.

Формат подтверждён ответами 20, 21, 22, 24, 27, 29:

  embeddings.npy   float32, shape (len(test_query) + len(test_gallery), D).
                   Сначала ВСЕ query в порядке строк файла, затем ВСЯ gallery.
                   Никакой дополнительной сортировки. L2-нормализация не
                   обязательна, но сдаём нормированные — так честнее и дешевле.

  submission.csv   query_id, gallery_id_1 ... gallery_id_10, БЕЗ ЗАГОЛОВКА —
                   так читает эталонный скрипт организаторов
                   (organizers/evaluate.py: «Формат submission.csv (без
                   заголовка)»). Заголовок он принял бы за запрос с именем
                   query_id и выдал предупреждение о десяти неизвестных id.
                   РОВНО десять кандидатов на каждый запрос, БЕЗ ИСКЛЮЧЕНИЙ,
                   включая те, где мы уверены в отказе. Отказ здесь никак не
                   выражается — это файл только про ранжирование.

  candidates.csv   query_id, gallery_id, confidence.
                   Отказ = ПОЛНОЕ ОТСУТСТВИЕ строк для этого query_id.
                   Не пустая строка, не спецзначение. confidence — произвольная
                   шкала, важна только монотонность по уверенности.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Observation
from .metrics import TOP_K, l2_normalize
from .rerank import build_gallery_index, rerank_query


# Состав сдачи: две модели CLIP ViT-B/16 (на конкурсных данных и после VeRi-776)
# и ResNet50-IBN-a после VeRi-776, все дообучены с закраской зоны номера
# (scripts/finetune_plate_erase.py), без отражения кадра. Затем все три
# дообучены ещё 15 эпох на 1445 машинах вместо 1156: в обучение отдана три
# четверти отложенных машин (--extra-train-share 0.75), а проверка идёт на
# оставшихся 96, которых не видела ни одна версия. На них mAP@10 ансамбля
# 0.8899 -> 0.9138, парный бутстрэп +0.024, 95% интервал [+0.008; +0.042]
# (runs/moredata-075/final/release_choice.json, docs/release_choice.json). Выбор по баллам
# точности, кандидатов и скорости — scripts/compare_ensembles.py, отчёт
# docs/ensemble_choice.json. Отражение удваивает вычисления, а даёт лишь
# +0.003 mAP@10: без него 143 кадра/с и 31 мс на RTX 3060 Laptop при порогах
# полного балла 100 кадров/с и 40 мс.
SUBMISSION_CHECKPOINTS = ("models/clip-ours.pt", "models/clip-veri-ours.pt",
                          "models/resnet-v2-veri.pt")
SUBMISSION_FLIP_TTA = False
# PCA-проекция склеенного вектора 3584 -> 256, подогнанная на обучающей части
# (scripts/fit_projection.py, falcon/extract.Projection).
SUBMISSION_PROJECTION = "models/projection.pt"

# Обогащение векторов перед ранжированием (falcon/submit.enrich_vectors).
# Галерея усредняется со своими ближайшими соседями (DBA), запрос — со своими
# соседями из галереи (alpha-QE). Связи «запрос — запрос» не используются:
# ответ 38 запрещает только их, а «запрос → галерея» и «галерея → галерея»
# разрешены, и выдача каждого запроса по-прежнему не зависит от остальных
# запросов (закреплено тестом).
SUBMISSION_DBA_K = 3
SUBMISSION_QE_K = 2
# Сосед подмешивается, только если сходство с ним не ниже этого значения:
# ниже 0.4 это заведомо другая машина. Страховка на случай галереи, где у
# машины мало снимков: тогда вектор просто не меняется.
SUBMISSION_NEIGHBOUR_MIN = 0.4
# Вес соседа — сходство в кубе (классический alpha-QE, alpha=3): дальний сосед
# почти не влияет, а близкий влияет почти как сам снимок. Проверено на
# галереях разного размера (500/750/1100/1541 снимков, 12 прогонов на размер):
# линейный вес даёт средний mAP@10 0.8004, куб — 0.8088, и при размере галереи
# закрытого теста (750) разрыв больше всего: 0.8042 против 0.8163.
SUBMISSION_EXPANSION_POWER = 3.0
# Сколько строк матрицы сходства обогащение считает за раз: память ограничена
# блоком, а не квадратом размера галереи.
MERGE_BLOCK = 2048

# Порог отказа этого ансамбля.
#
# Подобран на локальном сплите (890 запросов, 20% без пары — как в закрытом
# тесте) по баллу режима кандидатов 0.7·F1 + 0.3·TNR, посчитанному ЭТАЛОННЫМ
# скриптом организаторов (organizers/evaluate.py). Берётся не острый пик кривой,
# а максимум после сглаживания окном ±0.01: TNR считается всего по 182 запросам
# без пары, и точечный максимум шумит. Проверка: scripts/verify_official.py.
#
# Порог выбран не по одному прогону, а по среднему восьми
# (scripts/calibrate_threshold.py, отчёт docs/threshold_choice.json): четыре
# жеребьёвки отложенной выборки × две плотности галереи — наша (1541 снимок на
# 890 запросов) и такая же редкая, как в публичном тесте (750 снимков на 1110
# запросов). Обе поправки нужны: оптимум одной жеребьёвки подогнан под неё
# (0.595…0.71), а оптимум плотной галереи завышен относительно редкой (0.65
# против 0.565) — на редкой сходство с лучшим кандидатом ниже.
# На закрытом тесте будет ровно одна жеребьёвка и одна плотность, и какие —
# неизвестно. Шкала порога — ИСХОДНЫЙ косинус (до обогащения векторов).
#
# Для моделей, дообученных на трёх четвертях отложенных машин, порог
# калибруется тем же способом на оставшихся 96 машинах (--extra-train-share
# 0.75, прореженная галерея — 0.676 снимка на запрос, как в публичном тесте).
# У новых моделей шкала сходства выше: они увереннее, и при прежнем пороге
# 0.605 брали бы машины без пары за совпадение (TNR 0.73). Свой порог — 0.69,
# плато 0.66-0.70. На публичном тесте, где пара есть у каждого запроса, при нём
# отказов 3.5% (39 из 1110) против 5.4% у прежних моделей при их пороге.
CALIBRATED_THRESHOLD = 0.69


@dataclass
class SubmissionConfig:
    """Параметры сдачи.

    use_rerank выключен намеренно. На реалистичном локальном сплите лучшая
    конфигурация (k1=12, lambda=0.7) дала +0.0057 mAP@10, но парный бутстрэп по
    запросам даёт 95% интервал [-0.010; +0.022]: прирост неотличим от шума, а
    на предыдущем сплите та же конфигурация теряла 4.7%. Выигрыш сомнителен,
    потеря возможна — в сдаче метод не используется.

    Параметры по умолчанию — лучшие из проверенных, на случай ручного включения:
    прежние (k1=20, lambda=0.3) были худшими в сетке, mAP 0.487.
    """

    threshold: float | None = None
    dba_k: int = SUBMISSION_DBA_K
    qe_k: int = SUBMISSION_QE_K
    neighbour_min: float = SUBMISSION_NEIGHBOUR_MIN
    expansion_power: float = SUBMISSION_EXPANSION_POWER
    use_rerank: bool = False
    rerank_pool: int = 100
    k1: int = 12
    k2: int = 6
    lambda_value: float = 0.7
    candidates_per_query: int = 1


def enrich_vectors(query_vectors: np.ndarray, gallery_vectors: np.ndarray,
                   config: SubmissionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Обогащение векторов соседями: DBA для галереи, alpha-QE для запросов.

    Один снимок машины видит её с одной точки; средний вектор нескольких
    снимков описывает саму машину устойчивее, чем любой из них. Поэтому вектор
    объекта галереи усредняется (со взвешиванием по сходству) со своими
    ближайшими соседями по галерее, а вектор запроса — со своими ближайшими
    объектами уже обогащённой галереи.

    Вес соседа — его сходство в степени expansion_power (alpha-QE, alpha=3):
    чем дальше сосед, тем меньше он тянет вектор на себя.

    На отложенной выборке это самый дешёвый прирост из найденных: mAP@10
    0.7564 -> 0.7971 без какого-либо переобучения (бутстреп +0.040,
    95% интервал [+0.027; +0.053]). На галерее размера закрытого теста
    (750 снимков) прирост больше: 0.7956 -> 0.8163.

    Важно: выдача запроса зависит только от него самого и от галереи. Связей
    между запросами нет, поэтому потоковый протокол (ответ 38) соблюдён.
    """
    queries = l2_normalize(query_vectors)
    gallery = l2_normalize(gallery_vectors)

    def merge(base: np.ndarray, pool: np.ndarray, k: int, drop_self: bool) -> np.ndarray:
        if k <= 0 or len(pool) <= 1:
            return base
        k = min(k, len(pool) - 1 if drop_self else len(pool))
        # Сходства считаются блоками по MERGE_BLOCK строк: полная матрица
        # «галерея × галерея» на 50 000 снимков заняла бы 10 ГБ, а блок —
        # десятки мегабайт при любом размере галереи. Результат тот же.
        merged = np.empty_like(base)
        for start in range(0, len(base), MERGE_BLOCK):
            rows = slice(start, start + MERGE_BLOCK)
            similarity = base[rows] @ pool.T
            if drop_self:
                block = np.arange(similarity.shape[0])
                similarity[block, block + start] = -1.0
            neighbours = np.argsort(-similarity, axis=1)[:, :k]
            weights = np.take_along_axis(similarity, neighbours, axis=1)
            weights = np.where(weights >= config.neighbour_min,
                               np.clip(weights, 0.0, None) ** config.expansion_power, 0.0)
            merged[rows] = base[rows] + (pool[neighbours] * weights[:, :, None]).sum(axis=1)
        return l2_normalize(merged)

    gallery = merge(gallery, gallery, config.dba_k, drop_self=True)
    queries = merge(queries, gallery, config.qe_k, drop_self=False)
    return queries, gallery


def build_ranking(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    config: SubmissionConfig,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (индексы топ-10 для каждого запроса, их оценки уверенности)."""
    queries = l2_normalize(query_vectors)
    gallery = l2_normalize(gallery_vectors)
    if len(gallery) < TOP_K:
        raise ValueError(f"В галерее {len(gallery)} объектов, а сдавать нужно топ-{TOP_K}")

    # Порядок кандидатов определяется по обогащённым векторам, а уверенность —
    # всегда по исходному косинусу (см. ниже про шкалу порога).
    ranked_queries, ranked_gallery = enrich_vectors(queries, gallery, config)

    top_indices = np.zeros((len(queries), TOP_K), dtype=np.int64)
    top_scores = np.zeros((len(queries), TOP_K), dtype=np.float32)

    if config.use_rerank:
        # Индекс соседства галереи строится один раз: галерея статична (ответ 38).
        index = build_gallery_index(gallery, k=max(config.k1 + 10, 30))
        for i, vector in enumerate(queries):
            order, _ = rerank_query(
                vector, index,
                k1=config.k1, k2=config.k2, lambda_value=config.lambda_value,
                candidate_pool=config.rerank_pool,
            )
            top_indices[i] = order[:TOP_K]
            # Уверенность — ВСЕГДА исходная косинусная близость, даже когда
            # порядок определён переранжированием. Оценки re-ranking лежат в
            # другой шкале (смесь расстояний со знаком минус), и порог, найденный
            # калибровкой на косинусе, к ним неприменим: однажды это отклонило
            # все 1110 запросов и обнулило режим кандидатов.
            # Монотонность по уверенности — единственное требование к confidence
            # (ответ 26), и косинус ему удовлетворяет.
            top_scores[i] = gallery[top_indices[i]] @ vector
            if progress and i % 200 == 0:
                print(json.dumps({"reranked": i + 1, "total": len(queries)}), flush=True)
    else:
        for start in range(0, len(queries), 256):
            block = ranked_queries[start : start + 256] @ ranked_gallery.T
            plain = queries[start : start + 256] @ gallery.T
            for offset, scores in enumerate(block):
                order = np.argsort(-scores, kind="stable")[:TOP_K]
                top_indices[start + offset] = order
                # Уверенность — исходная косинусная близость: порог откалиброван
                # именно в этой шкале, а обогащение смещает сходства вверх.
                top_scores[start + offset] = plain[offset][order]

    return top_indices, top_scores


def write_submission(
    output_dir: Path,
    queries: list[Observation],
    gallery: list[Observation],
    embeddings: np.ndarray,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    config: SubmissionConfig,
    model_version: str = "unknown",
) -> dict:
    """Пишет три файла и манифест с контрольными суммами."""
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ids = [row.image_id for row in queries]
    gallery_ids = [row.image_id for row in gallery]

    if len(embeddings) != len(queries) + len(gallery):
        raise ValueError(
            f"embeddings.npy должен содержать {len(queries)} + {len(gallery)} = "
            f"{len(queries) + len(gallery)} строк, получено {len(embeddings)}"
        )
    if set(query_ids) & set(gallery_ids):
        raise ValueError("query_id и gallery_id пересекаются — это ошибка формата")

    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32, copy=False))

    with (output_dir / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for i, qid in enumerate(query_ids):
            writer.writerow([qid] + [gallery_ids[j] for j in top_indices[i]])

    accepted = 0
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(query_ids):
            if config.threshold is None:
                # Без порога честнее отказать во всём, чем выдать необоснованные пары.
                continue
            keep = [j for j in range(TOP_K) if float(top_scores[i][j]) >= config.threshold]
            keep = keep[: max(1, config.candidates_per_query)] if keep else []
            if keep:
                accepted += 1
            for j in keep:
                # Никаких строк для отклонённого запроса не пишем вообще (ответ 20).
                writer.writerow([qid, gallery_ids[top_indices[i][j]],
                                 format(float(top_scores[i][j]), ".9g")])

    manifest = {
        "format": "falcon-submission-v1",
        "model": model_version,
        "queries": len(queries),
        "gallery": len(gallery),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_dtype": str(embeddings.dtype),
        "threshold": config.threshold,
        "accepted_queries": accepted,
        "refused_queries": len(queries) - accepted,
        "refusal_rate": round((len(queries) - accepted) / max(1, len(queries)), 4),
        "vector_expansion": {
            "method": ("DBA по галерее + alpha-QE запроса по галерее; "
                       "связей запрос-запрос нет (ответ 38)"),
            "dba_k": config.dba_k,
            "qe_k": config.qe_k,
            "neighbour_min": config.neighbour_min,
            "weight": f"сходство^{config.expansion_power:g}",
            "affects": "только порядок кандидатов; confidence считается до обогащения",
        },
        "reranking": {
            "enabled": config.use_rerank,
            "method": "k-reciprocal, потоково-совместимый (без связей query-query)",
            "k1": config.k1, "k2": config.k2, "lambda": config.lambda_value,
            "pool": config.rerank_pool,
        },
        "confidence_definition": ("косинусное сходство исходных эмбеддингов (до обогащения), "
                                  "не калиброванная вероятность; обогащение и переранжирование "
                                  "меняют только порядок"),
        "refusal_encoding": "отсутствие строк для query_id в candidates.csv",
        "checksums": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("submission.csv", "candidates.csv", "embeddings.npy")
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_submission(output_dir: Path, expected_queries: int, expected_gallery: int) -> dict:
    """Самопроверка перед сдачей: расхождение формата сразу считается ошибкой."""
    output_dir = Path(output_dir)
    problems: list[str] = []

    embeddings = np.load(output_dir / "embeddings.npy")
    if embeddings.shape[0] != expected_queries + expected_gallery:
        problems.append(
            f"embeddings.npy: {embeddings.shape[0]} строк вместо "
            f"{expected_queries + expected_gallery}"
        )
    if embeddings.dtype != np.float32:
        problems.append(f"embeddings.npy: dtype {embeddings.dtype}, ожидается float32")
    if not np.isfinite(embeddings).all():
        problems.append("embeddings.npy: есть NaN или inf")

    with (output_dir / "submission.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        submission_ids = set()
        rows = 0
        for row in reader:
            rows += 1
            if rows == 1 and row and row[0] == "query_id":
                problems.append("submission.csv: заголовок не нужен, эталонный скрипт читает без него")
            if len(row) != TOP_K + 1:
                problems.append(f"submission.csv: строка {rows} содержит {len(row)} полей")
            if len(set(row[1:])) != TOP_K:
                problems.append(f"submission.csv: строка {rows} содержит повторяющиеся gallery_id")
            submission_ids.add(row[0])
        if rows != expected_queries:
            problems.append(f"submission.csv: {rows} строк вместо {expected_queries}")

    with (output_dir / "candidates.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["query_id", "gallery_id", "confidence"]:
            problems.append(f"candidates.csv: неверный заголовок {header}")
        answered = set()
        for row in reader:
            if len(row) != 3 or not row[1].strip() or not row[2].strip():
                problems.append(f"candidates.csv: пустые поля в строке {row} — отказ кодируется "
                                "отсутствием строки, а не пустыми значениями")
                continue
            answered.add(row[0])
        unknown = answered - submission_ids
        if unknown:
            problems.append(f"candidates.csv: query_id, которых нет в submission.csv: {sorted(unknown)[:3]}")

    return {
        "valid": not problems,
        "problems": problems,
        "queries": expected_queries,
        "gallery": expected_gallery,
        "answered_queries": len(answered),
        "refused_queries": expected_queries - len(answered),
    }
