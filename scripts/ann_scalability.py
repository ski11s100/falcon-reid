"""Демонстрация масштабируемости поиска до галереи порядка 10^6 объектов.

Раздел 10 ТЗ просит показать работу приближённого поиска ближайших соседей
(ANN) на галерее около миллиона объектов, чтобы доказать готовность решения к
реальным городским масштабам без деградации скорости.

Что здесь измеряется:
  * время построения индекса;
  * задержка одного запроса (медиана и p95);
  * Recall@10 относительно точного перебора — сколько правильных соседей
    приближённый поиск теряет ради скорости;
  * объём памяти под векторы.

Векторы генерируются не случайным шумом, а кластерами: C идентичностей, у каждой
несколько снимков с разбросом. На чистом гауссовом шуме в высокой размерности все
векторы почти ортогональны, Recall получается вырожденным и ничего не значит.

HNSW выбран потому, что именно этот алгоритм использует pgvector в рабочем
контуре (см. service/repository.py), — демонстрация меряет ту же технологию,
что стоит в поставке, а не постороннюю.

Запуск:
    python scripts/ann_scalability.py --sizes 10000 100000 1000000 --dim 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def perturb(vectors: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Добавляет шум заданной НОРМЫ, а не покомпонентного СКО.

    В высокой размерности это принципиально: шум с покомпонентным СКО s имеет
    норму s*sqrt(dim), и при dim=256 значение 0.35 даёт возмущение нормой 5.6
    против единичной нормы самого вектора. Сигнал тонет в шуме, кластеры
    исчезают, и любые замеры Recall начинают мерить случайность.
    """
    dim = vectors.shape[-1]
    noise = rng.normal(size=vectors.shape).astype(np.float32) * (strength / np.sqrt(dim))
    result = vectors + noise
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def synthetic_gallery(size: int, dim: int, shots_per_identity: int = 12,
                      spread: float = 0.45, seed: int = 42) -> np.ndarray:
    """Кластеризованные L2-нормированные векторы, похожие на реальные эмбеддинги."""
    rng = np.random.default_rng(seed)
    identities = max(1, size // shots_per_identity)
    centres = rng.normal(size=(identities, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    assignment = rng.integers(0, identities, size=size)
    vectors = perturb(centres[assignment], spread, rng)
    return np.ascontiguousarray(vectors, dtype=np.float32)


def exact_search(gallery: np.ndarray, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, float]:
    """Точный перебор блоками — эталон, относительно которого считается Recall."""
    started = time.perf_counter()
    result = np.zeros((len(queries), top_k), dtype=np.int64)
    for start in range(0, len(queries), 64):
        block = queries[start : start + 64] @ gallery.T
        result[start : start + 64] = np.argpartition(-block, top_k, axis=1)[:, :top_k]
        for row in range(len(block)):
            order = result[start + row]
            result[start + row] = order[np.argsort(-block[row][order], kind="stable")]
    return result, (time.perf_counter() - started) * 1000 / len(queries)


def measure(index, queries: np.ndarray, top_k: int, repeats: int = 200) -> dict:
    """Задержка одного запроса: так работает потоковый сценарий из ответа 38."""
    timings: list[float] = []
    for i in range(min(repeats, len(queries))):
        single = queries[i : i + 1]
        started = time.perf_counter()
        index.search(single, top_k)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "latency_ms_median": round(statistics.median(timings), 4),
        "latency_ms_p95": round(float(np.percentile(timings, 95)), 4),
    }


def recall_at_k(approximate: np.ndarray, exact: np.ndarray) -> float:
    """Доля истинных ближайших соседей, найденных приближённым поиском."""
    hits = sum(len(set(a.tolist()) & set(e.tolist())) for a, e in zip(approximate, exact))
    return hits / (len(exact) * exact.shape[1])


def run_scale(size: int, dim: int, top_k: int, queries_count: int, seed: int) -> dict:
    import faiss

    print(json.dumps({"stage": "generating", "size": size, "dim": dim}), flush=True)
    gallery = synthetic_gallery(size, dim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    picked = rng.choice(size, size=queries_count, replace=False)
    # Запросы — возмущённые снимки из галереи: имитация того же ТС с другой камеры.
    queries = np.ascontiguousarray(perturb(gallery[picked], 0.3, rng), dtype=np.float32)

    memory_mb = gallery.nbytes / 1e6
    print(json.dumps({"stage": "exact baseline", "vectors_mb": round(memory_mb, 1)}), flush=True)
    exact_ids, exact_ms = exact_search(gallery, queries, top_k)

    report: dict = {
        "gallery_size": size,
        "embedding_dim": dim,
        "vectors_memory_mb": round(memory_mb, 1),
        "exact": {"latency_ms_median": round(exact_ms, 4), "recall@10": 1.0},
    }

    # Метрика L2, а не inner product. Для L2-нормированных векторов это
    # эквивалентно косинусу по ранжированию, потому что ||a-b||^2 = 2 - 2*(a,b),
    # но граф HNSW строится в метрическом пространстве и с inner product даёт
    # заметно худший Recall: на 100 тысячах объектов падение было до 0.14.
    print(json.dumps({"stage": "building hnsw"}), flush=True)
    started = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(gallery)
    build_seconds = time.perf_counter() - started

    hnsw_report: dict = {"build_seconds": round(build_seconds, 1), "m": 32, "ef_construction": 64}
    for ef_search in (32, 64, 128):
        hnsw.hnsw.efSearch = ef_search
        stats = measure(hnsw, queries, top_k)
        _, approx = hnsw.search(queries, top_k)
        stats["recall@10"] = round(recall_at_k(approx, exact_ids), 4)
        hnsw_report[f"efSearch={ef_search}"] = stats
        print(json.dumps({"hnsw": ef_search, **stats}), flush=True)
    report["hnsw"] = hnsw_report

    del hnsw
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Масштабируемость поиска ФАЛЬКОН")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--dim", type=int, default=256,
                        help="Размерность эмбеддинга. На 10^6 объектов 2048 измерений "
                             "требуют 8 ГБ только под векторы")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("docs/ann_scalability.json"))
    args = parser.parse_args()

    results = [run_scale(size, args.dim, args.top_k, args.queries, args.seed)
               for size in args.sizes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
