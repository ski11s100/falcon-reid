"""Насколько точны наши числа: доверительные интервалы на отложенных машинах.

    python scripts/confidence_intervals.py <каталог данных> --extra-train-share 0.75

mAP@10 0.914 и балл режима кандидатов 0.930 посчитаны на 96 машинах, которых
модели не видели: 175 запросов с парой и 51 без пары. Это немного, и честный
ответ на вопрос «насколько этому верить» — интервал, а не одно число.

Интервал — бутстрэп по машинам, а не по запросам: снимки одной машины
похожи друг на друга, и если перетасовывать запросы по одному, интервал выйдет
обманчиво узким. Каждая из 2000 выборок берёт машины с возвращением вместе со
всеми их запросами.

Считается два режима:
  * пакетный — ровно как в сдаче (обогащение и запросов, и галереи);
  * живой сервис — без обогащения галереи (DBA): снимки в сервис добавляются
    по одному, и пересчитывать соседей всей галереи на каждую вставку нельзя.

Метрики на каждом запросе повторяют эталонный скрипт организаторов
(organizers/evaluate.py), и среднее по всем запросам сверяется с ним.
Отчёт — docs/confidence_intervals.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import TOP_K, l2_normalize, rank_from_embeddings  # noqa: E402
from falcon.submit import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    SUBMISSION_CHECKPOINTS,
    SUBMISSION_FLIP_TTA,
    SUBMISSION_PROJECTION,
    SubmissionConfig,
    enrich_vectors,
)

RESAMPLES = 2000


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_query(split, ranking: dict, top: dict, threshold: float) -> list[dict]:
    """AP@10 и исход режима кандидатов для каждого запроса — по правилам эталона."""
    gallery = {r.image_id: r for r in split.gallery}
    out = []
    for q in split.query:
        # Junk — та же машина с той же камеры: убирается до усечения до 10.
        clean = [g for g in ranking[q.image_id]
                 if not (gallery[g].vehicle_id == q.vehicle_id and gallery[g].camera_id == q.camera_id)]
        n_pos = sum(1 for g in split.gallery
                    if g.vehicle_id == q.vehicle_id and g.camera_id != q.camera_id)
        record = {"vehicle": q.vehicle_id, "has_pair": n_pos > 0, "ap": None}
        if n_pos:
            rel = np.array([gallery[g].vehicle_id == q.vehicle_id for g in clean[:TOP_K]], dtype=float)
            precision = np.cumsum(rel) / (np.arange(len(rel)) + 1)
            record["ap"] = float((precision * rel).sum() / min(n_pos, TOP_K))
        gid, score = top[q.image_id]
        answered = score >= threshold
        correct = gallery[gid].vehicle_id == q.vehicle_id
        record["outcome"] = ("TP" if n_pos and correct else "FP") if answered else ("FN" if n_pos else "TN")
        out.append(record)
    return out


def summarise(records: list[dict]) -> dict:
    aps = [r["ap"] for r in records if r["ap"] is not None]
    counts = defaultdict(int)
    fp_openset = 0
    for r in records:
        counts[r["outcome"]] += 1
        fp_openset += r["outcome"] == "FP" and not r["has_pair"]
    tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp_openset) if tn + fp_openset else float("nan")
    return {"mAP@10": float(np.mean(aps)), "F1": f1, "TNR": tnr, "балл кандидатов": 0.7 * f1 + 0.3 * tnr}


def bootstrap(records: list[dict], seed: int = 20260926) -> dict:
    by_vehicle = defaultdict(list)
    for r in records:
        by_vehicle[r["vehicle"]].append(r)
    vehicles = sorted(by_vehicle)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(RESAMPLES):
        picked = rng.choice(len(vehicles), size=len(vehicles), replace=True)
        sample = [r for i in picked for r in by_vehicle[vehicles[i]]]
        for key, value in summarise(sample).items():
            samples[key].append(value)
    point = summarise(records)
    return {key: {"значение": round(point[key], 4),
                  "95% интервал": [round(float(np.nanpercentile(v, 2.5)), 4),
                                   round(float(np.nanpercentile(v, 97.5)), 4)]}
            for key, v in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Доверительные интервалы метрик")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--extra-train-share", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "confidence_intervals.json")
    args = parser.parse_args()

    official = load_official()
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42, extra_train_share=args.extra_train_share)
    extractor = build_extractor([ROOT / p for p in SUBMISSION_CHECKPOINTS],
                                ExtractorConfig(num_workers=4, threads=True, flip_tta=SUBMISSION_FLIP_TTA),
                                projection=ROOT / SUBMISSION_PROJECTION)
    query = l2_normalize(extractor.extract(split.query, progress=False))
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    truth = pd.DataFrame([
        {"image_id": r.image_id, "vehicle_id": str(r.vehicle_id),
         "camera_id": str(r.camera_id), "split": part}
        for part, items in (("query", split.query), ("gallery", split.gallery)) for r in items])
    q_df = truth[truth.split == "query"].set_index("image_id")
    g_df = truth[truth.split == "gallery"].set_index("image_id")

    report = {
        "выборка": f"{len(split.query)} запросов, {len({r.vehicle_id for r in split.query})} машин вне "
                   f"обучения; бутстрэп по машинам, {RESAMPLES} выборок",
        "порог": CALIBRATED_THRESHOLD,
    }
    cosine = query @ gallery.T
    for name, config in (("пакетный режим (сдача)", SubmissionConfig()),
                         ("живой сервис (без DBA)", SubmissionConfig(dba_k=0))):
        ranked_query, ranked_gallery = enrich_vectors(query, gallery, config)
        ranking = rank_from_embeddings(ranked_query, ranked_gallery, qids, gids)
        order = (ranked_query @ ranked_gallery.T).argmax(1)
        # Порядок — по обогащённым векторам, уверенность — исходный косинус, как в сдаче.
        top = {qids[i]: (gids[order[i]], float(cosine[i, order[i]])) for i in range(len(qids))}
        records = per_query(split, ranking, top, CALIBRATED_THRESHOLD)
        with redirect_stdout(io.StringIO()):
            reference = official.ranking_metrics(q_df, g_df, ranking)
        mine = summarise(records)["mAP@10"]
        assert abs(reference["mAP@10"] - mine) < 1e-9, (reference["mAP@10"], mine)
        report[name] = bootstrap(records)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
