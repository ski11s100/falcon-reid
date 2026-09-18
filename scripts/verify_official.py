"""Сверка с эталонным скриптом организаторов на локальной отложенной выборке.

    python scripts/verify_official.py D:/falcon-data

Что делает:
  1. строит эмбеддинги отложенной выборки ансамблем из сдачи;
  2. считает ranking-метрики нашим кодом (falcon/metrics.py) и эталонным
     скриптом (organizers/evaluate.py) — числа обязаны совпасть;
  3. считает режим кандидатов эталонным скриптом при выбранном пороге и для
     двух базовых стратегий («всегда отвечать», «всегда отказывать»);
  4. перебирает порог и находит максимум балла 0.7·F1 + 0.3·TNR после
     сглаживания окном ±0.01 — так выбран CALIBRATED_THRESHOLD.

Отчёт пишется в docs/official_check.json. Нужен pandas (requirements-dev.txt):
эталонный скрипт на нём написан.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import (  # noqa: E402
    Identity,
    evaluate_ranking,
    evaluate_refusal,
    l2_normalize,
    rank_from_embeddings,
)
from falcon.submit import CALIBRATED_THRESHOLD  # noqa: E402


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(metrics: dict) -> float:
    return 0.7 * metrics["F1"] + 0.3 * metrics["TNR"]


def rounded(metrics: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Сверка с эталонным скриптом организаторов")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[ROOT / "models" / "model-a.pt", ROOT / "models" / "model-b.pt"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "official_check.json")
    args = parser.parse_args()

    official = load_official()
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    extractor = build_extractor(args.checkpoints, ExtractorConfig(num_workers=8, threads=True))
    query = l2_normalize(extractor.extract(split.query, progress=False))
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))

    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    truth = pd.DataFrame([
        {"image_id": r.image_id, "vehicle_id": str(r.vehicle_id), "camera_id": str(r.camera_id), "split": part}
        for part, items in (("query", split.query), ("gallery", split.gallery)) for r in items])
    q_df = truth[truth.split == "query"].set_index("image_id")
    g_df = truth[truth.split == "gallery"].set_index("image_id")
    q_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    g_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}

    ranking = rank_from_embeddings(query, gallery, qids, gids)
    with redirect_stdout(io.StringIO()):
        official_ranking = official.ranking_metrics(q_df, g_df, ranking)
    ours = evaluate_ranking(ranking, q_labels, g_labels)

    scores = query @ gallery.T
    top = scores.argmax(1)
    best = scores[np.arange(len(qids)), top]

    def candidates(threshold: float) -> dict:
        return {qids[i]: [(gids[top[i]], float(best[i]))] for i in range(len(qids)) if best[i] >= threshold}

    chosen = candidates(CALIBRATED_THRESHOLD)
    official_chosen = official.candidate_metrics(q_df, g_df, chosen)
    ours_chosen = evaluate_refusal(chosen, q_labels, g_labels)
    always = official.candidate_metrics(q_df, g_df, candidates(-1.0))
    never = official.candidate_metrics(q_df, g_df, {})

    grid = np.round(np.arange(0.38, 0.58, 0.0025), 4)
    curve = np.array([score(official.candidate_metrics(q_df, g_df, candidates(t))) for t in grid])
    window = 9
    smooth = np.convolve(curve, np.ones(window) / window, mode="same")
    inner = slice(window // 2, len(grid) - window // 2)
    best_index = int(np.argmax(smooth[inner])) + window // 2

    report = {
        "ranking": {
            "эталонный скрипт": rounded(official_ranking),
            "наш код": {"mAP@10": round(ours.mAP, 4), "Rank-1": round(ours.rank_1, 4),
                        "Rank-5": round(ours.rank_5, 4)},
            "совпадает": abs(official_ranking["mAP@10"] - ours.mAP) < 1e-9,
        },
        "полное ранжирование (справочно)": rounded(
            official.full_ranking_metrics(query, gallery, qids, gids, q_df, g_df)),
        "режим кандидатов": {
            "порог": CALIBRATED_THRESHOLD,
            "эталонный скрипт": rounded(official_chosen),
            "балл 0.7·F1+0.3·TNR": round(score(official_chosen), 4),
            "наш код совпадает": abs(official_chosen["F1"] - ours_chosen.f1) < 1e-9
                                  and abs(official_chosen["TNR"] - ours_chosen.tnr) < 1e-9,
            "всегда отвечать": {"F1": round(always["F1"], 4), "TNR": round(always["TNR"], 4),
                                "балл": round(score(always), 4)},
            "всегда отказывать": {"F1": round(never["F1"], 4), "TNR": round(never["TNR"], 4),
                                  "балл": round(score(never), 4)},
        },
        "подбор порога": {
            "метод": "максимум балла после сглаживания окном ±0.01",
            "сглаженный максимум": float(grid[best_index]),
            "кривая": [{"порог": float(t), "балл": round(float(v), 4)} for t, v in zip(grid, curve)],
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "подбор порога"}
    summary["сглаженный максимум порога"] = float(grid[best_index])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
