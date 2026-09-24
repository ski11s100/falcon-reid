"""Сверка двух сдач на одном и том же публичном тесте, без разметки.

    python scripts/compare_submissions.py submission runs/fulldata/submission

Зачем. Итоговую модель учим на всех размеченных машинах: рецепт уже проверен
на машинах, которых модель не видела (docs/release_choice.json), а для
последнего шага проверочной выборки не остаётся. Точность здесь не измерить,
зато можно поймать поломку: обе сдачи отвечают на те же 1110 запросов, и
исправная модель, обученная тем же рецептом на чуть большем числе машин,
должна почти всегда соглашаться с прежней, отказывать примерно так же часто
и давать сходства той же шкалы — иначе прежний порог к ней не переносится.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rankings(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0]: row[1:] for row in csv.reader(handle) if row}


def best_confidence(path: Path) -> np.ndarray:
    """Уверенность лучшего кандидата по каждому принятому запросу.

    confidence — косинус исходных векторов, та же шкала, что у порога;
    в embeddings.npy лежат уже обогащённые векторы, по ним шкалу не сравнить.
    """
    best: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            best[row["query_id"]] = max(best.get(row["query_id"], -1.0), float(row["confidence"]))
    return np.array(list(best.values()))


def summary(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    scores = best_confidence(directory / "candidates.csv")
    return {
        "threshold": manifest["threshold"],
        "refusal_rate": manifest["refusal_rate"],
        "refused_queries": manifest["refused_queries"],
        "accepted_confidence_quantiles": {
            f"p{q}": round(float(np.percentile(scores, q)), 4) for q in (5, 25, 50, 75, 95)
        },
    }


def compare(current_dir: Path, candidate_dir: Path) -> dict:
    current = read_rankings(current_dir / "submission.csv")
    candidate = read_rankings(candidate_dir / "submission.csv")
    if current.keys() != candidate.keys():
        raise ValueError("в сдачах разные наборы запросов")

    same_top1 = sum(current[q][0] == candidate[q][0] for q in current)
    overlap = np.mean([len(set(current[q]) & set(candidate[q])) / len(current[q]) for q in current])
    return {
        "queries": len(current),
        "top1_agreement": round(same_top1 / len(current), 4),
        "top10_overlap": round(float(overlap), 4),
        "current": summary(current_dir),
        "candidate": summary(candidate_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сверка двух сдач без разметки")
    parser.add_argument("current", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(compare(args.current, args.candidate), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
