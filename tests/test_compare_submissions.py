"""Сверка сдач без разметки: по ней решается судьба модели на всех машинах.

Модель, обученная на всех размеченных машинах, проверяется только сверкой с
прежней сдачей на публичном тесте. Ошибка в подсчёте совпадений или шкалы
уверенности здесь незаметно пропустила бы сломанную модель, поэтому подсчёт
закреплён на маленьких сдачах, где ответ известен заранее.
"""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("compare_submissions",
                                              ROOT / "scripts" / "compare_submissions.py")
compare_submissions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_submissions)


def write_submission(folder: Path, rankings: dict[str, list[str]],
                     candidates: list[tuple[str, str, float]], threshold: float = 0.69) -> Path:
    folder.mkdir(parents=True)
    with (folder / "submission.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([query, *ranking] for query, ranking in rankings.items())
    with (folder / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        writer.writerows(candidates)
    accepted = len({query for query, _, _ in candidates})
    refused = len(rankings) - accepted
    (folder / "manifest.json").write_text(json.dumps({
        "threshold": threshold, "refused_queries": refused,
        "refusal_rate": round(refused / len(rankings), 4),
    }), encoding="utf-8")
    return folder


class TestCompareSubmissions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_agreement_and_overlap(self):
        current = write_submission(self.root / "a", {
            "q1": ["g1", "g2", "g3", "g4"],
            "q2": ["g5", "g6", "g7", "g8"],
        }, [("q1", "g1", 0.9), ("q2", "g5", 0.8)])
        candidate = write_submission(self.root / "b", {
            "q1": ["g1", "g3", "g2", "g9"],      # тот же первый, 3 из 4 общие
            "q2": ["g6", "g5", "g7", "g8"],      # первый другой, все 4 общие
        }, [("q1", "g1", 0.95)])
        report = compare_submissions.compare(current, candidate)
        self.assertEqual(report["top1_agreement"], 0.5)
        self.assertEqual(report["top10_overlap"], 0.875)
        self.assertEqual(report["current"]["refused_queries"], 0)
        self.assertEqual(report["candidate"]["refusal_rate"], 0.5)

    def test_confidence_scale_takes_best_candidate_per_query(self):
        folder = write_submission(self.root / "a", {"q1": ["g1"], "q2": ["g2"]}, [
            ("q1", "g1", 0.9), ("q1", "g2", 0.7), ("q2", "g2", 0.8), ("q2", "g1", 0.6),
        ])
        quantiles = compare_submissions.summary(folder)["accepted_confidence_quantiles"]
        # Лучшие по запросам — 0.9 и 0.8; вторые кандидаты шкалу не сдвигают.
        self.assertAlmostEqual(quantiles["p50"], 0.85)
        self.assertLessEqual(quantiles["p95"], 0.9)

    def test_different_query_sets_are_rejected(self):
        current = write_submission(self.root / "a", {"q1": ["g1"]}, [("q1", "g1", 0.9)])
        candidate = write_submission(self.root / "b", {"q2": ["g1"]}, [("q2", "g1", 0.9)])
        with self.assertRaises(ValueError):
            compare_submissions.compare(current, candidate)


if __name__ == "__main__":
    unittest.main()
