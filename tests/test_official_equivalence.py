"""Наш протокол оценки обязан совпадать с эталонным скриптом организаторов.

organizers/evaluate.py — их скрипт без изменений. Здесь на случайных данных,
где есть всё трудное (junk той же камеры, open-set запросы, отказы, ничьи
по уверенности), сравниваются числа нашего falcon/metrics.py и их скрипта.
Если протокол у нас разойдётся с официальным, этот тест упадёт.

Нужен pandas (requirements-dev.txt): без него тест пропускается.
"""

import importlib.util
import io
import random
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.metrics import Identity, evaluate_ranking, evaluate_refusal  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(pd is None, "нужен pandas: pip install -r requirements-dev.txt")
class TestMatchesOrganizersScript(unittest.TestCase):
    def setUp(self):
        self.official = load_official()
        rng = random.Random(7)
        self.gallery = {}
        for g in range(120):
            self.gallery[f"g{g}"] = Identity(f"v{rng.randrange(40)}", f"c{rng.randrange(6)}")
        self.queries = {}
        for q in range(60):
            # Часть запросов — машины, которых нет в галерее (open-set).
            vehicle = f"v{rng.randrange(40)}" if q % 5 else f"new{q}"
            self.queries[f"q{q}"] = Identity(vehicle, f"c{rng.randrange(6)}")
        gids = list(self.gallery)
        self.ranking = {qid: rng.sample(gids, 10) for qid in self.queries}
        self.candidates = {qid: [(ranked[0], rng.choice([0.3, 0.5, 0.5, 0.9]))]
                           for qid, ranked in self.ranking.items() if rng.random() < 0.7}

        rows = [{"image_id": k, "vehicle_id": v.vehicle_id, "camera_id": v.camera_id, "split": "query"}
                for k, v in self.queries.items()]
        rows += [{"image_id": k, "vehicle_id": v.vehicle_id, "camera_id": v.camera_id, "split": "gallery"}
                 for k, v in self.gallery.items()]
        truth = pd.DataFrame(rows)
        self.q_df = truth[truth.split == "query"].set_index("image_id")
        self.g_df = truth[truth.split == "gallery"].set_index("image_id")

    def test_ranking_metrics(self):
        with redirect_stdout(io.StringIO()):
            theirs = self.official.ranking_metrics(self.q_df, self.g_df, self.ranking)
        ours = evaluate_ranking(self.ranking, self.queries, self.gallery)
        self.assertEqual(theirs["n_scored"], ours.scored_queries)
        self.assertAlmostEqual(theirs["mAP@10"], ours.mAP, places=12)
        self.assertAlmostEqual(theirs["Rank-1"], ours.rank_1, places=12)
        self.assertAlmostEqual(theirs["Rank-5"], ours.rank_5, places=12)

    def test_candidate_metrics(self):
        theirs = self.official.candidate_metrics(self.q_df, self.g_df, self.candidates)
        ours = evaluate_refusal(self.candidates, self.queries, self.gallery)
        self.assertEqual((theirs["TP"], theirs["FP"], theirs["FN"], theirs["TN"]),
                         (ours.tp, ours.fp, ours.fn, ours.tn))
        self.assertAlmostEqual(theirs["F1"], ours.f1, places=12)
        self.assertAlmostEqual(theirs["TNR"], ours.tnr, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
