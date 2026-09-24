"""Локальный сплит: правила протокола и отдельная жеребьёвка отложенной части.

Порог отказа калибруется по нескольким жеребьёвкам (scripts/calibrate_threshold.py),
и для этого разбиение умеет менять деление отложенных машин, не трогая
обучающие: иначе модель увидела бы при обучении машины, на которых её потом
проверяют.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.data import Observation, build_local_split  # noqa: E402


def dataset(vehicles=40, cameras=3, shots=3):
    rows = []
    row_id = 0
    for vehicle in range(vehicles):
        for camera in range(cameras):
            for shot in range(shots):
                row_id += 1
                rows.append(Observation(row_id=row_id, image_id=f"v{vehicle}c{camera}s{shot}",
                                        path=Path(f"{row_id}.jpg"), bbox=(0, 0, 10, 10),
                                        vehicle_id=str(vehicle), camera_id=str(camera)))
    return rows


class TestLocalSplit(unittest.TestCase):
    def setUp(self):
        self.rows = dataset()
        self.base = build_local_split(self.rows, seed=42)

    def test_train_and_validation_identities_do_not_overlap(self):
        train = {r.vehicle_id for r in self.base.train}
        held_out = {r.vehicle_id for r in self.base.query} | {r.vehicle_id for r in self.base.gallery}
        self.assertFalse(train & held_out)

    def test_open_set_queries_are_absent_from_gallery(self):
        gallery = {r.vehicle_id for r in self.base.gallery}
        by_id = {r.image_id: r for r in self.base.query}
        for image_id in self.base.open_set_query_ids:
            self.assertNotIn(by_id[image_id].vehicle_id, gallery)

    def test_partition_seed_keeps_training_identities(self):
        other = build_local_split(self.rows, seed=42, partition_seed=7)
        self.assertEqual({r.vehicle_id for r in self.base.train},
                         {r.vehicle_id for r in other.train})

    def test_partition_seed_changes_the_draw(self):
        other = build_local_split(self.rows, seed=42, partition_seed=7)
        base_queries = {r.image_id for r in self.base.query}
        other_queries = {r.image_id for r in other.query}
        self.assertNotEqual(base_queries, other_queries)
        self.assertNotEqual(set(self.base.open_set_query_ids), set(other.open_set_query_ids))

    def test_extra_train_share_moves_identities_into_training(self):
        """Часть отложенных машин уходит в обучение, остальные — чистая проверка."""
        more = build_local_split(self.rows, seed=42, extra_train_share=0.5)
        base_train = {r.vehicle_id for r in self.base.train}
        more_train = {r.vehicle_id for r in more.train}
        held_out = {r.vehicle_id for r in more.query} | {r.vehicle_id for r in more.gallery}
        base_held = {r.vehicle_id for r in self.base.query} | {r.vehicle_id for r in self.base.gallery}
        self.assertTrue(base_train < more_train)          # обучение только выросло
        self.assertFalse(more_train & held_out)             # и не видит проверочных машин
        self.assertTrue(held_out <= base_held)              # проверка — из прежних отложенных

    def test_checked_identities_do_not_depend_on_share(self):
        """Проверочные машины берутся с конца списка: варианты сравнимы напрямую."""
        small = build_local_split(self.rows, seed=42, extra_train_share=0.75)
        smaller = build_local_split(self.rows, seed=42, extra_train_share=0.5)
        held = lambda s: {r.vehicle_id for r in s.query} | {r.vehicle_id for r in s.gallery}
        self.assertTrue(held(small) <= held(smaller))

    def test_full_share_trains_on_every_identity(self):
        """Итоговая модель на всех машинах: проверочной выборки нет совсем."""
        full = build_local_split(self.rows, seed=42, extra_train_share=1.0)
        self.assertEqual({r.vehicle_id for r in full.train}, {r.vehicle_id for r in self.rows})
        self.assertEqual(full.query, [])
        self.assertEqual(full.gallery, [])

    def test_default_behaviour_is_unchanged(self):
        again = build_local_split(self.rows, seed=42)
        self.assertEqual([r.image_id for r in self.base.query], [r.image_id for r in again.query])
        self.assertEqual([r.image_id for r in self.base.gallery], [r.image_id for r in again.gallery])


if __name__ == "__main__":
    unittest.main(verbosity=2)
