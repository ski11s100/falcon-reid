"""Решения оператора: сохраняются, сводятся и выгружаются без кадров и векторов.

Двойников модель не различает, человек — различает. Его решение по паре из
окна сравнения должно сохраниться с тем, что он видел (сходство, ответ сервиса,
порог), попасть в журнал обращений и в выгрузку для перекалибровки.
"""

import csv
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from service.app import app, require_audit, require_repository  # noqa: E402
from service.audit import AuditLog  # noqa: E402
from service.repository import GalleryItem, SQLiteRepository  # noqa: E402


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestFeedback(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.audit = AuditLog(Path(self.directory.name) / "audit.jsonl")
        self.repository = SQLiteRepository(":memory:")
        self.repository.initialise(3)
        self.repository.upsert([
            GalleryItem(image_id="van-a", vehicle_id="ТС-925", embedding=unit(1, 0, 0), metadata={}),
            GalleryItem(image_id="van-b", vehicle_id="ТС-1022", embedding=unit(0, 1, 0), metadata={}),
        ])
        app.dependency_overrides[require_repository] = lambda: self.repository
        app.dependency_overrides[require_audit] = lambda: self.audit
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.directory.cleanup()

    def send(self, image_id, same, score, verdict="требуется проверка", threshold=0.69):
        return self.client.post("/api/feedback", json={
            "candidate_image_id": image_id, "same": same, "score": score,
            "verdict": verdict, "threshold": threshold})

    def test_decision_is_stored_with_the_candidate_vehicle(self):
        response = self.send("van-b", True, 0.968)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["candidate_vehicle_id"], "ТС-1022")
        rows = self.repository.list_feedback()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["candidate_vehicle_id"], rows[0]["same"]), ("ТС-1022", True))

    def test_summary_counts_agreement_above_threshold(self):
        self.send("van-b", False, 0.968)          # двойник выше порога — оператор отклонил
        self.send("van-a", True, 0.965)           # своя машина выше порога — подтвердил
        self.send("van-a", False, 0.40, verdict="совпадений нет")   # ниже порога
        summary = self.client.get("/api/feedback").json()
        self.assertEqual((summary["total"], summary["confirmed"], summary["rejected"]), (3, 1, 2))
        self.assertEqual(summary["above_threshold_confirmed"], 1)
        self.assertEqual(summary["above_threshold_rejected"], 1)
        self.assertEqual(summary["precision_above_threshold"], 0.5)
        self.assertEqual(summary["by_verdict"]["требуется проверка"], {"подтверждено": 1, "отклонено": 1})

    def test_empty_summary_has_no_precision(self):
        summary = self.client.get("/api/feedback").json()
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["precision_above_threshold"])

    def test_unknown_candidate_is_rejected(self):
        self.assertEqual(self.send("нет-такого", True, 0.9).status_code, 404)
        self.assertEqual(self.repository.list_feedback(), [])

    def test_invalid_score_and_verdict_are_rejected(self):
        self.assertEqual(self.send("van-a", True, 1.5).status_code, 422)
        self.assertEqual(self.send("van-a", True, 0.9, verdict="наверное").status_code, 422)

    def test_export_is_csv_with_every_decision(self):
        self.send("van-a", True, 0.965)
        self.send("van-b", False, 0.968)
        response = self.client.get("/api/feedback/export.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        rows = list(csv.DictReader(io.StringIO(response.text)))
        self.assertEqual(len(rows), 2)
        self.assertEqual({(r["candidate_vehicle_id"], r["same"]) for r in rows},
                         {("ТС-925", "1"), ("ТС-1022", "0")})

    def test_audit_log_gets_the_decision_without_images(self):
        self.send("van-b", False, 0.968)
        entries = [json.loads(line) for line in
                   (Path(self.directory.name) / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
        entry = entries[-1]
        self.assertEqual((entry["action"], entry["same"], entry["candidate_vehicle_id"]),
                         ("feedback", False, "ТС-1022"))
        self.assertNotIn("image_base64", entry)

    def test_retention_removes_old_decisions_too(self):
        self.send("van-a", True, 0.965)
        self.repository.delete_older_than(time.time() + 1)
        self.assertEqual(self.repository.list_feedback(), [])


if __name__ == "__main__":
    unittest.main()
