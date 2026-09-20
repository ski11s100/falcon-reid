"""Правила смены сдачи: прирост должен быть уверенным и ничего не ломать.

Решение «переключить сдачу на новый ансамбль» дороже, чем кажется: меняются
порог, файлы сдачи и все отчёты. Поэтому правила вынесены в отдельную функцию и
проверяются здесь, а не держатся в голове в пять утра после ночного прогона.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("compare_release", ROOT / "scripts" / "compare_release.py")
compare_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_release)


def report(interval=(0.004, 0.02), current_candidate=0.8677, new_candidate=0.8677,
           latency=30.0, fps=131.0):
    return {
        "сдача": {"балл кандидатов": current_candidate},
        "кандидат": {"балл кандидатов": new_candidate,
                     "задержка batch=1, мс": latency,
                     "пропускная способность, кадр/с": fps},
        "бутстрэп": {"95% интервал": list(interval)},
    }


class TestVerdict(unittest.TestCase):
    def test_confident_gain_switches(self):
        decision, reasons = compare_release.verdict(report())
        self.assertEqual(decision, "менять сдачу")
        self.assertEqual(reasons, [])

    def test_interval_touching_zero_is_not_a_gain(self):
        decision, reasons = compare_release.verdict(report(interval=(-0.002, 0.03)))
        self.assertEqual(decision, "оставить как есть")
        self.assertIn("прирост mAP@10 неотличим от нуля", reasons)

    def test_small_candidate_drop_is_tolerated(self):
        """Балл кандидатов шумит на третьем знаке: 0.002 — не повод отказываться."""
        decision, _ = compare_release.verdict(report(new_candidate=0.8657))
        self.assertEqual(decision, "менять сдачу")

    def test_large_candidate_drop_blocks_the_switch(self):
        decision, reasons = compare_release.verdict(report(new_candidate=0.8477))
        self.assertEqual(decision, "оставить как есть")
        self.assertTrue(any("балл кандидатов" in reason for reason in reasons))

    def test_speed_limits_block_the_switch(self):
        decision, reasons = compare_release.verdict(report(latency=41.0, fps=95.0))
        self.assertEqual(decision, "оставить как есть")
        self.assertIn("задержка выше 40 мс", reasons)
        self.assertIn("пропускная способность ниже 100 кадров/с", reasons)

    def test_speed_is_not_checked_when_benchmark_skipped(self):
        decision, reasons = compare_release.verdict(report(latency=99.0, fps=10.0), with_speed=False)
        self.assertEqual(decision, "менять сдачу")
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
