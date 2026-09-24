"""Вывод проверки номера в окне сравнения должен совпадать с числами рядом.

Оператор видит «закраска номера роняет сходство на 0.044, контрольных зон —
до 0.024» и вывод. Если вывод говорит «не опирается», а номер стоит почти
вдвое больше кузова, показ на защите выглядит как противоречие. Поэтому три
случая — не сильнее, в пределах погрешности, заметно сильнее — закреплены.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.explain import PLATE_CHECK_MARGIN, plate_check_verdict  # noqa: E402


class TestPlateCheckVerdict(unittest.TestCase):
    def test_plate_not_worse_than_body(self):
        relies, text = plate_check_verdict(0.010, 0.024)
        self.assertFalse(relies)
        self.assertIn("не опирается", text)

    def test_within_margin_is_named_as_such(self):
        relies, text = plate_check_verdict(0.044, 0.024)
        self.assertFalse(relies)
        self.assertIn("погрешности", text)
        self.assertNotIn("не опирается", text)

    def test_clearly_worse_is_flagged(self):
        relies, text = plate_check_verdict(0.024 + PLATE_CHECK_MARGIN + 0.01, 0.024)
        self.assertTrue(relies)
        self.assertIn("заметно сильнее", text)

    def test_numbers_are_in_the_text(self):
        _, text = plate_check_verdict(0.044, 0.024)
        self.assertIn("0.044", text)
        self.assertIn("0.024", text)


if __name__ == "__main__":
    unittest.main()
