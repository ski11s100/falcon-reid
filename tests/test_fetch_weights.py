"""Докачка весов вместо указателей Git LFS: скачанное сверяется с хешем указателя."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fetch_weights", ROOT / "scripts" / "fetch_weights.py")
fetch_weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_weights)


def pointer_text(payload: bytes) -> str:
    return ("version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
            f"size {len(payload)}\n")


class TestFetchWeights(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.models, self.mirror = root / "models", root / "mirror"
        self.models.mkdir()
        self.mirror.mkdir()
        self.payload = b"\x80\x02" + bytes(range(256)) * 40     # как будто веса

    def tearDown(self):
        self.directory.cleanup()

    def test_pointer_is_recognised_and_real_file_is_not(self):
        pointer = self.models / "model.pt"
        pointer.write_text(pointer_text(self.payload), encoding="utf-8")
        parsed = fetch_weights.read_pointer(pointer)
        self.assertEqual(parsed["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(parsed["size"], len(self.payload))
        real = self.models / "real.pt"
        real.write_bytes(self.payload)
        self.assertIsNone(fetch_weights.read_pointer(real))

    def test_download_replaces_pointer_when_hash_matches(self):
        target = self.models / "model.pt"
        target.write_text(pointer_text(self.payload), encoding="utf-8")
        (self.mirror / "model.pt").write_bytes(self.payload)
        pointer = fetch_weights.read_pointer(target)
        fetch_weights.download((self.mirror / "model.pt").as_uri(), target,
                               pointer["sha256"], pointer["size"])
        self.assertEqual(target.read_bytes(), self.payload)

    def test_tampered_file_is_rejected_and_pointer_kept(self):
        target = self.models / "model.pt"
        target.write_text(pointer_text(self.payload), encoding="utf-8")
        (self.mirror / "model.pt").write_bytes(self.payload + b"lishnee")
        pointer = fetch_weights.read_pointer(target)
        with self.assertRaises(RuntimeError):
            fetch_weights.download((self.mirror / "model.pt").as_uri(), target,
                                   pointer["sha256"], pointer["size"])
        self.assertIsNotNone(fetch_weights.read_pointer(target))
        self.assertFalse((self.models / "model.pt.part").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
