"""Create a deterministic identity-disjoint dataset split manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_identities(root: Path, seed: int = 42) -> dict:
    identities = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(identities) < 3:
        raise ValueError("Для train/validation/test нужно минимум три vehicle_id")
    random.Random(seed).shuffle(identities)
    validation_size = max(1, round(len(identities) * 0.15))
    test_size = max(1, round(len(identities) * 0.15))
    while validation_size + test_size >= len(identities):
        if validation_size >= test_size and validation_size > 1:
            validation_size -= 1
        elif test_size > 1:
            test_size -= 1
        else:
            break
    train_size = len(identities) - validation_size - test_size
    return {
        "seed": seed,
        "strategy": "identity-disjoint",
        "train": identities[:train_size],
        "validation": identities[train_size:train_size + validation_size],
        "test": identities[train_size + validation_size:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Разбиение датасета без утечки vehicle_id")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = split_identities(args.dataset, seed=args.seed)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: len(values) for name, values in manifest.items() if isinstance(values, list)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

