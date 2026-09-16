"""Challenge CSV reader. One row is one observation, even in shared frames."""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from .fingerprint import crop_bbox


@dataclass(frozen=True)
class Observation:
    row_id: int
    image_id: str
    path: Path
    bbox: tuple[float, float, float, float]
    vehicle_id: str | None
    camera_id: str | None


def read_manifest(csv_path, images_dir, require_labels=False):
    root = Path(images_dir).resolve()
    result = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        w, h = ("w" if "w" in fields else "width"), ("h" if "h" in fields else "height")
        required = {"image_id", "x", "y", w, h}
        if require_labels:
            required.add("vehicle_id")
        if required - fields:
            raise ValueError(f"Missing CSV columns: {sorted(required - fields)}")
        for index, row in enumerate(reader):
            name = (row["image_id"] or "").strip()
            path = (root / name).resolve()
            if not name or not path.is_relative_to(root):
                raise ValueError(f"Row {index}: invalid image path")
            if not path.is_file() and not path.suffix:
                options = [path.with_suffix(s) for s in (".jpg", ".jpeg", ".png") if path.with_suffix(s).is_file()]
                if len(options) == 1:
                    path = options[0]
            if not path.is_file():
                raise ValueError(f"Row {index}: missing image {name}")
            box = tuple(float(row[k]) for k in ("x", "y", w, h))
            if not all(math.isfinite(v) for v in box) or min(box[2:]) <= 1:
                raise ValueError(f"Row {index}: invalid bbox")
            label = (row.get("vehicle_id") or "").strip() or None
            if require_labels and label is None:
                raise ValueError(f"Row {index}: missing vehicle_id")
            result.append(Observation(index, name, path, box, label, row.get("camera_id") or None))
    if not result:
        raise ValueError("Empty manifest")
    return result


def load_crop(observation):
    # BBox coordinates refer to the encoded frame; do not rotate before cropping.
    with Image.open(observation.path) as image:
        image.load()
        return crop_bbox(image.convert("RGB"), observation.bbox)


def audit_manifest(rows):
    issues, clipped, hashes = [], [], {}
    for row in rows:
        try:
            with Image.open(row.path) as image:
                image.load()
                x, y, w, h = row.bbox
                if x < 0 or y < 0 or x + w > image.width or y + h > image.height:
                    clipped.append(row.row_id)
                crop_bbox(image, row.bbox)
            if row.path not in hashes:
                hashes[row.path] = hashlib.sha256(row.path.read_bytes()).hexdigest()
        except Exception as exc:
            issues.append({"row_id": row.row_id, "error": str(exc)})
    groups = {}
    for path, digest in hashes.items():
        groups.setdefault(digest, []).append(path.name)
    counts = Counter(r.vehicle_id for r in rows if r.vehicle_id is not None)
    return {"rows": len(rows), "frames": len({r.path for r in rows}),
            "identities": len(counts), "singletons": sum(v == 1 for v in counts.values()),
            "clipped_bbox_rows": clipped, "broken": issues,
            "duplicate_frames": [v for v in groups.values() if len(v) > 1],
            "cross_camera_verifiable": all(r.camera_id is not None for r in rows)}


def identity_split(rows, fraction=0.2, seed=42):
    import random
    identities = {r.vehicle_id for r in rows}
    if None in identities or len(identities) < 2 or not 0 < fraction < 1:
        raise ValueError("Need labelled rows, at least two identities and fraction in (0,1)")
    ids = sorted(identities)
    random.Random(seed).shuffle(ids)
    held = set(ids[:max(1, min(len(ids)-1, round(len(ids)*fraction)))])
    return [r for r in rows if r.vehicle_id not in held], [r for r in rows if r.vehicle_id in held]


def grouped_identity_split(rows, fraction=0.2, seed=42):
    """Keep identities linked by identical full frames together to avoid leakage."""
    import random
    identities = {r.vehicle_id for r in rows}
    if None in identities or len(identities)<2 or not 0<fraction<1:
        raise ValueError('Invalid grouped split input')
    parents = {key:key for key in identities}
    def find(key):
        while parents[key]!=key:
            parents[key]=parents[parents[key]]
            key=parents[key]
        return key
    owners, digests = {}, {}
    for row in rows:
        if row.path not in digests:
            digests[row.path]=hashlib.sha256(row.path.read_bytes()).hexdigest()
        digest=digests[row.path]
        if digest in owners:
            parents[find(row.vehicle_id)]=find(owners[digest])
        else:
            owners[digest]=row.vehicle_id
    components={}
    for key in sorted(identities):
        components.setdefault(find(key),[]).append(key)
    groups=sorted(components.values())
    if len(groups)<2:
        raise ValueError('All identities connected by shared frames; cannot split independently')
    random.Random(seed).shuffle(groups)
    held=set()
    target=max(1,round(len(identities)*fraction))
    for group in groups[:-1]:
        held.update(group)
        if len(held)>=target:
            break
    return [r for r in rows if r.vehicle_id not in held],[r for r in rows if r.vehicle_id in held]
