"""Audit, embed, evaluate and export the challenge CSV format."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.dataset import read_manifest, audit_manifest, load_crop
from app.fingerprint import create_fingerprint, FINGERPRINT_VERSION
from app.retrieval import normalize, ranked, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["audit", "infer", "evaluate"])
    p.add_argument("csv", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cross-camera", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    rows = read_manifest(args.csv, args.images, require_labels=args.action == "evaluate")
    report = audit_manifest(rows)
    if report["broken"]:
        raise ValueError(f"Invalid images: {report['broken'][:3]}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.action == "audit":
        print(json.dumps(report)); return
    encoder, version = create_fingerprint, FINGERPRINT_VERSION
    if args.checkpoint:
        from app.neural import NeuralEncoder
        encoder = NeuralEncoder(args.checkpoint, args.device)
        version = encoder.version
    matrix = np.lib.format.open_memmap(args.output / "embeddings.npy", mode="w+", dtype="float32",
                                     shape=(len(rows), len(encoder(load_crop(rows[0])))))
    durations = []
    for i, row in enumerate(rows):
        start = time.perf_counter()
        matrix[i] = normalize(encoder(load_crop(row))[None])[0]
        durations.append((time.perf_counter()-start)*1000)
    matrix.flush()
    manifest = {"model": version, "rows": len(rows), "dimensions": matrix.shape[1],
                "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "row_order": [{"row_id": r.row_id, "image_id": r.image_id, "bbox": r.bbox} for r in rows],
                "threshold": args.threshold, "threshold_calibrated": False,
                "latency_ms": {"mean": float(np.mean(durations)), "p95": float(np.percentile(durations,95))},
                "export_status": "local schema; verify organizer sample before submission"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.action == "evaluate":
        metrics = evaluate(matrix, rows, args.cross_camera)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in metrics.items() if k != "errors"})); return
    with (args.output / "submission.csv").open("w", encoding="utf-8", newline="") as s, (args.output / "candidates.csv").open("w", encoding="utf-8", newline="") as c:
        submission, candidates = csv.writer(s), csv.writer(c)
        submission.writerow(["query_row_id", "image_id", "top10_row_ids"])
        candidates.writerow(["query_row_id", "candidate_row_ids", "scores", "accepted"])
        for i, row in enumerate(rows):
            order, scores = ranked(matrix, rows, i, args.cross_camera)
            order, scores = order[:10], scores[:10]
            submission.writerow([i, row.image_id, json.dumps(order.tolist())])
            accepted = bool(args.threshold is not None and len(scores) and scores[0] >= args.threshold)
            candidates.writerow([i, json.dumps(order.tolist() if accepted else []),
                                 json.dumps(scores.tolist() if accepted else []), accepted])
    print(json.dumps({"rows": len(rows), "model": version, "output": str(args.output)}))


if __name__ == "__main__":
    main()
