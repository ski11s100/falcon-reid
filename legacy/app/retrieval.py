"""Bounded-memory exact retrieval and metrics. No NxN similarity allocation."""
import numpy as np


def normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 2D embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Zero embedding")
    return matrix / norms


def ranked(matrix, rows, index, cross_camera=False):
    scores = matrix @ matrix[index]
    allowed = np.array([r.path != rows[index].path for r in rows])
    if cross_camera:
        if any(r.camera_id is None for r in rows):
            raise ValueError("camera_id missing; cross-camera metrics cannot be verified")
        allowed &= np.array([r.camera_id != rows[index].camera_id for r in rows])
    candidates = np.flatnonzero(allowed)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order, scores[order]


def evaluate(matrix, rows, cross_camera=False):
    matrix = normalize(matrix)
    if len(matrix) != len(rows) or any(r.vehicle_id is None for r in rows):
        raise ValueError("Embedding row count or identity labels invalid")
    aps, inp, r1, r5, failures = [], [], [], [], []
    for i, row in enumerate(rows):
        order, scores = ranked(matrix, rows, i, cross_camera)
        hits = np.array([rows[j].vehicle_id == row.vehicle_id for j in order])
        positions = np.flatnonzero(hits) + 1
        if not len(positions):
            continue
        aps.append(float(np.mean(np.arange(1, len(positions)+1) / positions)))
        inp.append(float(len(positions) / positions[-1]))
        r1.append(bool(hits[:1].any()))
        r5.append(bool(hits[:5].any()))
        if not r1[-1]:
            failures.append({"query_row": i, "query_image": row.image_id,
                             "candidate_row": int(order[0]), "score": float(scores[0]),
                             "first_correct_rank": int(positions[0])})
    if not aps:
        raise ValueError("No eligible positive pairs after exclusions")
    return {"mAP": float(np.mean(aps)), "Rank-1": float(np.mean(r1)),
            "Rank-5": float(np.mean(r5)), "mINP": float(np.mean(inp)),
            "eligible_queries": len(aps), "excluded_queries": len(rows)-len(aps),
            "protocol": "cross-camera" if cross_camera else "different-frame proxy (not official)",
            "errors": failures}


def refusal_metrics(scores, has_match, correct, threshold):
    scores, known, correct = np.asarray(scores), np.asarray(has_match, bool), np.asarray(correct, bool)
    if scores.shape != known.shape or scores.shape != correct.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration arrays")
    accept = scores >= threshold
    tp = int((accept & known & correct).sum())
    fp = int((accept & ~(known & correct)).sum())
    fn = int((known & ~(accept & correct)).sum())
    negatives = int((~known).sum())
    return {"threshold": float(threshold), "F1": 2*tp/max(1, 2*tp+fp+fn),
            "TNR": float((~accept & ~known).sum()/negatives) if negatives else None,
            "tp": tp, "fp": fp, "fn": fn, "unknown_queries": negatives,
            "definition": "top-1 identity accepted correctly; wrong known-ID acceptance counts FP and FN"}


def calibrate(scores, has_match, correct):
    if not len(scores) or not any(has_match) or all(has_match):
        raise ValueError("Calibration needs both known and unknown validation queries")
    thresholds = np.r_[np.unique(scores), np.nextafter(float(max(scores)), float('inf'))]
    results = [refusal_metrics(scores, has_match, correct, t) for t in thresholds]
    return max(results, key=lambda x: (x["F1"], x["TNR"], x["threshold"]))
