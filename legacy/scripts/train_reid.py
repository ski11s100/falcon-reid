"""Train a compact ReID baseline with identity-balanced batches and held-out IDs."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.nn import functional as F
from app.dataset import read_manifest, grouped_identity_split, load_crop, audit_manifest
from app.neural import ReID, preprocessing, batch_hard_loss
from app.retrieval import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--identities-per-batch", type=int, default=4)
    p.add_argument("--samples-per-identity", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true", help="Explicitly download public ImageNet initialization")
    p.add_argument("--cross-camera", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.steps) < 1 or min(args.identities_per_batch, args.samples_per_identity) < 2:
        p.error("Need positive epochs/steps and at least two identities and samples per batch")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rows = read_manifest(args.csv, args.images, require_labels=True)
    audit = audit_manifest(rows)
    if audit["broken"]:
        raise ValueError(f"Broken input: {audit['broken'][:3]}")
    train, validation = grouped_identity_split(rows, seed=args.seed)
    labels = sorted({r.vehicle_id for r in train})
    groups = {label: [r for r in train if r.vehicle_id == label] for label in labels}
    eligible = [key for key, group in groups.items() if len({r.path for r in group}) >= 2]
    if len(eligible) < args.identities_per_batch:
        raise ValueError("Not enough train identities with at least two different frames")
    # No frame may cross the identity split, even when a full frame contains several vehicles.
    if {r.path for r in train} & {r.path for r in validation}:
        raise ValueError("Shared frames across train/validation; supply a grouped split before training")
    model = ReID(len(labels), pretrained=args.pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    augment, transform = preprocessing(True), preprocessing(False)
    history, best = [], -1
    split = {"train_ids": labels, "validation_ids": sorted({r.vehicle_id for r in validation}), "seed": args.seed}
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({'train_rows':len(train), 'validation_rows':len(validation), 'train_ids':len(labels), 'device':args.device}), flush=True)
    for epoch in range(args.epochs):
        model.train(); losses = []
        for step in range(args.steps):
            chosen = random.sample(eligible, args.identities_per_batch)
            batch = []
            for label in chosen:
                by_frame = list({r.path: r for r in groups[label]}.values())
                batch += random.sample(by_frame, min(len(by_frame), args.samples_per_identity))
            images = torch.stack([augment(load_crop(r)) for r in batch]).to(args.device)
            targets = torch.tensor([labels.index(r.vehicle_id) for r in batch], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            features, logits = model(images)
            loss = F.cross_entropy(logits, targets, label_smoothing=.1) + batch_hard_loss(features, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            losses.append(float(loss.detach()))
            if (step+1) % 20 == 0:
                print(json.dumps({'epoch':epoch+1,'step':step+1,'steps':args.steps,'loss':float(np.mean(losses))}),flush=True)
        model.eval()
        with torch.inference_mode():
            vectors = np.vstack([model(transform(load_crop(r)).unsqueeze(0).to(args.device))[0].cpu().numpy() for r in validation])
        metrics = evaluate(vectors, validation, cross_camera=args.cross_camera)
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if metrics["mAP"] > best:
            best = metrics["mAP"]
            torch.save({"architecture": "resnet18-reid-v1", "classes": len(labels),
                        "model": model.cpu().state_dict(), "epoch": epoch+1,
                        "initialization": "ImageNet1K_V1" if args.pretrained else "random",
                        "seed": args.seed, "validation_mAP": best}, args.output / "best.pt")
            model.to(args.device)
        print(json.dumps({k: record[k] for k in ("epoch", "loss", "mAP", "Rank-1")}), flush=True)


if __name__ == "__main__":
    main()
