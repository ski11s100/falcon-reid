"""Conservative metric-only adaptation with frozen BN and baseline checkpoint selection."""
import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from app.dataset import read_manifest,load_crop
from app.neural import ReID,preprocessing,batch_hard_loss
from app.retrieval import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',type=Path)
    p.add_argument('split',type=Path,help='Previously audited grouped identity split')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.steps<1 or args.rounds<1 or args.lr<=0:p.error('Invalid training configuration')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    split=json.loads(args.split.read_text())
    train_ids,val_ids=set(split['train_ids']),set(split['validation_ids'])
    if train_ids&val_ids:raise ValueError('Identity leakage')
    rows=read_manifest(args.dataset/'train.csv',args.dataset/'images',True)
    if {r.vehicle_id for r in rows}!=train_ids|val_ids:raise ValueError('Split does not match dataset identities')
    train=[r for r in rows if r.vehicle_id in train_ids]
    val=[r for r in rows if r.vehicle_id in val_ids]
    groups={}
    for row in train:groups.setdefault(row.vehicle_id,{}).setdefault(row.camera_id,[]).append(row)
    eligible=sorted(k for k,v in groups.items() if len(v)>=2 and None not in v)
    if len(eligible)<4:raise ValueError('Need four identities with multiple cameras')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'split.json').write_text(json.dumps(split,indent=2))
    # Initial classification head is unused; all trainable signal is cross-camera triplet loss.
    model=ReID(1,pretrained=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.backbone.layer4.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    transform,augment=preprocessing(),preprocessing(True)
    @lru_cache(maxsize=256)
    def crop(row):return load_crop(row).resize((256,160))
    history=[];best=-1
    for round_index in range(args.rounds+1):
        model.eval()  # Freeze BatchNorm running statistics even while gradients are enabled.
        losses=[]
        if round_index:
            for step in range(args.steps):
                selected=random.sample(eligible,4)
                batch=[]
                for identity in selected:
                    cams=random.sample(sorted(groups[identity]),2)
                    batch.extend(random.choice(groups[identity][c]) for c in cams)
                images=torch.stack([augment(crop(r)) for r in batch])
                labels=torch.tensor([i for i in range(4) for _ in range(2)])
                optimizer.zero_grad(set_to_none=True)
                vectors,_=model(images)
                loss=batch_hard_loss(vectors,labels)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                losses.append(float(loss.detach()))
                if (step+1)%50==0:print(json.dumps({'round':round_index,'step':step+1,'loss':float(np.mean(losses))}),flush=True)
        encoded=[]
        with torch.inference_mode():
            for begin in range(0,len(val),16):
                batch=torch.stack([transform(crop(r)) for r in val[begin:begin+16]])
                encoded.append(model(batch)[0].numpy())
        matrix=np.vstack(encoded)
        metrics=evaluate(matrix,val,True)
        record={'round':round_index,'loss':float(np.mean(losses)) if losses else None,**metrics}
        history.append(record)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='errors'}),flush=True)
        if metrics['mAP']>best:
            best=metrics['mAP']
            torch.save({'architecture':'resnet18-reid-v1','classes':1,'model':model.state_dict(),
                        'round':round_index,'seed':args.seed,'validation_mAP':best,
                        'preprocessing':'pil-bicubic-256x160-v1',
                        'initialization':'ImageNet1K_V1','training':'layer4 metric learning, frozen BN, cross-camera positives'},args.output/'best.pt')
            np.save(args.output/'validation_embeddings.npy',matrix)
    (args.output/'row_ids.json').write_text(json.dumps([r.image_id for r in val]))


if __name__=='__main__':main()
