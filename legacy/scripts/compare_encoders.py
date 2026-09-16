"""Compare initial and trained encoders on exactly the same held-out identities."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from app.dataset import read_manifest,load_crop
from app.neural import ReID,preprocessing,NeuralEncoder
from app.fingerprint import create_fingerprint
from app.retrieval import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',type=Path)
    p.add_argument('split',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    selected=set(json.loads(args.split.read_text())['validation_ids'])
    rows=[r for r in read_manifest(args.dataset/'train.csv',args.dataset/'images',True) if r.vehicle_id in selected]
    args.output.mkdir(parents=True,exist_ok=False)
    crops=[]
    for i,r in enumerate(rows):
        crops.append(load_crop(r).resize((256,160)))
        if i%500==0: print(json.dumps({'crops_loaded':i+1,'total':len(rows)}),flush=True)
    initial=ReID(1,pretrained=True).eval()
    transform=preprocessing()
    def pretrained(image):
        with torch.inference_mode():
            return initial(transform(image)[None])[0][0].numpy()
    encoders={'classical':create_fingerprint,'imagenet_resnet18':pretrained}
    if args.checkpoint:
        encoders['pilot']=NeuralEncoder(args.checkpoint)
    report={}
    for name,encoder in encoders.items():
        vectors=[]
        start=time.perf_counter()
        for i,image in enumerate(crops):
            vectors.append(encoder(image))
            if i%500==0:print(json.dumps({'encoder':name,'embedded':i+1}),flush=True)
        matrix=np.vstack(vectors)
        np.save(args.output/f'{name}.npy',matrix)
        metrics=evaluate(matrix,rows,True)
        metrics['elapsed_seconds']=time.perf_counter()-start
        report[name]=metrics
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'encoder':name,**{k:v for k,v in metrics.items() if k!='errors'}}),flush=True)
    (args.output/'row_ids.json').write_text(json.dumps([r.image_id for r in rows]),encoding='utf-8')


if __name__=='__main__':main()
