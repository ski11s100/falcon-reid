"""Export query-to-gallery retrieval in the supplied dataset README format."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.dataset import read_manifest, load_crop
from app.fingerprint import create_fingerprint, FINGERPRINT_VERSION
from app.retrieval import normalize


def write_rankings(folder, queries, gallery, embeddings, threshold=None):
    if len(gallery) < 10:
        raise ValueError('Official submission requires at least ten gallery objects')
    qids, gids = [r.image_id for r in queries], [r.image_id for r in gallery]
    if len(qids) != len(set(qids)) or len(gids) != len(set(gids)) or set(qids) & set(gids):
        raise ValueError('Duplicate image IDs or overlapping query/gallery sets')
    matrix = normalize(embeddings)
    if len(matrix) != len(queries)+len(gallery):
        raise ValueError('Expected query embeddings first, then gallery')
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    with (folder/'submission.csv').open('w',newline='',encoding='utf-8') as s, (folder/'candidates.csv').open('w',newline='',encoding='utf-8') as c:
        submission, candidates = csv.writer(s), csv.writer(c)
        submission.writerow(['query_id']+[f'gallery_id_{i}' for i in range(1,11)])
        candidates.writerow(['query_id','gallery_id','confidence'])
        for i, query in enumerate(queries):
            scores = matrix[len(queries):] @ matrix[i]
            order = np.argsort(-scores,kind='stable')[:10]
            submission.writerow([query.image_id]+[gids[j] for j in order])
            accepted = [j for j in order if threshold is not None and float(scores[j]) >= threshold]
            if not accepted:
                candidates.writerow([query.image_id,'',''])
            for j in accepted:
                candidates.writerow([query.image_id,gids[j],format(float(scores[j]),'.9g')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4,help='CPU threads for neural inference')
    parser.add_argument('--threshold',type=float)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    queries = read_manifest(args.dataset/'test_query.csv',args.dataset/'images')
    gallery = read_manifest(args.dataset/'test_gallery.csv',args.dataset/'images')
    encoder, version = create_fingerprint, FINGERPRINT_VERSION
    if args.checkpoint:
        import torch
        torch.set_num_threads(args.threads)
        from app.neural import NeuralEncoder
        encoder = NeuralEncoder(args.checkpoint,args.device)
        version = encoder.version
    args.output.mkdir(parents=True,exist_ok=False)
    rows = queries+gallery
    dimensions = len(encoder(load_crop(rows[0])))
    vectors = np.lib.format.open_memmap(args.output/'embeddings.npy',mode='w+',dtype='float32',shape=(len(rows),dimensions))
    for i,row in enumerate(rows):
        vectors[i] = normalize(encoder(load_crop(row))[None])[0]
        if i%100 == 0:
            print(json.dumps({'embedded':i+1,'total':len(rows)}),flush=True)
    vectors.flush()
    write_rankings(args.output,queries,gallery,vectors,args.threshold)
    manifest = {'format':'dataset-readme-query-gallery-v1','model':version,
                'query_ids':[r.image_id for r in queries], 'gallery_ids':[r.image_id for r in gallery],
                'threshold':args.threshold,'confidence_definition':'cosine similarity, not calibrated probability',
                'refusal_encoding':'one row with empty gallery_id and confidence',
                'inputs':{name:hashlib.sha256((args.dataset/name).read_bytes()).hexdigest()
                          for name in ['test_query.csv','test_gallery.csv']}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Official-format export finished',flush=True)


if __name__ == '__main__':
    main()
