"""Check supplied README format and reproduce Top-10 from saved embeddings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def validate(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    queries, gallery=manifest['query_ids'],manifest['gallery_ids']
    vectors=np.load(folder/'embeddings.npy',mmap_mode='r',allow_pickle=False)
    if vectors.ndim!=2 or len(vectors)!=len(queries)+len(gallery) or vectors.dtype!=np.float32:
        raise ValueError('Embedding shape/order count mismatch')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('Invalid embeddings')
    gallery_matrix=vectors[len(queries):]
    expected={}
    with (folder/'submission.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.reader(stream)
        if next(reader)!=['query_id']+[f'gallery_id_{i}' for i in range(1,11)]:
            raise ValueError('Wrong submission columns')
        count=0
        for i,row in enumerate(reader):
            if i>=len(queries) or len(row)!=11 or row[0]!=queries[i]:
                raise ValueError('Wrong query row')
            scores=gallery_matrix@vectors[i]
            order=np.argsort(-scores,kind='stable')[:10]
            if row[1:]!=[gallery[j] for j in order]:
                raise ValueError('Submission does not match saved embeddings')
            threshold=manifest['threshold']
            expected[row[0]]={gallery[j]:float(scores[j]) for j in order if threshold is not None and scores[j]>=threshold}
            count+=1
        if count!=len(queries):
            raise ValueError('Missing queries')
    actual={q:{} for q in queries}
    seen_refusals=set()
    with (folder/'candidates.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['query_id','gallery_id','confidence']:
            raise ValueError('Wrong candidates columns')
        for row in reader:
            q,g=row['query_id'],row['gallery_id']
            if q not in actual:
                raise ValueError('Unknown query')
            if not g:
                if row['confidence'] or q in seen_refusals or actual[q]:
                    raise ValueError('Invalid refusal')
                seen_refusals.add(q)
            else:
                if g in actual[q] or q in seen_refusals:
                    raise ValueError('Duplicate candidate or mixed refusal')
                actual[q][g]=float(row['confidence'])
    for q in queries:
        if actual[q].keys()!=expected[q].keys() or (not actual[q] and q not in seen_refusals):
            raise ValueError('Candidate decision mismatch')
        for g,score in actual[q].items():
            if not np.isclose(score,expected[q][g],atol=1e-6):
                raise ValueError('Confidence mismatch')
    return {'passed':True,'queries':len(queries),'gallery':len(gallery),'dimensions':vectors.shape[1],
            'refused_queries':len(seen_refusals),'scope':'format and reproducibility, not identification accuracy'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    print(json.dumps(validate(parser.parse_args().folder)))
