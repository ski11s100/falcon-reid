"""Check internal consistency of a local inference export, not organizer schema."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def validate(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    vectors = np.load(folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    n = manifest['rows']
    if vectors.shape != (n, manifest['dimensions']) or vectors.dtype != np.float32:
        raise ValueError('Embedding shape/dtype mismatch')
    for begin in range(0, n, 1024):
        block = vectors[begin:begin+1024]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
            raise ValueError('Embeddings must be finite and normalized')
    for filename in ('submission.csv','candidates.csv'):
        with (folder / filename).open(encoding='utf-8', newline='') as stream:
            count = 0
            for i, row in enumerate(csv.DictReader(stream)):
                if int(row['query_row_id']) != i:
                    raise ValueError('Query row order mismatch')
                ids = json.loads(row.get('top10_row_ids', row.get('candidate_row_ids')))
                if len(ids) > 10 or len(ids) != len(set(ids)):
                    raise ValueError('Duplicate or excessive candidates')
                if any(not isinstance(j, int) or j < 0 or j >= n or j == i for j in ids):
                    raise ValueError('Candidate row out of range or self match')
                if any(manifest['row_order'][j]['image_id'] == manifest['row_order'][i]['image_id'] for j in ids):
                    raise ValueError('Same-frame candidate')
                if 'scores' in row:
                    scores = json.loads(row['scores'])
                    if len(scores) != len(ids) or not np.isfinite(scores).all():
                        raise ValueError('Candidate scores mismatch')
                    if row['accepted'] == 'False' and ids:
                        raise ValueError('Refused query has candidates')
                count += 1
            if count != n:
                raise ValueError('Export row count mismatch')
    return {'passed': True, 'rows': n, 'dimensions': vectors.shape[1],
            'scope': 'local export consistency only; organizer schema not verified'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    print(json.dumps(validate(parser.parse_args().folder)))
