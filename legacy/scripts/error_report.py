"""Generate a self-contained, reproducible cross-camera retrieval error review."""
import argparse
import base64
import hashlib
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.dataset import load_crop, read_manifest
from app.retrieval import evaluate, normalize, ranked


def thumbnail(row):
    crop = load_crop(row)
    crop.thumbnail((320, 200))
    buffer = io.BytesIO()
    crop.save(buffer, format='JPEG', quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    caption = html.escape(f'ID {row.vehicle_id}; камера {row.camera_id}; {row.image_id}')
    return f'<figure><img src="data:image/jpeg;base64,{encoded}" alt="Автомобиль"><figcaption>{caption}</figcaption></figure>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('embeddings', type=Path)
    parser.add_argument('row_ids', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    ids = json.loads(args.row_ids.read_text(encoding='utf-8'))
    manifest = read_manifest(args.dataset/'train.csv', args.dataset/'images', True)
    by_id = {row.image_id: row for row in manifest}
    if len(by_id) != len(manifest) or len(set(ids)) != len(ids):
        raise ValueError('Ambiguous or duplicate image IDs')
    rows = [by_id[value] for value in ids]
    matrix = normalize(np.load(args.embeddings, allow_pickle=False))
    metrics = evaluate(matrix, rows, True)
    errors = sorted(metrics['errors'], key=lambda error: (-error['score'], error['query_row']))
    cards = []
    for error in errors[:args.limit]:
        index = error['query_row']
        order, scores = ranked(matrix, rows, index, True)
        positive_position = next(i for i,j in enumerate(order) if rows[j].vehicle_id == rows[index].vehicle_id)
        positive = int(order[positive_position])
        cards.append(
            f'<section><h2>Ошибка: сходство {error["score"]:.4f}; верный автомобиль на позиции {positive_position+1}</h2>'
            '<div class="grid"><div><h3>Запрос</h3>'+thumbnail(rows[index])+'</div>'
            '<div><h3>Ошибочный Top-1</h3>'+thumbnail(rows[error['candidate_row']])+'</div>'
            '<div><h3>Первый верный кандидат</h3>'+thumbnail(rows[positive])+'</div></div></section>'
        )
    summary = {key:value for key,value in metrics.items() if key != 'errors'}
    summary.update({'embedding_sha256': hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),
                    'row_ids_sha256': hashlib.sha256(args.row_ids.read_bytes()).hexdigest(),
                    'review_selection': 'Highest cosine similarity among incorrect Top-1 matches',
                    'displayed_errors': len(cards), 'total_top1_errors': len(errors)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    header = f'<p>mAP: {metrics["mAP"]:.2%} · Rank-1: {metrics["Rank-1"]:.2%} · Rank-5: {metrics["Rank-5"]:.2%} · запросов: {len(rows)}</p>'
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ФАЛЬКОН — проверка ошибок</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:32px auto;padding:16px}section{background:#1f2937;padding:20px;margin:20px 0;border-radius:12px}h2{font-size:19px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}img{max-width:100%;height:200px;object-fit:contain}figcaption{font-size:12px;overflow-wrap:anywhere;color:#cbd5e1}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>
<h1>ФАЛЬКОН: ошибки поиска между камерами</h1>'''+header+'''
<p>Локальная валидация, не результат закрытого теста организаторов. Тот же кадр и та же камера исключены. Показаны самые уверенные ошибочные Top-1: это намеренно сложная выборка, а не случайная подборка. Сходство не является вероятностью совпадения.</p>
<p>При разборе проверяйте ракурс, фон, перекрытия, качество BBox, цвет и остаточные признаки размытого номера. Причины ошибок автоматически не устанавливаются.</p>'''+''.join(cards)+'</html>'
    args.output.write_text(document, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
