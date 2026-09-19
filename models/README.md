# Веса модели

Здесь лежат веса ансамбля, который идёт в сдачу и в сервис (Git LFS).
Dockerfile копирует этот каталог в образ: во время инференса сеть недоступна,
поэтому веса должны быть внутри образа (раздел 8 ТЗ, ответ 39).

| Файл | Модель | Как получен | Размер |
|---|---|---|---|
| `clip-ours.pt` | CLIP ViT-B/16 + BNNeck | `runs/plate/clip-ours/best.pt`, экспорт в fp16 | 174 МБ |
| `clip-veri-ours.pt` | CLIP ViT-B/16 + BNNeck | `runs/plate/clip-veri-ours/best.pt`, экспорт в fp16 | 174 МБ |
| `resnet-v2-veri.pt` | ResNet50-IBN-a + GeM + BNNeck | `runs/plate/resnet-v2-veri/best.pt`, экспорт в fp16 | 52 МБ |

Итого 399 МБ при ограничении раздела 7 ТЗ в 2 ГБ на все веса инференса.
Все три модели — основное обучение и затем 15 эпох дообучения с закраской
зоны номера (`scripts/finetune_plate_erase.py`).

Получить заново:

```powershell
.\.venv\Scripts\python.exe scripts/overnight_clip.py <каталог данных>
.\.venv\Scripts\python.exe scripts/finetune_plate_erase.py <каталог данных>
.\.venv\Scripts\python.exe scripts/export_weights.py runs/plate/clip-ours/best.pt models/clip-ours.pt
.\.venv\Scripts\python.exe scripts/export_weights.py runs/plate/clip-veri-ours/best.pt models/clip-veri-ours.pt
.\.venv\Scripts\python.exe scripts/export_weights.py runs/plate/resnet-v2-veri/best.pt models/resnet-v2-veri.pt
```

Экспорт в fp16 вдвое уменьшает файл и не меняет векторы: вывод и так идёт в
fp16, косинус между векторами исходного и экспортированного чекпойнта не ниже
0.999999, метрики совпадают до знака. Состав ансамбля и порог отказа заданы
константами `SUBMISSION_CHECKPOINTS` и `CALIBRATED_THRESHOLD` в
`falcon/submit.py`: порог откалиброван под этот состав и на другой не переносится.
