# Датасет подключён: актуальный протокол v4

Полный архив получен из D:/Abeba/Documents/dataset.zip. Предыдущие сообщения о нулевом архиве относятся к старому пути OneDrive и больше не блокируют работу.

train.csv: 9 556 объектов, 1 541 vehicle_id, 96 camera_id. На каждый ID 4–8 снимков, минимум две камеры. test_query.csv: 1 110 строк. test_gallery.csv: 750 строк.

Аудит: все изображения читаются, BBox внутри кадра, image_id не пересекаются между частями. Девять пар одинаковых полных кадров принадлежат различным ID. Связанные по хешу кадра ID объединяются при разбиении train/validation.

## Актуальная команда экспорта

В архиве v4 `models/best.pt` — реальные экспериментальные веса, а не искусственный smoke-test. `metrics.json` содержит сравнение и SHA-256 весов, `split.json` — ID разбиения. Весовой файл не делает решение готовым: текущая точность низкая.

## Запуск нейросервиса в Windows

Распакуйте весь архив. Откройте PowerShell в папке, где находятся `app`, `models` и `requirements-ml.txt`. Установка зависимостей требует интернета; после установки сам нейросетевой инференс веса не скачивает.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
$env:FALCON_CHECKPOINT = (Resolve-Path models/best.pt).Path
$env:FALCON_DATA_DIR = (Join-Path (Get-Location) 'data-neural')
.\.venv\Scripts\python.exe -m app.server
```

Откройте http://127.0.0.1:8000. До калибровки порога нейросервис показывает кандидатов, но отказывается подтверждать совпадение. Это намеренное ограничение, не поломка. Не переносите демонстрационные пороги классического алгоритма на нейросеть. `start_falcon.bat` без переменных окружения запускает классический вариант. Имеющийся Dockerfile также относится к базовому варианту, не к нейросетевой поставке.

## Экспорт без утверждения о калиброванном отказе

```powershell
.\.venv\Scripts\python.exe scripts/export_official.py dataset --checkpoint models/best.pt --output runs/technical-export
.\.venv\Scripts\python.exe scripts/validate_official.py runs/technical-export
```

Папку `dataset` подставьте свою: она должна содержать CSV и `images`. Этот технический экспорт создаёт Top-10, но все запросы в candidates.csv отклоняются, поскольку порог не указан. Это не готовая конкурсная подача.

После отдельного измерения и выбора порога используется команда ниже; `ПОРОГ` нужно заменить числом, а не копировать буквально:

```powershell
python scripts/export_official.py dataset --checkpoint models/best.pt --output runs/submission --threshold ПОРОГ
python scripts/validate_official.py runs/submission
```

Новый экспорт заменяет локальную схему scripts/challenge.py infer. В embeddings.npy сначала query, затем gallery. submission.csv содержит query_id и десять gallery_id. В candidates.csv строки query_id,gallery_id,confidence; отказ — строка с двумя пустыми полями. Confidence сейчас косинусное сходство, а не вероятность.

Без --threshold экспорт намеренно отклоняет все запросы в candidates.csv, сохраняя Top-10 в submission.csv. Порог выбирается по отдельной валидации, не по скрытым тестовым меткам.

## Первый эксперимент

Короткий запуск: ResNet18 ImageNet1K_V1, CPU, seed=42, два цикла по 50 сбалансированных мини-батчей. Каждый цикл здесь — заданное число шагов, не полный проход по train. Обучение на 7 629 наблюдениях / 1 233 ID; валидация на 1 927 наблюдениях / 308 ID. Отбор кандидатов на валидации исключает ту же камеру и тот же кадр. Это локальная валидация, не закрытый тест организатора.

Исходные публичные веса: https://download.pytorch.org/models/resnet18-f37072fd.pth
SHA-256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.

Требуются дальнейшее сравнение с исходной моделью, калибровка отказа, контроль остаточных признаков номера, нагрузочные проверки и финальная контейнерная сборка. Короткий эксперимент не означает готовность проекта к сдаче.

## Осторожное дообучение и разбор ошибок

Вторая конфигурация: только последний блок ResNet18, фиксированные статистики BatchNorm, triplet loss, шаг 1e-5, четыре ID по две камеры в мини-батче. Исходная модель тоже участвует в выборе лучшей контрольной точки. RGB-кроп приводится к 256×160 методом PIL BICUBIC; это сохранено в метаданных checkpoint и воспроизводится в API/экспорте. Метрики использовались для выбора checkpoint, поэтому это validation, а не независимый финальный test.

```powershell
python scripts/train_metric.py dataset split.json --output runs/metric --rounds 2 --steps 100
python scripts/error_report.py dataset runs/metric/validation_embeddings.npy runs/metric/row_ids.json --output runs/errors.html
```

Обучение с публичной инициализацией при отсутствии её в локальном кеше требует интернета. HTML содержит кропы данных — не публикуйте его вне команды без проверки условий использования датасета. Набор примеров намеренно выбирает самые уверенные ошибочные ответы, а не случайные снимки.
