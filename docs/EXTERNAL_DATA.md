# Внешние датасеты для предобучения

Раздел 7 ТЗ разрешает публичные предобученные веса и сторонние открытые датасеты
и требует перечислить все внешние источники в Readme.md. Ответ организаторов 46
уточняет критерий допустимости: источник должен быть публичным и проверяемым
(URL, версия, автор), тип лицензии значения не имеет — включая CC BY-NC и доступ
по заявке. Недопустимы только приватные репозитории и внутренние данные компаний.

Зачем это нужно: в конкурсном train.csv 9 556 снимков и 1 541 автомобиль. Это
мало для обучения ReID с нуля. Предобучение на большом датасете машин с
последующим дообучением на конкурсных данных — самый крупный одиночный прирост
точности, доступный без изменения архитектуры.

## Что запрашиваем

| Датасет | Объём | Доступ | Приоритет |
|---|---|---|---|
| VeRi-776 | 49 357 снимков, 776 ТС, 20 камер | письмо автору | высокий |
| VERI-Wild | 416 314 снимков, 40 671 ТС, 174 камеры | письмо автору | высокий |
| VehicleID (PKU) | 221 763 снимка, 26 267 ТС | форма PKU | средний |

VeRi-776 ближе всего к нашей задаче по природе данных: городское видеонаблюдение,
кросс-камерные съёмки, тот же порядок разрешения и ракурсов.

## Письмо 1: VeRi-776

Кому: `xinchenliu@bupt.cn`
Тема: `Request for access to the VeRi-776 dataset`

```
Dear Dr. Liu,

I am writing to request access to the VeRi-776 dataset for non-commercial
research use.

Full name: <ФИО латиницей>
Affiliation: <организация или университет>
Purpose: academic research on cross-camera vehicle re-identification;
the dataset will be used only for model pre-training and will not be
redistributed to any third party or published publicly.

Thank you for making this dataset available to the research community.

Best regards,
<ФИО>
<контактный email>
```

## Письмо 2: VERI-Wild

Кому: `yanbai@pku.edu.cn`
Тема: `Request for access to the VERI-Wild dataset`

```
Dear Dr. Bai,

I would like to request access to the VERI-Wild dataset for non-commercial
research purposes.

Full name: <ФИО латиницей>
Affiliation: <организация или университет>

I confirm that the dataset will be used for non-commercial research only,
will not be given to any third party and will not be published publicly.

Thank you for your time.

Best regards,
<ФИО>
<контактный email>
```

## Доступно немедленно, без заявки

**VeRi-CARLA** — https://github.com/sekilab/VehicleReIdentificationDataset
Лицензия Apache-2.0, скачивается сразу. 55 000 снимков, 85 камер, но данные
синтетические (симулятор CARLA), а не реальное видеонаблюдение. Как замена
VeRi-776 не годится, как дополнительная аугментация разнообразия ракурсов —
может дать небольшой прирост. Низкий приоритет, берём если останется время.

**Зеркало VeRi-776 на Kaggle** — существует по адресу
kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

Формально оно удовлетворяет критерию организаторов: публичная ссылка, которую
жюри может открыть и проверить. Но это чужая перезаливка, а исходная лицензия
VeRi-776 прямо запрещает передачу третьим лицам и публикацию. То есть само
зеркало нарушает условия авторов датасета, даже если нам за это ничего не будет.

Решение за командой. Разумный вариант: отправить официальные письма сегодня, а
зеркалом при необходимости воспользоваться как временным, пока идёт ответ, и в
любом случае указать в Readme именно официальный источник, если доступ придёт.

## Что фиксируем в документации

По разделу 7 ТЗ и ответу 39 в Readme решения для каждого внешнего источника
указываем: название, URL, версию или тег релиза, автора, дату получения доступа
и роль в пайплайне (предобучение / инициализация / вспомогательная модель).

Веса ImageNet+IBN, которые уже используются:

| Источник | URL | Контрольная сумма |
|---|---|---|
| ResNet50-IBN-a | github.com/XingangPan/IBN-Net, релиз v1.0 | `resnet50_ibn_a-d9d0bb7b.pth` |
