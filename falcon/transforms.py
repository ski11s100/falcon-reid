"""Препроцессинг и аугментации.

Разрешение 256x256 выбрано осознанно. Пешеходный ReID использует вытянутые
128x256, но автомобиль в кадре ближе к квадрату, и вытягивание искажает
пропорции кузова — один из основных различающих признаков. Признаки, которые
перечислены в задаче (наклейки, диски, повреждения, рейлинги), мелкие, поэтому
уходить ниже 256 нельзя: на 160 пикселях они физически не разрешаются.

Набор аугментаций отвечает конкретным пунктам раздела 5 ТЗ про покрытие условий:
  * ColorJitter          -> разное время суток и погода;
  * RandomErasing        -> частичное перекрытие ТС другим объектом;
  * Pad + RandomCrop     -> неточность рамки BBox и разный масштаб;
  * HorizontalFlip       -> левый и правый ракурс;
  * PlateZoneErase       -> закраска зоны номера (ответ 48, см. ниже).

Вертикальное отражение и сильный поворот НЕ используются: камеры городской
инфраструктуры не переворачивают кадр, и такая аугментация только зашумляет.
"""

from __future__ import annotations

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_SIZE = (256, 256)


class PlateZoneErase:
    """Закраска зоны номера сплошным цветом при обучении.

    Ответ 48 организаторов: финалистов проверяют на тесте, где зона номера
    закрашена сплошной заливкой, и падение метрики — основание для
    дисквалификации. Номер уже размыт, но размытое пятно и артефакты его границ
    остаются признаком, за который модель может зацепиться. Если при обучении
    та же зона то и дело закрашена, модели выгоднее опираться на кузов.

    Номер висит внизу по центру кропа, но точное место зависит от ракурса и
    модели машины, поэтому центр, ширина и высота зоны случайны в пределах
    типичных положений, а цвет заливки — случайный сплошной. Работает с
    нормированным тензором, как RandomErasing.
    """

    def __init__(self, probability: float = 0.5):
        self.probability = probability
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.probability:
            return tensor
        _, height, width = tensor.shape
        cx, cy, bw, bh = (lo + (hi - lo) * torch.rand(1).item() for lo, hi in
                          ((0.35, 0.65), (0.62, 0.86), (0.16, 0.34), (0.08, 0.20)))
        x0, x1 = int(max(0.0, cx - bw / 2) * width), int(min(1.0, cx + bw / 2) * width)
        y0, y1 = int(max(0.0, cy - bh / 2) * height), int(min(1.0, cy + bh / 2) * height)
        fill = (torch.rand(3, 1, 1) - self.mean) / self.std
        tensor = tensor.clone()
        tensor[:, y0:y1, x0:x1] = fill
        return tensor


def build_train_transform(size: tuple[int, int] = DEFAULT_SIZE, *,
                          erasing_probability: float = 0.5,
                          jitter: float = 0.25,
                          plate_erase: float = 0.0) -> transforms.Compose:
    height, width = size
    extra = [PlateZoneErase(plate_erase)] if plate_erase > 0 else []
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=jitter, contrast=jitter,
                               saturation=jitter, hue=jitter * 0.2),
        # Padding + случайный кроп имитируют неточность BBox: в реальной системе
        # рамку даёт детектор, и она никогда не идеальна.
        transforms.Pad(10, fill=0, padding_mode="edge"),
        transforms.RandomCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # RandomErasing идёт ПОСЛЕ нормализации: стирается участок тензора,
        # а не пикселей, что соответствует оригинальной статье.
        transforms.RandomErasing(p=erasing_probability, scale=(0.02, 0.33),
                                 ratio=(0.3, 3.3), value=0),
        *extra,
    ])


def build_eval_transform(size: tuple[int, int] = DEFAULT_SIZE) -> transforms.Compose:
    """Детерминированный препроцессинг. Точно этот же путь идёт в сдачу."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
