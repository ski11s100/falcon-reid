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
  * HorizontalFlip       -> левый и правый ракурс.

Вертикальное отражение и сильный поворот НЕ используются: камеры городской
инфраструктуры не переворачивают кадр, и такая аугментация только зашумляет.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_SIZE = (256, 256)


def build_train_transform(size: tuple[int, int] = DEFAULT_SIZE, *,
                          erasing_probability: float = 0.5,
                          jitter: float = 0.25) -> transforms.Compose:
    height, width = size
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
    ])


def build_eval_transform(size: tuple[int, int] = DEFAULT_SIZE) -> transforms.Compose:
    """Детерминированный препроцессинг. Точно этот же путь идёт в сдачу."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
