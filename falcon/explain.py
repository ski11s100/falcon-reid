"""Интерпретируемость: на что модель смотрит, принимая решение о совпадении.

Два назначения, оба существенные.

1. Раздел 10 ТЗ: визуализация областей, повлиявших на решение, повышает доверие
   оператора. Это критерий-арбитр при равенстве основных метрик.

2. Защита от дисквалификации. Ответ 48 организаторов: финалистам прогоняют
   решение на контрольной версии теста, где зона номера дополнительно закрашена
   сплошной заливкой. Заметное падение метрики означает, что модель опиралась на
   остаточный сигнал в зоне номера — артефакты границ блюра, характерную
   текстуру. Раздел 9 ТЗ относит использование признаков номера к основаниям для
   дисквалификации. Отсутствие OCR в пайплайне этого само по себе НЕ доказывает,
   поэтому проверять нужно измерением, а не рассуждением.

Grad-CAM для ReID отличается от классификационного: на инференсе нет логита
класса. Скаляром для обратного распространения берётся косинусная близость
между запросом и конкретным кандидатом. Это отвечает ровно на тот вопрос,
который формулирует ТЗ: какие области повлияли на решение о СОПОСТАВЛЕНИИ
этих двух снимков.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import VehicleReID, ViTReID


class GradCAM:
    """Карта важности по градиентам последнего блока сети.

    ResNet: выход layer4, карта признаков 16×16. ViT: вход последнего блока
    внимания (blocks[-1].norm1). Вектор ViT берётся из CLS-токена, и патч-токены
    на ВЫХОДЕ последнего блока на него уже не влияют: градиент по ним нулевой.
    Поэтому берётся вход последнего внимания, где CLS ещё собирает информацию
    с патчей, а токены раскладываются обратно в сетку патчей 16×16.

    Использование:
        with GradCAM(model) as cam:
            heat = cam.similarity_map(query_tensor, reference_embedding)
    """

    def __init__(self, model: VehicleReID | ViTReID, layer: nn.Module | None = None):
        self.model = model
        self.tokens = isinstance(model, ViTReID)
        if layer is None:
            layer = model.backbone.blocks[-1].norm1 if self.tokens else model.backbone.layer4
        self.layer = layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> GradCAM:
        self._handles.append(self.layer.register_forward_hook(self._save_activations))
        self._handles.append(self.layer.register_full_backward_hook(self._save_gradients))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activations(self, module, inputs, output) -> None:
        self._activations = output

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def _as_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Токены ViT (B, 1+N, C) -> карта (B, C, h, w) без служебных токенов."""
        backbone = self.model.backbone
        height, width = backbone.patch_embed.grid_size
        patches = tokens[:, backbone.num_prefix_tokens:, :]
        return patches.reshape(tokens.size(0), height, width, -1).permute(0, 3, 1, 2)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """Прямой проход с сохранением графа: inference_mode здесь недопустим."""
        if self.tokens:
            return self.model(images)  # в режиме eval ViTReID отдаёт нормированный вектор
        pooled = self.model.pool(self.model.backbone(images))
        feature = self.model.bottleneck(self.model.reduce(pooled))
        return F.normalize(feature, dim=1)

    def similarity_map(self, images: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
        """Карта важности для близости между запросом и эталонным вектором.

        images: (1, 3, H, W); reference: (D,) или (1, D) — эмбеддинг кандидата.
        Возвращает массив (H, W) со значениями в диапазоне 0..1.
        """
        was_training = self.model.training
        self.model.eval()
        # Веса замораживаем: градиенты нужны только по активациям карты признаков.
        frozen = [p.requires_grad for p in self.model.parameters()]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        try:
            images = images.clone().requires_grad_(True)
            reference = reference.reshape(1, -1).to(images.device, images.dtype)
            reference = F.normalize(reference, dim=1)

            embedding = self._embed(images)
            score = (embedding * reference).sum()

            self.model.zero_grad(set_to_none=True)
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Не удалось перехватить активации: проверьте выбранный слой")

            # Вес канала — среднее его градиента: насколько сильно канал влияет
            # на близость. Отрицательный вклад отбрасывается ReLU: интересуют
            # области, которые СБЛИЖАЮТ снимки, а не отдаляют.
            activations, gradients = self._activations, self._gradients
            if self.tokens:
                activations, gradients = self._as_grid(activations), self._as_grid(gradients)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

            heat = cam[0, 0].detach().float().cpu().numpy()
            span = float(heat.max() - heat.min())
            return (heat - heat.min()) / span if span > 1e-12 else np.zeros_like(heat)
        finally:
            for parameter, flag in zip(self.model.parameters(), frozen):
                parameter.requires_grad_(flag)
            self.model.train(was_training)


@dataclass
class PlateRegionReport:
    """Сколько внимания модели приходится на зону вероятного номера."""

    attention_share: float
    region_area_share: float
    concentration: float
    verdict: str

    def as_dict(self) -> dict:
        return {
            "attention_share": round(self.attention_share, 4),
            "region_area_share": round(self.region_area_share, 4),
            "concentration": round(self.concentration, 3),
            "verdict": self.verdict,
        }


def plate_region_mask(height: int, width: int) -> np.ndarray:
    """Маска зоны, где чаще всего находится номер на кропе ТС.

    Точные координаты пластин участникам не передаются (ответ 48), поэтому зона
    задаётся геометрической эвристикой: центральная треть по горизонтали, полоса
    от 55% до 90% высоты. Для фронтальных и задних ракурсов номер попадает сюда
    почти всегда; для строго боковых — нет, и это ограничение метода.
    """
    mask = np.zeros((height, width), dtype=bool)
    mask[int(0.55 * height):int(0.90 * height), int(0.33 * width):int(0.67 * width)] = True
    return mask


def plate_attention(heatmap: np.ndarray) -> PlateRegionReport:
    """Доля внимания в зоне номера относительно её площади.

    concentration = доля внимания / доля площади. Значение около 1 означает, что
    зона получает внимание пропорционально своему размеру, то есть ничем не
    выделена. Значения заметно выше 1 — повод разбираться.
    """
    mask = plate_region_mask(*heatmap.shape)
    total = float(heatmap.sum())
    if total <= 1e-12:
        return PlateRegionReport(0.0, float(mask.mean()), 0.0, "карта пуста")

    attention_share = float(heatmap[mask].sum() / total)
    area_share = float(mask.mean())
    concentration = attention_share / max(area_share, 1e-9)

    # Карта внимания — только подсказка: в «зону номера» по геометрии попадают
    # и решётка радиатора, фары, бампер. Вывод об опоре на номер делается
    # маскированием (plate_masking_boxes и /api/explain), а не отсюда.
    if concentration < 1.3:
        verdict = "нижняя центральная зона не выделена"
    else:
        verdict = "нижняя центральная зона заметна на карте; решает проверка маскированием"
    return PlateRegionReport(attention_share, area_share, concentration, verdict)


def plate_masking_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """Зона номера и контрольные зоны той же площади, в пикселях кропа.

    Для проверки одной пары «запрос — кандидат»: каждая зона закрашивается на
    запросе, и смотрится, насколько падает сходство. Если зона номера роняет его
    не сильнее контрольных, модель на номер не опирается. Контроли фиксированы,
    чтобы проверка была воспроизводимой.
    """
    def box(x0, y0, x1, y1):
        return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))

    return {
        "зона номера": box(0.33, 0.55, 0.67, 0.90),
        "верх": box(0.33, 0.05, 0.67, 0.40),
        "левый бок": box(0.02, 0.35, 0.36, 0.70),
        "правый бок": box(0.64, 0.35, 0.98, 0.70),
    }


def mask_region(images: torch.Tensor, mask: np.ndarray, fill: float = 0.0) -> torch.Tensor:
    """Закрашивает область тензора — имитация контрольной версии организаторов."""
    masked = images.clone()
    region = torch.from_numpy(mask).to(images.device)
    masked[..., region] = fill
    return masked


@torch.inference_mode()
def masking_robustness(model: VehicleReID, images: torch.Tensor,
                       masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Насколько меняется эмбеддинг при закрашивании разных областей.

    Ключевая проверка перед защитой. Смысл в СРАВНЕНИИ: если закрашивание зоны
    номера рушит признак сильнее, чем закрашивание случайной области той же
    площади, модель на номер опирается. Если падение сопоставимо — не опирается.
    """
    was_training = model.training
    model.eval()
    try:
        baseline = model(images)
        result: dict[str, float] = {}
        for name, mask in masks.items():
            altered = model(mask_region(images, mask))
            similarity = F.cosine_similarity(baseline, altered, dim=1)
            result[name] = float(similarity.mean())
        return result
    finally:
        model.train(was_training)


def control_masks(height: int, width: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Зона номера и контрольные области той же площади для сравнения."""
    plate = plate_region_mask(height, width)
    area = int(plate.sum())
    box_h = int(0.35 * height)
    box_w = max(1, area // max(1, box_h))

    rng = np.random.default_rng(seed)
    masks = {"зона номера": plate}
    for index, label in enumerate(("контроль: верх", "контроль: случайная область")):
        mask = np.zeros((height, width), dtype=bool)
        if index == 0:
            top = int(0.05 * height)
        else:
            top = int(rng.integers(0, max(1, height - box_h)))
        left = int(rng.integers(0, max(1, width - box_w)))
        mask[top:top + box_h, left:left + box_w] = True
        masks[label] = mask
    return masks


def overlay_heatmap(image, heatmap: np.ndarray, alpha: float = 0.45):
    """Накладывает карту важности на изображение. Возвращает PIL.Image."""
    from PIL import Image

    base = image.convert("RGB")
    heat = np.asarray(Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255))
                      .resize(base.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    # Палитра под интерфейс: холодное тёмное для неважного, красный для важного.
    coloured = np.zeros((*heat.shape, 3), dtype=np.float32)
    coloured[..., 0] = np.clip(heat * 2.0, 0, 1)
    coloured[..., 1] = np.clip(heat * 1.4 - 0.5, 0, 1)
    coloured[..., 2] = np.clip(0.35 - heat * 0.35, 0, 1)

    original = np.asarray(base, dtype=np.float32) / 255.0
    weight = (alpha * heat)[..., None]
    blended = original * (1.0 - weight) + coloured * weight
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
