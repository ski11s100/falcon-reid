"""Функции потерь ReID: классификация идентичностей + batch-hard triplet.

Почему именно две потери, а не одна:

* Cross-entropy по vehicle_id учит модель разделять машины из обучающей выборки,
  но ничего не говорит о том, как должны располагаться НЕ виденные машины.
  А в тесте по условиям задачи (раздел 5 ТЗ) все ТС — новые.
* Triplet loss учит геометрии пространства напрямую: «тот же автомобиль ближе,
  чем любой другой», и это свойство переносится на незнакомые машины.

Вместе они дают заметно больше, чем по отдельности: классификация быстро
структурирует пространство, triplet доводит границы между похожими машинами.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy со сглаживанием меток.

    Без сглаживания сеть переобучается на 1233 обучающих идентичности и начинает
    выдавать переуверенные логиты. Сглаживание оставляет долю epsilon на «не знаю»
    и заметно улучшает перенос на новые машины.
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon должен лежать в [0, 1)")
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.epsilon / (n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        return torch.mean(torch.sum(-smooth * log_probs, dim=1))


def euclidean_distances(features: torch.Tensor) -> torch.Tensor:
    """Попарные евклидовы расстояния внутри батча, устойчиво к нулевым нормам."""
    squared = torch.cdist(features, features, p=2)
    return squared.clamp(min=1e-12)


class BatchHardTripletLoss(nn.Module):
    """Triplet loss с отбором самого трудного позитива и негатива в батче.

    Идея: тянуть надо не за случайные пары, а за те, на которых модель ошибается
    сильнее всего. Для каждого якоря берётся самый ДАЛЁКИЙ снимок того же ТС и
    самый БЛИЗКИЙ снимок чужого ТС. Работает только если батч собран PK-сэмплером.

    soft_margin=True использует softplus вместо margin: не требует подбора margin
    и не «выключает» градиент, когда пара уже удовлетворяет условию.
    """

    def __init__(self, margin: float = 0.3, soft_margin: bool = True):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                cameras: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        distances = euclidean_distances(features)
        n = labels.size(0)

        same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)
        not_self = ~torch.eye(n, dtype=torch.bool, device=labels.device)
        positive_mask = same_identity & not_self
        negative_mask = ~same_identity

        if cameras is not None:
            # В метрике засчитываются только кросс-камерные совпадения (ответ 11),
            # поэтому позитивом считаем снимок того же ТС с ДРУГОЙ камеры, если
            # такой в батче есть. Для ID без второй камеры откатываемся обратно.
            cross_camera = positive_mask & (cameras.unsqueeze(0) != cameras.unsqueeze(1))
            has_cross = cross_camera.any(dim=1, keepdim=True)
            positive_mask = torch.where(has_cross, cross_camera, positive_mask)

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            zero = features.sum() * 0.0
            return zero, {"triplet_active": 0.0, "margin_violation": 0.0}

        hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        hardest_positive = hardest_positive[valid]
        hardest_negative = hardest_negative[valid]

        if self.soft_margin:
            loss = F.softplus(hardest_positive - hardest_negative).mean()
        else:
            loss = F.relu(hardest_positive - hardest_negative + self.margin).mean()

        with torch.no_grad():
            violation = (hardest_positive + self.margin > hardest_negative).float().mean()
            stats = {
                "triplet_active": float(valid.float().mean()),
                "margin_violation": float(violation),
                "d_pos": float(hardest_positive.detach().mean()),
                "d_neg": float(hardest_negative.detach().mean()),
            }
        return loss, stats


class CenterLoss(nn.Module):
    """Стягивает признаки каждой идентичности к обучаемому центру.

    Опциональная добавка с малым весом (~0.0005): уменьшает внутриклассовый
    разброс, что особенно помогает, когда у ТС всего 4-8 снимков.
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers = self.centers.index_select(0, labels)
        return F.mse_loss(features, centers, reduction="sum") / (2.0 * features.size(0))


class ReIDCriterion(nn.Module):
    """Итоговая целевая функция: ID-loss + triplet (+ опционально center)."""

    def __init__(self, num_classes: int, feature_dim: int = 2048, *,
                 id_weight: float = 1.0, triplet_weight: float = 1.0,
                 center_weight: float = 0.0, label_smoothing: float = 0.1,
                 margin: float = 0.3, soft_margin: bool = True):
        super().__init__()
        self.identity = LabelSmoothingCrossEntropy(label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=margin, soft_margin=soft_margin)
        self.center = CenterLoss(num_classes, feature_dim) if center_weight > 0 else None
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight

    def forward(self, triplet_features: torch.Tensor, logits: torch.Tensor,
                labels: torch.Tensor, cameras: torch.Tensor | None = None):
        id_loss = self.identity(logits, labels)
        triplet_loss, stats = self.triplet(triplet_features, labels, cameras)
        total = self.id_weight * id_loss + self.triplet_weight * triplet_loss

        center_loss = None
        if self.center is not None:
            center_loss = self.center(triplet_features, labels)
            total = total + self.center_weight * center_loss

        # Все числа для журнала снимаются под no_grad и через detach: иначе
        # обращение float(tensor) к тензору с градиентом тянет за собой граф.
        with torch.no_grad():
            parts = {
                "id": float(id_loss.detach()),
                "triplet": float(triplet_loss.detach()),
                "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
                "total": float(total.detach()),
                **stats,
            }
            if center_loss is not None:
                parts["center"] = float(center_loss.detach())
        return total, parts
