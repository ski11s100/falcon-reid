"""ReID-энкодер: ResNet50-IBN-a + GeM + BNNeck.

Обоснование архитектуры (это придётся защищать на питче):

1. IBN (Instance-Batch Normalization). Главная сложность задачи — один и тот же
   автомобиль выглядит по-разному на разных камерах: другое освещение, баланс
   белого, время суток, погода. InstanceNorm нормирует статистики каждого снимка
   по отдельности и тем самым вычищает именно «стиль» кадра, сохраняя содержание.
   IBN-a ставит половину каналов под IN, половину под BN в первых трёх стадиях:
   низкие слои избавляются от стиля, высокие сохраняют различающую информацию.
   Ровно то, что нужно для кросс-камерного ReID.

2. BNNeck (Luo et al., «Bag of Tricks for Person Re-ID»). Triplet loss работает
   в евклидовом пространстве и тянет признаки врозь, а ID-классификация работает
   с косинусными углами — если считать обе по одному и тому же вектору, градиенты
   конфликтуют. BNNeck разводит их: triplet считается ДО BatchNorm, классификация
   ПОСЛЕ, а на инференсе берётся вектор после BN. Стабильно даёт заметный прирост.

3. last_stride = 1. Убираем прореживание в layer4: карта признаков становится
   16x16 вместо 8x8. Детали, которые задача просит различать, — диски, наклейки,
   повреждения — мелкие, и терять по ним разрешение нельзя.

4. GeM pooling. Обобщение average/max pooling с обучаемой степенью p. Для машин
   важные признаки локальны и занимают малую долю кадра, поэтому усреднение по
   всей карте их размывает; GeM сам подбирает нужную резкость.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Официальные веса IBN-Net (XingangPan/IBN-Net, релиз v1.0).
# Версия и контрольная сумма зафиксированы: раздел 7 ТЗ требует воспроизводимости,
# а ответ 39 — фиксации внешних весов по тегу релиза с проверкой sha256.
IBN_WEIGHTS_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
IBN_WEIGHTS_SHA256_PREFIX = "d9d0bb7b"


class IBN(nn.Module):
    """Половина каналов через InstanceNorm, половина через BatchNorm."""

    def __init__(self, planes: int, ratio: float = 0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, dim=1)
        return torch.cat((self.IN(split[0].contiguous()), self.BN(split[1].contiguous())), dim=1)


class BottleneckIBN(nn.Module):
    """Bottleneck ResNet с опциональным IBN на первой нормализации."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, ibn: str | None = None,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if ibn == "a" else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == "b" else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.IN is not None:
            out = self.IN(out)
        return self.relu(out)


class ResNetIBN(nn.Module):
    """ResNet50-IBN-a без классификационной головы ImageNet.

    Именование слоёв совпадает с официальной реализацией IBN-Net, чтобы
    предобученный state_dict загружался напрямую, без переименований.
    """

    def __init__(self, layers=(3, 4, 6, 3), ibn_cfg=("a", "a", "a", None), last_stride: int = 1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64, affine=True) if ibn_cfg[0] == "b" else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(256, layers[2], stride=2, ibn=ibn_cfg[2])
        # last_stride=1 сохраняет разрешение карты признаков в последней стадии.
        self.layer4 = self._make_layer(512, layers[3], stride=last_stride, ibn=ibn_cfg[3])
        self.out_channels = 512 * BottleneckIBN.expansion

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.affine:
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, ibn: str | None = None) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BottleneckIBN.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )
        # IBN-a не применяется в последней стадии: там нужны стилевые признаки.
        layers = [BottleneckIBN(self.inplanes, planes, None if ibn == "b" else ibn, stride, downsample)]
        self.inplanes = planes * BottleneckIBN.expansion
        for index in range(1, blocks):
            layers.append(
                BottleneckIBN(self.inplanes, planes,
                              None if (ibn == "b" and index < blocks - 1) else ibn)
            )
        return nn.Sequential(*layers)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Пересчитывать активации при обратном проходе вместо хранения.

        Память под активации падает втрое-вчетверо ценой примерно 30% скорости.
        На видеокарте с 6 ГБ это единственный способ держать батч 64: batch-hard
        triplet ищет самые трудные пары ВНУТРИ батча, и на маленьком батче
        по-настоящему трудных пар просто не оказывается.

        Альтернатива — уменьшить батч — дешевле по коду, но бьёт прямо по тому
        механизму, ради которого triplet и используется.
        """
        self._checkpointing = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        if getattr(self, "_checkpointing", False) and self.training and x.requires_grad:
            # use_reentrant=False — вариант, корректно работающий с BatchNorm
            # и не требующий, чтобы все входы имели requires_grad.
            for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
                x = checkpoint(layer, x, use_reentrant=False)
            return x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class GeM(nn.Module):
    """Generalized Mean pooling с обучаемой степенью p.

    p -> 1 даёт average pooling, p -> inf даёт max pooling. Модель сама находит
    компромисс: для машин обычно сходится к p ≈ 3, то есть ближе к max.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Степень считается в float32 и при инференсе в fp16. Потолок fp16 —
        # 65504, а обучаемая p за 140 эпох выросла до 3.95: активация 16.6 в
        # такой степени уже бесконечна. Ночной перебор ансамблей упал именно
        # на этом (NaN в признаках), хотя валидация при обучении была чистой:
        # там работает autocast, который сам считает pow в float32.
        # Результат после корня снова порядка самих активаций и в fp16 влезает;
        # тип выхода берём у параметра, то есть у остальной модели.
        pooled = F.avg_pool2d(x.float().clamp(min=self.eps).pow(self.p.float()),
                              (x.size(-2), x.size(-1)))
        return pooled.pow(1.0 / self.p.float()).flatten(1).to(self.p.dtype)


def weights_init_kaiming(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Linear" in name:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif "BatchNorm" in name and module.affine:
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class VehicleReID(nn.Module):
    """Полная модель: backbone -> GeM -> BNNeck -> (признак, логиты).

    На инференсе возвращается только признак после BN — именно он идёт в
    embeddings.npy и сравнивается косинусом.
    """

    ARCHITECTURE = "resnet50-ibn-a-bnneck-gem-v1"

    def __init__(self, num_classes: int = 0, embedding_dim: int = 2048,
                 last_stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.backbone = ResNetIBN(last_stride=last_stride)
        self.pool = GeM()

        self.reduce: nn.Module = nn.Identity()
        feature_dim = self.backbone.out_channels
        if embedding_dim and embedding_dim != feature_dim:
            # Понижение размерности удешевляет поиск и хранение в pgvector.
            self.reduce = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
            feature_dim = embedding_dim
        self.feature_dim = feature_dim

        # BNNeck: bias отключён, чтобы признак оставался центрированным.
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

    def forward(self, images: torch.Tensor):
        pooled = self.pool(self.backbone(images))
        triplet_feature = self.reduce(pooled)          # до BN — для triplet loss
        inference_feature = self.bottleneck(triplet_feature)  # после BN — для поиска

        if not self.training or self.classifier is None:
            return F.normalize(inference_feature, dim=1)

        logits = self.classifier(self.dropout(inference_feature))
        return triplet_feature, logits

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.backbone.set_gradient_checkpointing(enabled)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Признак для поиска: L2-нормированный вектор после BNNeck."""
        was_training = self.training
        self.eval()
        try:
            return F.normalize(self.bottleneck(self.reduce(self.pool(self.backbone(images)))), dim=1)
        finally:
            self.train(was_training)


def load_pretrained_backbone(model: VehicleReID, weights_path: Path | str | None = None,
                             verbose: bool = True) -> dict:
    """Грузит ImageNet+IBN веса в backbone. Классификационная голова не нужна.

    Возвращает отчёт о том, что реально совпало: молчаливая загрузка «нуля»
    совпавших слоёв — самая дорогая ошибка в такой сборке.
    """
    if weights_path is None:
        state = torch.hub.load_state_dict_from_url(IBN_WEIGHTS_URL, map_location="cpu", progress=verbose)
    else:
        path = Path(weights_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if IBN_WEIGHTS_SHA256_PREFIX not in path.name and verbose:
            print(f"[falcon] локальные веса {path.name}, sha256={digest[:16]}")

    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # fc.* — голова ImageNet на 1000 классов, нам не нужна.
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}

    result = model.backbone.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    report = {
        "loaded_tensors": loaded,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }
    if loaded == 0:
        raise RuntimeError("Ни один слой backbone не совпал с чекпоинтом — веса не те")
    if verbose:
        print(f"[falcon] загружено {loaded} тензоров backbone, "
              f"пропущено {len(result.missing_keys)}, лишних {len(result.unexpected_keys)}")
    return report


def build_model(num_classes: int = 0, embedding_dim: int = 2048, pretrained: bool = True,
                weights_path: Path | str | None = None, last_stride: int = 1,
                dropout: float = 0.0, verbose: bool = True) -> VehicleReID:
    model = VehicleReID(num_classes=num_classes, embedding_dim=embedding_dim,
                        last_stride=last_stride, dropout=dropout)
    if pretrained:
        load_pretrained_backbone(model, weights_path, verbose=verbose)
    return model


# ---------------------------------------------------------------------------
# CLIP ViT-B/16: вторая архитектура ансамбля
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ViTReID(nn.Module):
    """Image-энкодер CLIP ViT-B/16 -> BNNeck -> (признак, логиты).

    Зачем вторая архитектура. Наши ResNet50 предобучены на ImageNet (1.3 млн
    картинок), а CLIP — на 400 млн пар «картинка — текст». Проба без
    дообучения на отложенной выборке: mAP@10 0.110 у CLIP против 0.061 у
    ResNet50-IBN и 0.055–0.064 у DINOv2. На этом же наблюдении построен
    CLIP-ReID — один из сильнейших методов поиска машин. Вдобавок модель другой
    архитектуры ошибается иначе, и в ансамбле это ценнее ещё одной ResNet.

    Интерфейс тот же, что у VehicleReID: на обучении — (признак до BN, логиты),
    на инференсе — L2-нормированный признак после BNNeck. Препроцессинг у всех
    участников ансамбля общий (256x256, нормализация ImageNet), поэтому
    модель сама переводит вход в нормализацию CLIP — точно и почти бесплатно.
    """

    ARCHITECTURE = "clip-vit-b16-bnneck-v1"
    BACKBONE = "vit_base_patch16_clip_224.openai"

    def __init__(self, num_classes: int = 0, img_size: int = 256, pretrained: bool = False):
        super().__init__()
        import timm  # только для этой архитектуры; ResNet без timm работает

        # pretrained=False не ходит в сеть: архитектура описана в коде timm,
        # веса приходят из нашего чекпоинта. Сеть нужна только при старте
        # обучения с весов CLIP.
        self.backbone = timm.create_model(self.BACKBONE, pretrained=pretrained,
                                          num_classes=0, img_size=img_size)
        feature_dim = self.backbone.num_features
        self.feature_dim = feature_dim

        scale = torch.tensor(IMAGENET_STD) / torch.tensor(CLIP_STD)
        shift = (torch.tensor(IMAGENET_MEAN) - torch.tensor(CLIP_MEAN)) / torch.tensor(CLIP_STD)
        self.register_buffer("input_scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_shift", shift.view(1, 3, 1, 1), persistent=False)

        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

    def forward(self, images: torch.Tensor):
        images = images * self.input_scale + self.input_shift
        triplet_feature = self.backbone(images)            # CLS-токен после LayerNorm
        inference_feature = self.bottleneck(triplet_feature)
        if not self.training or self.classifier is None:
            return F.normalize(inference_feature, dim=1)
        return triplet_feature, self.classifier(inference_feature)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.backbone.set_grad_checkpointing(enabled)

    def head_parameters(self):
        """Параметры поверх энкодера: им нужна скорость обучения выше."""
        yield from self.bottleneck.parameters()
        if self.classifier is not None:
            yield from self.classifier.parameters()


ARCHITECTURES = {
    VehicleReID.ARCHITECTURE: "resnet50-ibn",
    ViTReID.ARCHITECTURE: "clip-vit-b16",
}


def build_named(name: str, num_classes: int = 0, pretrained: bool = True,
                embedding_dim: int = 2048, verbose: bool = True) -> nn.Module:
    """Модель по короткому имени архитектуры: resnet50-ibn | clip-vit-b16."""
    if name == "clip-vit-b16":
        return ViTReID(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet50-ibn":
        return build_model(num_classes=num_classes, embedding_dim=embedding_dim,
                           pretrained=pretrained, verbose=verbose)
    raise ValueError(f"Неизвестная архитектура {name!r}")


def build_for_checkpoint(architecture: str, num_classes: int, feature_dim: int) -> nn.Module:
    """Пустая модель под чекпоинт: без загрузки предобученных весов и без сети."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Чекпоинт собран неизвестной архитектурой: {architecture!r}")
    return build_named(ARCHITECTURES[architecture], num_classes=num_classes,
                       pretrained=False, embedding_dim=feature_dim, verbose=False)
