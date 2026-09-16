"""Local ResNet18 ReID encoder; inference never downloads weights."""
import hashlib
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


def preprocessing(training=False):
    steps = [transforms.Resize((160, 256))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(.3, .3, .2, .05)]
    steps += [transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])]
    if training:
        steps += [transforms.RandomErasing(p=.4)]
    return transforms.Compose(steps)


class ReID(nn.Module):
    def __init__(self, classes, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, classes)

    def forward(self, images):
        features = self.backbone(images)
        return F.normalize(features, dim=1), self.classifier(features)


def batch_hard_loss(features, labels, margin=.3):
    distances = torch.cdist(features, features)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return features.sum() * 0
    hardest_positive = distances.masked_fill(~positive, -1).max(1).values
    hardest_negative = distances.masked_fill(~negative, float('inf')).min(1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


class NeuralEncoder:
    def __init__(self, checkpoint, device="cpu"):
        path = Path(checkpoint)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("architecture") != "resnet18-reid-v1":
            raise ValueError("Unsupported model checkpoint")
        self.model = ReID(state["classes"])
        self.model.load_state_dict(state["model"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.transform = preprocessing()
        self.preprocessing_version = state.get('preprocessing', 'torchvision-bilinear-v1')
        if self.preprocessing_version not in ('torchvision-bilinear-v1', 'pil-bicubic-256x160-v1'):
            raise ValueError('Unsupported checkpoint preprocessing')
        self.version = "reid-" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def __call__(self, image):
        if self.preprocessing_version == 'pil-bicubic-256x160-v1':
            image = image.resize((256, 160), Image.Resampling.BICUBIC)
        with torch.inference_mode():
            result, _ = self.model(self.transform(image).unsqueeze(0).to(self.device))
            return result[0].cpu().numpy()
