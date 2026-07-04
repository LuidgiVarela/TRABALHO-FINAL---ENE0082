import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """CNN convolucional simples, treinada do zero (sem pesos pre-treinados)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_simple_cnn(num_classes: int = 2):
    return SimpleCNN(num_classes=num_classes)


def build_resnet18(num_classes: int = 2, freeze_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_densenet121(num_classes: int = 2, freeze_backbone: bool = False):
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def build_transfer_model(
    model_name: str,
    num_classes: int = 2,
    freeze_backbone: bool = False,
):
    builders = {
        "resnet18": build_resnet18,
        "densenet121": build_densenet121,
    }
    if model_name not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(f"Modelo desconhecido: {model_name}. Opcoes: {available}")

    return builders[model_name](
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )
