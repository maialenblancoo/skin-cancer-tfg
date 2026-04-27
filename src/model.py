import torch
import torch.nn as nn
import timm
from src.config import NUM_CLASSES, BACKBONE


class ImageBranch(nn.Module):
    """
    Rama de imagen basada en EfficientNet-B0 preentrenado.
    Devuelve un vector de características de 256 dimensiones.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            num_classes=0,   # quitar la cabeza de clasificación original
        )
        in_features = self.backbone.num_features  # 1280 para efficientnet_b0

        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)


class MetadataBranch(nn.Module):
    """
    Rama MLP para metadatos tabulares.
    Devuelve un vector de 64 dimensiones.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class SkinLesionModel(nn.Module):
    """
    Modelo multimodal con fusión tardía (late fusion).

    Si metadata_dim == 0 -> modelo unimodal (solo imagen).
    Si metadata_dim  > 0 -> modelo multimodal (imagen + metadatos).
    """
    def __init__(self, metadata_dim: int = 0):
        super().__init__()
        self.metadata_dim  = metadata_dim
        self.image_branch  = ImageBranch()

        if metadata_dim > 0:
            self.metadata_branch = MetadataBranch(metadata_dim)
            fusion_dim = 256 + 64
        else:
            self.metadata_branch = None
            fusion_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, image, metadata=None):
        img_features = self.image_branch(image)

        if self.metadata_branch is not None and metadata is not None:
            meta_features = self.metadata_branch(metadata)
            features = torch.cat([img_features, meta_features], dim=1)
        else:
            features = img_features

        return self.classifier(features)