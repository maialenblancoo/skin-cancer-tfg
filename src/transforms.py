import torchvision.transforms as T
from src.config import IMAGE_SIZE


def get_transforms(split: str):
    """
    Devuelve las transformaciones según la fase.

    split:
        "train" -> augmentations + normalización
        "val"   -> solo resize + normalización
        "test"  -> solo resize + normalización
    """

    # Normalización estándar ImageNet (usada por EfficientNet)
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            normalize,
        ])

    else:  # val y test
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            normalize,
        ])