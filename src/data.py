from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]


def build_transforms(train: bool = False):
    """Cria transformacoes compativeis com modelos pre-treinados no ImageNet."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_datasets(data_dir: str | Path):
    data_path = Path(data_dir)
    expected_splits = ["train", "val", "test"]
    missing = [split for split in expected_splits if not (data_path / split).exists()]
    if missing:
        raise FileNotFoundError(
            f"Pastas ausentes em {data_path}: {', '.join(missing)}. "
            "A estrutura esperada e train/val/test com NORMAL e PNEUMONIA."
        )

    datasets_by_split = {
        "train": datasets.ImageFolder(data_path / "train", transform=build_transforms(train=True)),
        "val": datasets.ImageFolder(data_path / "val", transform=build_transforms(train=False)),
        "test": datasets.ImageFolder(data_path / "test", transform=build_transforms(train=False)),
    }

    for split, dataset in datasets_by_split.items():
        if dataset.classes != EXPECTED_CLASSES:
            raise ValueError(
                f"Classes inesperadas em {split}: {dataset.classes}. "
                f"O esperado e {EXPECTED_CLASSES}, nessa ordem."
            )

    return datasets_by_split


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
):
    datasets_by_split = load_datasets(data_dir)

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return datasets_by_split, dataloaders


def count_images_by_class(dataset):
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    return counts


class ImageListDataset(Dataset):
    """Dataset a partir de uma lista (path, label) fixa, com classes/samples no
    mesmo formato do ImageFolder para reaproveitar count_images_by_class etc."""

    def __init__(self, samples: list[tuple[str, int]], classes: list[str], transform):
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_custom_split_dataloaders(
    data_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Junta o train+val oficiais e faz um novo split estratificado train/val.

    O split oficial do dataset tem apenas 16 imagens de validacao (8/8), o que torna
    a selecao do melhor checkpoint por F1 de validacao muito ruidosa. Esta funcao
    mantem o `test` oficial intocado (mesmo benchmark usado pela Pessoa 2) e apenas
    redistribui train/val, preservando a proporcao das classes.
    """
    data_path = Path(data_dir)
    raw_train = datasets.ImageFolder(data_path / "train")
    raw_val = datasets.ImageFolder(data_path / "val")

    if raw_train.classes != EXPECTED_CLASSES or raw_val.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Classes inesperadas. O esperado e {EXPECTED_CLASSES}, nessa ordem."
        )

    combined_samples = raw_train.samples + raw_val.samples
    labels = [label for _, label in combined_samples]

    train_samples, val_samples = train_test_split(
        combined_samples,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )

    datasets_by_split = {
        "train": ImageListDataset(train_samples, EXPECTED_CLASSES, build_transforms(train=True)),
        "val": ImageListDataset(val_samples, EXPECTED_CLASSES, build_transforms(train=False)),
        "test": datasets.ImageFolder(data_path / "test", transform=build_transforms(train=False)),
    }

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            datasets_by_split["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            datasets_by_split["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    return datasets_by_split, dataloaders
