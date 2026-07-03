import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.data import EXPECTED_CLASSES, count_images_by_class, load_datasets


def save_class_counts(datasets_by_split, output_path: Path):
    counts = {split: count_images_by_class(dataset) for split, dataset in datasets_by_split.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(counts, file, indent=2)
    return counts


def save_sample_grid(dataset, output_path: Path, samples_per_class: int = 3, seed: int = 42):
    rng = random.Random(seed)
    samples_by_class = {class_name: [] for class_name in EXPECTED_CLASSES}
    for path, label in dataset.samples:
        class_name = dataset.classes[label]
        samples_by_class[class_name].append(path)

    fig, axes = plt.subplots(
        len(EXPECTED_CLASSES),
        samples_per_class,
        figsize=(3 * samples_per_class, 3 * len(EXPECTED_CLASSES)),
    )

    for row, class_name in enumerate(EXPECTED_CLASSES):
        chosen_paths = rng.sample(samples_by_class[class_name], k=samples_per_class)
        for col, path in enumerate(chosen_paths):
            image = Image.open(path).convert("L")
            ax = axes[row][col]
            ax.imshow(image, cmap="gray")
            ax.set_title(class_name)
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Confere a estrutura do dataset, conta imagens por classe e gera exemplos."
    )
    parser.add_argument("--data-dir", default="data/chest_xray")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    datasets_by_split = load_datasets(args.data_dir)
    print(f"Estrutura de pastas OK. Classes encontradas: {EXPECTED_CLASSES}")

    counts = save_class_counts(datasets_by_split, output_dir / "dataset_class_counts.json")
    for split, split_counts in counts.items():
        total = sum(split_counts.values())
        print(f"{split}: {split_counts} (total: {total})")

    sample_path = output_dir / "dataset_sample_images.png"
    save_sample_grid(
        datasets_by_split["train"],
        sample_path,
        samples_per_class=args.samples_per_class,
    )
    print(f"Contagem por classe salva em: {output_dir / 'dataset_class_counts.json'}")
    print(f"Exemplos de radiografias salvos em: {sample_path}")


if __name__ == "__main__":
    main()
