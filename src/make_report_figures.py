from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results")
REPORT_DIR = RESULTS_DIR / "report_figures"
CLASSES = ["NORMAL", "PNEUMONIA"]


CONFUSION_MATRICES = {
    "cnn_simples": {
        "title": "Matriz de confusão - CNN simples",
        "matrix": [[195, 39], [38, 352]],
    },
    "resnet18": {
        "title": "Matriz de confusão - ResNet18",
        "matrix": [[195, 39], [20, 370]],
    },
    "densenet121": {
        "title": "Matriz de confusão - DenseNet121",
        "matrix": [[178, 56], [23, 367]],
    },
}


HISTORY_FILES = {
    "resnet18": {
        "title": "ResNet18 Transfer Learning",
        "path": RESULTS_DIR / "transfer_frozen_modelo" / "transfer_training_history.csv",
    },
    "densenet121": {
        "title": "DenseNet121 Transfer Learning",
        "path": RESULTS_DIR / "transfer_densenet121_modelo" / "transfer_training_history.csv",
    },
}


def save_confusion_matrix(name: str, title: str, matrix: list[list[int]]) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_history_plots(name: str, title: str, path: Path) -> None:
    history = pd.read_csv(path)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Loss treino")
    plt.plot(history["epoch"], history["val_loss"], label="Loss validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Histórico de perda - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{name}_loss_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_accuracy"], label="Acurácia treino")
    plt.plot(history["epoch"], history["val_accuracy"], label="Acurácia validação")
    plt.plot(history["epoch"], history["val_f1_score"], label="F1 validação")
    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.title(f"Histórico de métricas - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{name}_metrics_history.png", dpi=150)
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in CONFUSION_MATRICES.items():
        save_confusion_matrix(name, config["title"], config["matrix"])

    for name, config in HISTORY_FILES.items():
        save_history_plots(name, config["title"], config["path"])


if __name__ == "__main__":
    main()
