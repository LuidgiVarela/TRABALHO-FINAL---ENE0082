import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import build_dataloaders, count_images_by_class
from src.evaluate import compute_metrics, plot_confusion_matrix, predict, save_metrics
from src.models import build_resnet18


MODEL_NAME = "ResNet18 Transfer Learning"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Treino", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_loss_accuracy(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validacao", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_training_history(history: list[dict], output_dir: Path):
    history_path = output_dir / "transfer_training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    plt.plot(epochs, [row["train_loss"] for row in history], label="Loss treino")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Loss validacao")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Historico de perda - Transfer Learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_loss_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Acuracia treino")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="Acuracia validacao")
    plt.plot(epochs, [row["val_f1_score"] for row in history], label="F1 validacao")
    plt.xlabel("Epoca")
    plt.ylabel("Metrica")
    plt.title("Historico de metricas - Transfer Learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_metrics_history.png", dpi=150)
    plt.close()

    return history_path


def save_run_config(args, output_dir: Path, class_names: list[str], device: torch.device):
    config = {
        "model": MODEL_NAME,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_workers": args.num_workers,
        "freeze_backbone": args.freeze_backbone,
        "classes": class_names,
        "device": str(device),
    }
    with (output_dir / "transfer_run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Treina ResNet18 com Transfer Learning.")
    parser.add_argument("--data-dir", default="data/chest_xray")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_by_split, dataloaders = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Dispositivo: {device}")
    print(f"Classes: {datasets_by_split['train'].classes}")
    for split, dataset in datasets_by_split.items():
        print(f"{split}: {count_images_by_class(dataset)}")

    save_run_config(args, output_dir, datasets_by_split["train"].classes, device)

    model = build_resnet18(
        num_classes=len(datasets_by_split["train"].classes),
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    best_val_f1 = -1.0
    best_checkpoint = output_dir / "resnet18_transfer_best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_loss_accuracy(
            model, dataloaders["val"], criterion, device
        )
        val_true, val_pred = predict(model, dataloaders["val"], device)
        val_metrics = compute_metrics(val_true, val_pred, model_name=MODEL_NAME)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1_score": val_metrics["f1_score"],
            }
        )

        print(
            f"Epoca {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), best_checkpoint)

    history_path = save_training_history(history, output_dir)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_true, test_pred = predict(model, dataloaders["test"], device)
    test_metrics = compute_metrics(
        test_true,
        test_pred,
        model_name=MODEL_NAME,
    )

    save_metrics(test_metrics, output_dir / "transfer_metrics.json")
    plot_confusion_matrix(
        test_true,
        test_pred,
        datasets_by_split["test"].classes,
        output_dir / "transfer_confusion_matrix.png",
    )

    print("Metricas finais no teste:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
    print(f"Historico de treino salvo em: {history_path}")


if __name__ == "__main__":
    main()
