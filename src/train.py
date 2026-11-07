"""Script principal: entrena el clasificador base y registra métricas por época.

Uso (desde la raíz del repo):
    python -m src.train --epochs 30 --lr 0.001
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import make_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import BaselineClassifier
from src.utils import get_device, seed_everything

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline base de entrenamiento en PyTorch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="Train")
    ax[0].plot(epochs, history["val_loss"], label="Validación")
    ax[0].set(title="Curva de pérdida", xlabel="Época", ylabel="CrossEntropy")
    ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="Train")
    ax[1].plot(epochs, history["val_acc"], label="Validación")
    ax[1].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"PyTorch {torch.__version__} | dispositivo: {device}")

    train_loader, val_loader, input_dim = make_dataloaders(
        batch_size=args.batch_size, seed=args.seed
    )
    model = BaselineClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    RESULTS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_f1]):
            history[k].append(v)

        if va_loss < best_val_loss:  # guarda el mejor checkpoint según pérdida de validación
            best_val_loss = va_loss
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")

        print(
            f"Época [{epoch:02d}/{args.epochs}] | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% F1: {va_f1:.4f}"
        )

    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump({"args": vars(args), "torch": torch.__version__, "history": history}, f, indent=2)
    plot_history(history, RESULTS_DIR / "loss_curve.png")
    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
