"""Ciclos de entrenamiento y validación."""
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Una época de entrenamiento. Devuelve (loss_promedio, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                 # 1. reinicia gradientes (PyTorch los acumula)
        outputs = model(inputs)               # 2. forward pass
        loss = criterion(outputs, targets)    # 3. pérdida
        loss.backward()                       # 4. backward pass (autograd)
        optimizer.step()                      # 5. actualización de pesos (Adam)

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evalúa en datos no vistos. Devuelve (loss, accuracy, f1)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    f1 = f1_score(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy())
    return running_loss / total, correct / total, f1
