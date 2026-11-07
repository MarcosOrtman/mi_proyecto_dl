"""Arquitectura base: MLP de pocas capas con nn.Sequential."""
import torch
import torch.nn as nn


class BaselineClassifier(nn.Module):
    """MLP: input -> 64 -> ReLU -> 32 -> ReLU -> num_classes (logits)."""

    def __init__(self, input_dim: int, hidden_dims=(64, 32), num_classes: int = 2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))  # logits: CrossEntropyLoss aplica softmax internamente
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
