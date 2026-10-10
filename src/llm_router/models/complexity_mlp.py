"""Small feed-forward classifier over cached query features (requires torch).

Default: Linear(d, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 128) -> ReLU -> Dropout(0.1)
-> Linear(128, 3) logits, in `LABELS` order.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] = (256, 128),
    dropouts: Sequence[float] = (0.2, 0.1),
    n_classes: int = 3,
) -> nn.Sequential:
    if len(hidden_dims) != len(dropouts):
        raise ValueError("need one dropout per hidden layer")
    layers: list[nn.Module] = []
    previous = input_dim
    for width, p in zip(hidden_dims, dropouts, strict=True):
        layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(p)]
        previous = width
    layers.append(nn.Linear(previous, n_classes))
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
