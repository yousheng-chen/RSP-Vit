from __future__ import annotations

from torch import nn
from torchvision import models


def build_mnasnet1_0(num_classes: int, use_pretrained: bool = False) -> nn.Module:
    """
    Build the official torchvision MnasNet1.0.

    Project-side adaptation:
    - Keep the official backbone unchanged.
    - Replace only the final classifier layer so it matches the current dataset class count.
    """
    weights = models.MNASNet1_0_Weights.DEFAULT if use_pretrained else None
    model = models.mnasnet1_0(weights=weights)
    last_linear = model.classifier[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected the last MnasNet1.0 classifier layer to be nn.Linear.")
    model.classifier[-1] = nn.Linear(last_linear.in_features, num_classes)
    return model
