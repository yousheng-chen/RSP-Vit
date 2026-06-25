from __future__ import annotations

from torch import nn
from torchvision import models


def build_googlenet(num_classes: int, use_pretrained: bool = False) -> nn.Module:
    """
    Build the official torchvision GoogLeNet.

    Project-side adaptation:
    - Keep the official backbone unchanged.
    - Disable auxiliary classifier heads so the model returns a single tensor.
    - Replace only the final classifier layer so it matches the current dataset class count.
    """
    weights = models.GoogLeNet_Weights.DEFAULT if use_pretrained else None
    model = models.googlenet(weights=weights, aux_logits=False)
    if not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected GoogLeNet.fc to be nn.Linear.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
