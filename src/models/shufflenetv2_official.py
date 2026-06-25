from __future__ import annotations

from torch import nn
from torchvision import models


def build_shufflenet_v2_x1_0(num_classes: int, use_pretrained: bool = False) -> nn.Module:
    """
    Build the official torchvision ShuffleNetV2 x1.0.

    Project-side adaptation:
    - Keep the official backbone unchanged.
    - Replace only the final classification layer so it matches the current dataset class count.
    """
    weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if use_pretrained else None
    model = models.shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
