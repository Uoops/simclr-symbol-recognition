"""Wrapper around the SimCLR repo's model for fine-tuning on symbols.

Loads a pretrained SimCLR checkpoint (contrastive-learned on STL10) and
fine-tunes it on the Bobyard symbol dataset.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add SimCLR repo to path
sys.path.insert(0, str(Path(__file__).parent / "SimCLR"))
from models.resnet_simclr import ResNetSimCLR

# Default pretrained SimCLR checkpoint (ResNet-18 trained on STL10 for 100 epochs)
DEFAULT_CHECKPOINT = str(
    Path(__file__).parent / "pretrained" / "simclr_stl10" / "checkpoint_0100.pth.tar"
)


def create_simclr_model(base_model="resnet18", out_dim=128, checkpoint_path=None):
    """Create a SimCLR model and load pretrained SimCLR weights.

    Args:
        base_model: Backbone architecture ("resnet18" or "resnet50").
        out_dim: Projection head output dimension (must match checkpoint).
        checkpoint_path: Path to a SimCLR checkpoint. If None, uses the
            default STL10-pretrained ResNet-18 checkpoint.

    Returns:
        ResNetSimCLR model with pretrained SimCLR weights loaded.
    """
    model = ResNetSimCLR(base_model=base_model, out_dim=out_dim)

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model.load_state_dict(state_dict)
    print(f"Loaded pretrained SimCLR checkpoint from {checkpoint_path}")
    print(f"  arch={ckpt.get('arch')}, epoch={ckpt.get('epoch')}, "
          f"keys={len(state_dict)}")

    return model


def get_features(model, x):
    """Extract features before the projection head.

    The ResNetSimCLR model has backbone.fc = Sequential(Linear, ReLU, Linear).
    We need the representation BEFORE fc for downstream evaluation.
    """
    original_fc = model.backbone.fc
    model.backbone.fc = nn.Identity()
    with torch.no_grad():
        features = model.backbone(x)
    model.backbone.fc = original_fc
    return features
