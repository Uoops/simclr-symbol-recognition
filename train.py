"""Training script for fine-tuning SimCLR on symbol crops.

Uses the SimCLR repo's training infrastructure (info_nce_loss, training loop)
adapted for the Bobyard symbol dataset with pretrained backbone weights.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SymbolDataset, UniqueClassBatchSampler, get_simclr_transform
from evaluate import few_shot_evaluate
from simclr_model import create_simclr_model


def info_nce_loss(features, batch_size, n_views=2, temperature=0.07, device="cuda"):
    """Compute Info NCE loss — adapted from SimCLR repo's simclr.py.

    This is the same loss function from the repo, extracted as a standalone
    function for flexibility.
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both labels and similarity matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives and negatives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    temperature=0.07, n_views=2):
    """Train one epoch using the SimCLR repo's training pattern."""
    model.train()
    total_loss = 0
    n_batches = 0

    for images, _ in dataloader:
        # images is a list of n_views tensors (from ContrastiveLearningViewGenerator)
        images = torch.cat(images, dim=0).to(device)

        features = model(images)
        logits, labels = info_nce_loss(
            features, batch_size=len(images) // n_views,
            n_views=n_views, temperature=temperature, device=device,
        )
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def run_experiment(config, data_dir, output_dir, device="cuda"):
    """Run a single fine-tuning experiment with given config."""
    exp_name = config["name"]
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}\n")

    img_size = config.get("img_size", 64)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    temperature = config.get("temperature", 0.07)
    epochs = config.get("epochs", 50)
    backbone = config.get("backbone", "resnet18")
    aug_strength = config.get("aug_strength", 1.0)
    out_dim = config.get("out_dim", 128)
    n_views = config.get("n_views", 2)

    # Dataset with SimCLR view generator (produces n_views augmented copies)
    transform = get_simclr_transform(img_size, n_views=n_views, s=aug_strength)
    train_ds = SymbolDataset(data_dir, split="train", transform=transform)

    # Use UniqueClassBatchSampler: every sample in a batch from a different class
    batch_sampler = UniqueClassBatchSampler(train_ds, batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Using UniqueClassBatchSampler (batch_size={batch_size}, "
          f"{train_ds.num_classes} classes, 0 same-class collisions)")

    # Model: ResNetSimCLR loaded from pretrained SimCLR checkpoint (STL10)
    checkpoint_path = config.get("checkpoint_path", None)
    model = create_simclr_model(
        base_model=backbone, out_dim=out_dim, checkpoint_path=checkpoint_path
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR scheduler: cosine decay or none (constant LR)
    lr_scheduler = config.get("lr_scheduler", "cosine")
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0, last_epoch=-1
        )
    elif lr_scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    # Training loop
    history = {"loss": [], "lr": [], "eval_1shot": [], "eval_5shot": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            temperature=temperature, n_views=n_views,
        )
        # Skip first 10 epochs as warmup (same as SimCLR repo)
        if scheduler is not None and epoch >= 10:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Evaluate every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == epochs:
            result_1shot = few_shot_evaluate(
                model, data_dir, device, img_size, n_shots=1, k_neighbors=1
            )
            result_5shot = few_shot_evaluate(
                model, data_dir, device, img_size, n_shots=5, k_neighbors=3
            )
            history["eval_1shot"].append({
                "epoch": epoch, "accuracy": result_1shot["accuracy"]
            })
            history["eval_5shot"].append({
                "epoch": epoch, "accuracy": result_5shot["accuracy"]
            })
            print(f"  -> 1-shot acc: {result_1shot['accuracy']:.4f} | "
                  f"5-shot acc: {result_5shot['accuracy']:.4f}")

    # Save checkpoint (same format as SimCLR repo)
    checkpoint_name = f"checkpoint_{epochs:04d}.pth.tar"
    torch.save({
        "epoch": epochs,
        "arch": backbone,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, exp_dir / checkpoint_name)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to {exp_dir}")
    return history, model


# Experiment configurations — varying key hyperparameters (all use img_size=224)
EXPERIMENTS = {
    "baseline": {
        "name": "baseline",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
    },
    "high_temp": {
        "name": "high_temp",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.5,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
    },
    "low_temp": {
        "name": "low_temp",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.01,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
    },
    "strong_aug": {
        "name": "strong_aug",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "epochs": 50,
        "aug_strength": 1.5,
        "out_dim": 128,
    },
    "no_scheduler": {
        "name": "no_scheduler",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
        "lr_scheduler": "none",
    },
    "high_wd": {
        "name": "high_wd",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "temperature": 0.07,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
    },
    "low_wd": {
        "name": "low_wd",
        "backbone": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "epochs": 50,
        "aug_strength": 1.0,
        "out_dim": 128,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SimCLR on symbol dataset")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--experiments", nargs="+", default=["baseline"],
                        choices=list(EXPERIMENTS.keys()) + ["all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if "all" in args.experiments:
        exp_list = list(EXPERIMENTS.keys())
    else:
        exp_list = args.experiments

    all_results = {}
    for exp_name in exp_list:
        config = EXPERIMENTS[exp_name]
        history, model = run_experiment(config, args.data_dir, args.output_dir, args.device)
        all_results[exp_name] = history

    output_path = Path(args.output_dir) / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")
