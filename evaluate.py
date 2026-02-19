"""Few-shot evaluation for SimCLR embeddings."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from dataset import SymbolDataset, get_eval_transform
from simclr_model import get_features


@torch.no_grad()
def extract_embeddings(model, dataloader, device="cuda"):
    """Extract backbone features (before projection head) and labels."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        embeddings = get_features(model, imgs)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return embeddings, labels


def knn_evaluate(train_embeddings, train_labels, test_embeddings, test_labels, k=1):
    """k-NN evaluation on embeddings."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, predictions


def few_shot_evaluate(model, data_dir, device="cuda", img_size=64,
                      n_shots=1, k_neighbors=1, batch_size=128, n_trials=10):
    """Run a few-shot evaluation averaged over multiple trials.

    For each class, sample n_shots examples as the support set,
    then evaluate on the remaining examples (test set).
    Repeats n_trials times and returns the mean accuracy.
    """
    eval_transform = get_eval_transform(img_size)

    train_ds = SymbolDataset(data_dir, split="train", transform=eval_transform)
    test_ds = SymbolDataset(data_dir, split="test", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)

    if n_shots > 0:
        accuracies = []
        for _ in range(n_trials):
            support_indices = []
            for cls in np.unique(train_labels):
                cls_indices = np.where(train_labels == cls)[0]
                n = min(n_shots, len(cls_indices))
                chosen = np.random.choice(cls_indices, size=n, replace=False)
                support_indices.extend(chosen)
            support_indices = np.array(support_indices)
            support_emb = train_emb[support_indices]
            support_labels = train_labels[support_indices]

            acc, _ = knn_evaluate(
                support_emb, support_labels, test_emb, test_labels, k=k_neighbors
            )
            accuracies.append(acc)
        accuracy = np.mean(accuracies)
    else:
        support_emb = train_emb
        support_labels = train_labels
        accuracy, _ = knn_evaluate(
            support_emb, support_labels, test_emb, test_labels, k=k_neighbors
        )

    return {
        "accuracy": accuracy,
        "test_labels": test_labels,
        "test_embeddings": test_emb,
    }


def full_evaluate(model, data_dir, device="cuda", img_size=64, batch_size=128):
    """Evaluate using all training data as support (upper bound)."""
    return few_shot_evaluate(
        model, data_dir, device, img_size,
        n_shots=0, k_neighbors=5, batch_size=batch_size,
    )
