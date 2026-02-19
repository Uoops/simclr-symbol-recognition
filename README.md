# SimCLR Fine-tuning for Symbol Recognition

Fine-tunes a pretrained [SimCLR](https://github.com/sthalles/SimCLR) model to recognize symbols in engineering/technical drawings using contrastive learning, enabling few-shot (1-shot or 5-shot) symbol classification via k-NN on learned embeddings.

## Problem

Given a dataset of engineering drawings annotated with 42 symbol classes (valves, detectors, sprinklers, etc.), build a model that can recognize symbols with as few as 1 example per class.

## Approach

### Pipeline

1. **Pretrained backbone**: Start from a SimCLR ResNet-18 checkpoint pretrained on STL10 with contrastive learning (100 epochs)
2. **Fine-tune on symbols**: Continue contrastive training on cropped symbol patches from the drawing dataset
3. **Evaluate via k-NN**: Extract 512-dim backbone features, classify test symbols by finding nearest neighbors in the training set

### Key Design Decisions

- **Image size 224x224**: ResNet's native input resolution. Preserves fine details in technical symbols (thin lines, small text) that are lost at smaller sizes.
- **UniqueClassBatchSampler**: Each batch of 32 samples contains symbols from 32 different classes (out of 42 total). This avoids the "false negative" problem in SimCLR where same-class samples in a batch are incorrectly treated as negatives.
- **No backbone freezing**: Since evaluation features come from the backbone itself, freezing it would make fine-tuning pointless. All layers are trainable.
- **Info NCE loss**: The standard SimCLR contrastive loss. Each symbol crop is augmented twice (2 views), and the model learns to match the two views of the same crop while pushing apart views of different crops.

## Dataset

| Split | Crops | Source |
|-------|-------|--------|
| Train | 2,606 | COCO-annotated engineering drawings |
| Valid | 755 | |
| Test | 424 | |

- 42 symbol classes (e.g., `24V-power-cord`, `smoke-detector`, `sprinkler-upright`)
- Highly imbalanced: most common class has 940 samples, rarest has 4
- Crops are extracted from full drawing images using COCO bounding box annotations with 10% padding

## Experiments

All experiments use: ResNet-18 backbone, img_size=224, batch_size=32, 50 epochs, out_dim=128.

| Experiment | Temperature | LR Scheduler | Weight Decay | Aug Strength | 1-shot | 5-shot | Full kNN-5 |
|---|---|---|---|---|---|---|---|
| baseline | 0.07 | cosine | 1e-4 | 1.0 | 78.5% | 89.4% | 99.3% |
| high_temp | 0.5 | cosine | 1e-4 | 1.0 | 79.7% | 91.3% | 99.1% |
| low_temp | 0.01 | cosine | 1e-4 | 1.0 | 77.8% | 87.5% | 99.3% |
| strong_aug | 0.07 | cosine | 1e-4 | 1.5 | 77.6% | 86.8% | 99.3% |
| no_scheduler | 0.07 | none | 1e-4 | 1.0 | **85.4%** | 89.2% | 99.3% |
| high_wd | 0.07 | cosine | 1e-3 | 1.0 | 75.2% | 88.9% | 99.3% |
| low_wd | 0.07 | cosine | 1e-5 | 1.0 | 80.2% | 90.6% | 99.3% |

### What We Learned

**Fine-tuning vs pretrained**: The pretrained STL10 model scores 35.6% on 1-shot. Fine-tuning improves this to 85.4% (+42 points), confirming that domain-specific contrastive learning is essential.

## Project Structure

```
.
├── README.md
├── train.py              # Training script with experiment configs
├── dataset.py            # SymbolDataset, UniqueClassBatchSampler, augmentations
├── simclr_model.py       # Model creation and pretrained checkpoint loading
├── evaluate.py           # k-NN evaluation (1-shot, 5-shot, full)
├── run_experiments.ipynb # Notebook with all results, plots, and analysis
├── data/                 # Dataset (train/valid/test splits, COCO annotations)
├── experiments/          # Saved checkpoints and training histories
├── pretrained/           # SimCLR pretrained checkpoint (STL10)
└── SimCLR/               # Cloned SimCLR repo (model architecture, augmentations)
```

## How to Run

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision scikit-learn matplotlib numpy pyyaml tensorboard

# Run all experiments
python train.py --data_dir data --output_dir experiments --experiments all --device cuda

# Run a single experiment
python train.py --experiments baseline --device cuda

# View results in the notebook
jupyter notebook run_experiments.ipynb
```

## Evaluation Metrics

- **1-shot**: 1 random example per class as reference, classify test samples by nearest neighbor (k=1)
- **5-shot**: 5 random examples per class, classify by majority vote of 3 nearest neighbors (k=3)
- **Full kNN-5**: All training data as reference, k=5 nearest neighbors

Features are extracted from the ResNet-18 backbone (512-dim) before the projection head, using cosine distance for k-NN.
