# Nostalgia — Hessian-Guided Continual Learning on TPU

> **Gradient projection via Hessian eigenspace memory** for catastrophic-forgetting-free fine-tuning of Vision Transformers across sequential domains, accelerated on Google Cloud TPU v3-8.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Ideas](#key-ideas)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Quick Start](#quick-start)
8. [CLI Reference](#cli-reference)
9. [Architecture](#architecture)
10. [Algorithm Detail](#algorithm-detail)
11. [TPU-Specific Notes](#tpu-specific-notes)
12. [Logging & Monitoring](#logging--monitoring)
13. [Experiment Modes](#experiment-modes)

---

## Overview

**Nostalgia** is a continual learning framework for Vision Transformers (ViT) that prevents catastrophic forgetting by projecting gradients *out of* the high-curvature subspace of the Hessian accumulated over all previously seen tasks.

The core intuition: parameters that matter most for past tasks correspond to directions of **high loss curvature**. By identifying those directions via Lanczos-based Hessian eigenspace estimation and subtracting the corresponding gradient components, the model is free to adapt to new tasks while minimally disturbing old knowledge.

This implementation runs on **Google Cloud TPU v3-8** (8 cores) using PyTorch/XLA (`torch_xla`) with full SPMD-style data parallelism.

---

## Key Ideas

| Concept | Description |
|---|---|
| **Hessian eigenspace** | Top-*k* eigenvectors of the Fisher/Hessian computed over a task's training data |
| **Gradient projection** | `g' = g − Q(Qᵀg)` — removes gradient components in the remembered subspace |
| **Eigenspace accumulation** | Running PSD-weighted average of per-task Hessian factors across all past domains |
| **LoRA adapters** | Only LoRA parameters are trained; the pre-trained ViT backbone weights are frozen |
| **Eigenvalue scaling** | Optional — scales projected components by `λ / (median(λ) + λ)` to weight by curvature magnitude |
| **Lanczos algorithm** | Memory-efficient HVP-based iterative Hessian approximation with full reorthogonalisation |

---

## Project Structure

```
TPU/
├── main.py                      # Entry point: CLI args → NostalgiaConfig → xmp.spawn
├── VisionExperiment.py          # NostalgiaExperiment class — training loop, Q/Λ update
├── Experiment.py                # Minimal experiment scaffold (development/CPU use)
├── models/
│   └── model.py                 # NostalgiaConfig, ViTClassifier, ContinualLearnerViT
└── utils/
    ├── nostalgia.py             # NostalgiaOptimizer — gradient projection wrapper
    ├── hessians.py              # hvp_flat, lanczos, compute_Q_for_task, recover_eigenspace_from_factor
    ├── accumulate.py            # accumulate_hessian_eigenspace_stable (CPU-safe QR/eigh)
    ├── TPU.py                   # broadcast_tensor, broadcast_Q_Lambda (XLA collective ops)
    └── logging.py               # WandbLogger (master-only W&B wrapper)
```

### File Roles at a Glance

| File | Responsibility |
|---|---|
| `main.py` | Parse CLI, prefetch HuggingFace checkpoint, launch `xmp.spawn` |
| `VisionExperiment.py` | Orchestrate domain-sequential training; compute & accumulate Q/Λ; broadcast to all ranks |
| `models/model.py` | `ContinualLearnerViT`: ViT-Base + LoRA + per-domain task heads + `configure_optimizers` |
| `utils/nostalgia.py` | `NostalgiaOptimizer`: wraps any `torch.optim` optimizer and injects the projection step |
| `utils/hessians.py` | Lanczos HVP loop, full-space Q recovery via Ritz lift, cross-rank HVP reduction |
| `utils/accumulate.py` | Stable multi-task eigenspace merge with CPU offload for XLA-incompatible linalg ops |
| `utils/TPU.py` | SPMD-safe broadcast via `all_gather`; avoids branching for XLA compliance |
| `utils/logging.py` | W&B logging guarded behind `is_master_ordinal()` to avoid multi-rank log spam |

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- `torch_xla` (matching your TPU runtime; see [pytorch/xla](https://github.com/pytorch/xla))
- `transformers` ≥ 4.40
- `peft` ≥ 0.10
- `torchvision`
- `pytorch-adapt` (for `DomainNet` loader)
- `wandb`
- `tqdm`

---

## Installation

```bash
# 1. Clone
git clone https://github.com/MandaKausthubh/Nostalgia_TPU.git 
cd Nostalgia_TPU

# IF using kaggle to run these files use the following python code in your first python cell
# import os
# os.environ.pop('TPU_PROCESS_ADDRESSES')

pip install --upgrade pip
pip install torch torchvision torchaudio pytorch_adapt peft wandb weave protobuf==7.34.1
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# RUN the following to start the training on domain net dataset
python /kaggle/working/Nostalgia_TPU/main.py --lr 1e-4 --downstream-lr 5e-4 --batch-size 64 --head-warmup-epochs 15 --num-epochs 5
```

## Dataset

This project trains on **DomainNet** — a large-scale multi-domain image classification benchmark with 345 classes across 6 visual domains.

The experiment uses the following **5 domains** in sequence:
`clipart → quickdraw → sketch → infograph → painting`

Download DomainNet and point `--root-dir` at the root folder containing the per-domain subdirectories. The loader is provided by `pytorch-adapt`:

```python
from pytorch_adapt.datasets import DomainNet
train_ds = DomainNet(root, domain="clipart", train=True, transform=...)
```

On Kaggle, the default path is:
```
/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset
```
---

## Citation

If you use this code in your research, please consider citing the relevant Nostalgia / continual learning work that inspired this implementation.

---

*Developed as part of the Nostalgia Project — TPU continual learning research.*
