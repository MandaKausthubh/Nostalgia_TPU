"""
main.py  –  entry point for the Nostalgia continual-learning TPU experiment.

Usage examples
--------------
# Default (8-core TPU, all hyperparams from NostalgiaConfig defaults):
python main.py

# Quick smoke-test on CPU with tiny settings:
python main.py --no-tpu --num-epochs 1 --head-warmup-epochs 0 \
               --batch-size 32 --hessian-eigenspace-dim 4

# Sweep backbone LR and LoRA rank:
python main.py --lr 1e-4 --lora-r 32 --lora-alpha 64

# Full custom run:
python main.py \
    --root-dir /kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset \
    --batch-size 512 \
    --lr 3e-5 --downstream-lr 3e-4 \
    --optimizer adamw --weight-decay 1e-2 \
    --num-epochs 10 --head-warmup-epochs 3 --warmup-steps 100 \
    --grad-clip-norm 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --hessian-eigenspace-dim 16 --iterations-of-accumulation 4 \
    --seed 42
"""

import argparse
import functools
import os

from transformers import AutoConfig, AutoModel, AutoProcessor
import torch_xla.distributed.xla_multiprocessing as xmp

from VisionExperiment import NostalgiaExperiment
from models.model import NostalgiaConfig


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Nostalgia continual-learning experiment on DomainNet (TPU / CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument(
        "--root-dir", type=str,
        default=NostalgiaConfig.root_dir,
        help="Root directory of the DomainNet dataset.",
    )
    g.add_argument(
        "--batch-size", type=int,
        default=NostalgiaConfig.batch_size,
        help="Global batch size (split evenly across all TPU cores).",
    )
    g.add_argument(
        "--batch-size-for-accumulation", type=int,
        default=NostalgiaConfig.batch_size_for_accumulation,
        help="Per-step batch size used when computing the Hessian.",
    )
    g.add_argument(
        "--num-workers", type=int,
        # IMPORTANT: must be 0 when using xmp.spawn (start_method='spawn').
        # DataLoader worker subprocesses try to share tensors with the parent
        # via _share_fd_cpu_() which requires CPU tensors. XLA tensors live on
        # device and cannot be shared this way → RuntimeError: _share_fd_:
        # only available on CPU.  Keep at 0 unless you know what you're doing.
        default=0,
        help=(
            "DataLoader workers per replica. "
            "MUST be 0 with xmp.spawn (start_method='spawn') on TPU; "
            "non-zero causes '_share_fd_: only available on CPU'."
        ),
    )

    # ── Learning rates ────────────────────────────────────────────────────
    g = p.add_argument_group("Learning rates")
    g.add_argument(
        "--lr", type=float,
        default=NostalgiaConfig.lr,
        help="Backbone (LoRA) peak learning rate.",
    )
    g.add_argument(
        "--downstream-lr", type=float,
        default=NostalgiaConfig.downstream_lr,
        help="Task-head learning rate.",
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Optimiser")
    g.add_argument(
        "--optimizer", type=str,
        default=NostalgiaConfig.base_optimizer,
        choices=["adamw", "adam", "sgd"],
        help="Base optimiser type.",
    )
    g.add_argument(
        "--weight-decay", type=float,
        default=NostalgiaConfig.weight_decay,
        help="Decoupled weight decay (backbone only; head uses 0).",
    )

    # ── Training schedule ─────────────────────────────────────────────────
    g = p.add_argument_group("Training schedule")
    g.add_argument(
        "--num-epochs", type=int,
        default=NostalgiaConfig.num_epochs,
        help="Training epochs per domain.",
    )
    g.add_argument(
        "--head-warmup-epochs", type=int,
        default=NostalgiaConfig.head_warmup_epochs,
        help="Epochs to train ONLY the task head before unfreezing the backbone.",
    )
    g.add_argument(
        "--warmup-steps", type=int,
        default=NostalgiaConfig.warmup_steps,
        help="Linear LR warmup steps at the start of each domain.",
    )
    g.add_argument(
        "--validate-after-steps", type=int,
        default=NostalgiaConfig.validate_after_steps,
        help="Run validation every N steps (within epoch).",
    )

    # ── Regularisation ────────────────────────────────────────────────────
    g = p.add_argument_group("Regularisation")
    g.add_argument(
        "--grad-clip-norm", type=float,
        default=NostalgiaConfig.grad_clip_norm,
        help="Max gradient norm for clipping (0 = disabled).",
    )
    g.add_argument(
        "--ewc-lambda", type=float,
        default=NostalgiaConfig.ewc_lambda,
        help="EWC regularisation strength.",
    )
    g.add_argument(
        "--l2sp-lambda", type=float,
        default=NostalgiaConfig.l2sp_lambda,
        help="L2-SP regularisation strength.",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("LoRA")
    g.add_argument(
        "--lora-r", type=int,
        default=NostalgiaConfig.lora_r,
        help="LoRA rank r.",
    )
    g.add_argument(
        "--lora-alpha", type=int,
        default=NostalgiaConfig.lora_alpha,
        help="LoRA scaling alpha (convention: 2 × r).",
    )
    g.add_argument(
        "--lora-dropout", type=float,
        default=NostalgiaConfig.lora_dropout,
        help="LoRA adapter dropout probability.",
    )
    g.add_argument(
        "--no-reset-lora", action="store_true",
        help="Do NOT reset LoRA weights between domains.",
    )

    # ── Hessian / Nostalgia ───────────────────────────────────────────────
    g = p.add_argument_group("Hessian / Nostalgia")
    g.add_argument(
        "--hessian-eigenspace-dim", type=int,
        default=NostalgiaConfig.hessian_eigenspace_dim,
        help="Number of Lanczos vectors (k) kept in the Hessian eigenspace.",
    )
    g.add_argument(
        "--iterations-of-accumulation", type=int,
        default=NostalgiaConfig.iterations_of_accumulation,
        help="How many data passes to use when estimating Q / Lambda.",
    )
    g.add_argument(
        "--gamma", type=float,
        default=float(NostalgiaConfig.gamma),
        help="EMA coefficient for Hessian accumulation.",
    )
    g.add_argument(
        "--no-scaling", action="store_true",
        help="Disable eigenvalue-aware gradient scaling.",
    )
    g.add_argument(
        "--accumulate-mode", type=str,
        default=NostalgiaConfig.accumulate_mode,
        choices=["accumulate", "union"],
        help="How to merge eigenspaces across epochs.",
    )
    g.add_argument(
        "--merge-tasks", type=str,
        default=NostalgiaConfig.merge_tasks,
        choices=["accumulate", "union"],
        help="How to merge eigenspaces across tasks.",
    )

    # ── Hardware / misc ───────────────────────────────────────────────────
    g = p.add_argument_group("Hardware / misc")
    g.add_argument(
        "--no-tpu", action="store_true",
        help="Run on CPU/MPS instead of XLA TPU (useful for local debugging).",
    )
    g.add_argument(
        "--world-size", type=int,
        default=NostalgiaConfig.world_size,
        help="Number of TPU cores / processes to spawn.",
    )
    g.add_argument(
        "--seed", type=int,
        default=NostalgiaConfig.seed,
        help="Global random seed.",
    )
    g.add_argument(
        "--mode", type=str,
        default=NostalgiaConfig.mode,
        choices=["nostalgia", "finetune", "ewc", "l2sp"],
        help="Experiment mode.",
    )
    g.add_argument(
        "--log-dir", type=str,
        default=None,
        help="Override the auto-generated W&B / TensorBoard log directory.",
    )
    g.add_argument(
        "--no-log-deltas", action="store_true",
        help="Disable logging of parameter deltas.",
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def args_to_config(args: argparse.Namespace) -> NostalgiaConfig:
    """Convert parsed CLI args into a NostalgiaConfig dataclass."""

    cfg = NostalgiaConfig(
        # data
        root_dir                     = args.root_dir,
        batch_size                   = args.batch_size,
        batch_size_for_accumulation  = args.batch_size_for_accumulation,
        num_workers                  = args.num_workers,   # safe default = 0
        # learning rates
        lr                           = args.lr,
        downstream_lr                = args.downstream_lr,
        # optimiser
        base_optimizer               = args.optimizer,
        weight_decay                 = args.weight_decay,
        # schedule
        num_epochs                   = args.num_epochs,
        head_warmup_epochs           = args.head_warmup_epochs,
        warmup_steps                 = args.warmup_steps,
        validate_after_steps         = args.validate_after_steps,
        # regularisation
        grad_clip_norm               = args.grad_clip_norm,
        ewc_lambda                   = args.ewc_lambda,
        l2sp_lambda                  = args.l2sp_lambda,
        # LoRA
        lora_r                       = args.lora_r,
        lora_alpha                   = args.lora_alpha,
        lora_dropout                 = args.lora_dropout,
        reset_lora                   = not args.no_reset_lora,
        # Hessian / Nostalgia
        hessian_eigenspace_dim       = args.hessian_eigenspace_dim,
        iterations_of_accumulation   = args.iterations_of_accumulation,
        gamma                        = args.gamma,
        use_scaling                  = not args.no_scaling,
        accumulate_mode              = args.accumulate_mode,
        merge_tasks                  = args.merge_tasks,
        # hardware / misc
        use_tpu                      = not args.no_tpu,
        world_size                   = args.world_size,
        seed                         = args.seed,
        mode                         = args.mode,
        log_deltas                   = not args.no_log_deltas,
    )

    if args.log_dir is not None:
        cfg.log_dir = args.log_dir

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model prefetch  (called ONCE on the main process, before xmp.spawn)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_ID = "google/vit-base-patch32-224-in21k"

def prefetch_model(hf_home: str) -> None:
    """
    Download the ViT checkpoint to the local HF cache exactly once, on the
    main process, before xmp.spawn launches the 8 TPU worker processes.

    Without this, every worker calls AutoModel.from_pretrained() as soon as
    it spawns, producing up to 24 simultaneous requests (config + processor +
    weights, times 8 workers) against HuggingFace Hub.  Kaggle free-tier VMs
    throttle that traffic, causing the download to stall indefinitely.

    Workers inherit the HF_HOME env-var set in main() and therefore find the
    weights already on disk — no network calls needed inside the workers.
    """
    os.makedirs(hf_home, exist_ok=True)
    print(f"[prefetch] Downloading '{_MODEL_ID}' to '{hf_home}' ...")
    AutoConfig.from_pretrained(_MODEL_ID)
    AutoProcessor.from_pretrained(_MODEL_ID, use_fast=True)
    AutoModel.from_pretrained(_MODEL_ID)
    print(f"[prefetch] Model cached — spawning TPU workers.")


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (called once per TPU core by xmp.spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _mp_fn(rank: int, config: NostalgiaConfig) -> None:
    experiment = NostalgiaExperiment(config)
    experiment.train(rank)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    config = args_to_config(args)

    print("\n=== Nostalgia Experiment Config ===")
    for field, value in vars(config).items():
        print(f"  {field:<35} {value}")
    print("===================================\n")

    # ── Model prefetch ────────────────────────────────────────────────────
    # Point HuggingFace to a writable Kaggle working directory and download
    # the model ONCE here, before spawning the 8 TPU worker processes.
    # Workers inherit HF_HOME from the environment and load from disk cache.
    hf_home = "/kaggle/working/.cache/hf"
    os.environ["HF_HOME"] = hf_home
    prefetch_model(hf_home)

    # Pass config through xmp.spawn via functools.partial so every spawned
    # process receives an identical config without re-parsing CLI args.
    xmp.spawn(
        functools.partial(_mp_fn, config=config),
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
