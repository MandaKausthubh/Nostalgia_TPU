"""
main_GPU.py  –  entry point for the GPU version of the Nostalgia experiment.

Uses torchrun / torch.multiprocessing.spawn instead of xmp.spawn.

Usage examples
--------------
# Single GPU:
python main_GPU.py --no-distributed

# Multi-GPU with torchrun (recommended):
torchrun --nproc_per_node=4 main_GPU.py

# All hyperparams from CLI (same flags as main.py):
torchrun --nproc_per_node=4 main_GPU.py \\
    --root-dir /path/to/domainnet \\
    --batch-size 256 --lr 3e-5 --lora-r 16
"""

import argparse
import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoConfig, AutoModel, AutoProcessor

from VisionExperiment_GPU.VisionExperiment import NostalgiaExperiment
from models.model import NostalgiaConfig


# ─────────────────────────────────────────────────────────────────────────────
# Model prefetch  (identical to TPU version — download once before workers spawn)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_ID = "google/vit-base-patch32-224-in21k"

def prefetch_model(hf_home: str) -> None:
    """
    Download the ViT checkpoint to the local HF cache once, on the main
    process, before spawning GPU worker processes.  Workers inherit HF_HOME
    and load from disk — no concurrent HuggingFace requests.
    """
    os.makedirs(hf_home, exist_ok=True)
    print(f"[prefetch] Downloading '{_MODEL_ID}' to '{hf_home}' ...")
    AutoConfig.from_pretrained(_MODEL_ID)
    AutoProcessor.from_pretrained(_MODEL_ID, use_fast=True)
    AutoModel.from_pretrained(_MODEL_ID)
    print("[prefetch] Done — spawning GPU workers.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser  (same flags as main.py for drop-in substitution)
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main_GPU.py",
        description="Nostalgia continual-learning experiment on DomainNet (GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--root-dir",   type=str, default=NostalgiaConfig.root_dir)
    g.add_argument("--batch-size", type=int, default=NostalgiaConfig.batch_size)
    g.add_argument("--batch-size-for-accumulation", type=int,
                   default=NostalgiaConfig.batch_size_for_accumulation)
    g.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per GPU (safe to use >0 on GPU).")

    g = p.add_argument_group("Learning rates")
    g.add_argument("--lr",            type=float, default=NostalgiaConfig.lr)
    g.add_argument("--downstream-lr", type=float, default=NostalgiaConfig.downstream_lr)

    g = p.add_argument_group("Optimiser")
    g.add_argument("--optimizer",     type=str, default=NostalgiaConfig.base_optimizer,
                   choices=["adamw", "adam", "sgd"])
    g.add_argument("--weight-decay",  type=float, default=NostalgiaConfig.weight_decay)

    g = p.add_argument_group("Training schedule")
    g.add_argument("--num-epochs",          type=int, default=NostalgiaConfig.num_epochs)
    g.add_argument("--head-warmup-epochs",  type=int, default=NostalgiaConfig.head_warmup_epochs)
    g.add_argument("--warmup-steps",        type=int, default=NostalgiaConfig.warmup_steps)
    g.add_argument("--validate-after-steps",type=int, default=NostalgiaConfig.validate_after_steps)

    g = p.add_argument_group("Regularisation")
    g.add_argument("--grad-clip-norm", type=float, default=NostalgiaConfig.grad_clip_norm)
    g.add_argument("--ewc-lambda",     type=float, default=NostalgiaConfig.ewc_lambda)
    g.add_argument("--l2sp-lambda",    type=float, default=NostalgiaConfig.l2sp_lambda)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r",       type=int,   default=NostalgiaConfig.lora_r)
    g.add_argument("--lora-alpha",   type=int,   default=NostalgiaConfig.lora_alpha)
    g.add_argument("--lora-dropout", type=float, default=NostalgiaConfig.lora_dropout)
    g.add_argument("--no-reset-lora", action="store_true")

    g = p.add_argument_group("Hessian / Nostalgia")
    g.add_argument("--hessian-eigenspace-dim",    type=int,   default=NostalgiaConfig.hessian_eigenspace_dim)
    g.add_argument("--iterations-of-accumulation",type=int,   default=NostalgiaConfig.iterations_of_accumulation)
    g.add_argument("--gamma",                     type=float, default=float(NostalgiaConfig.gamma))
    g.add_argument("--no-scaling",   action="store_true")
    g.add_argument("--accumulate-mode", type=str, default=NostalgiaConfig.accumulate_mode,
                   choices=["accumulate", "union"])
    g.add_argument("--merge-tasks",  type=str,  default=NostalgiaConfig.merge_tasks,
                   choices=["accumulate", "union"])

    g = p.add_argument_group("Hardware / misc")
    g.add_argument("--no-distributed", action="store_true",
                   help="Run on a single GPU without torch.distributed.")
    g.add_argument("--world-size", type=int, default=torch.cuda.device_count() or 1,
                   help="Number of GPUs to use (only applies to mp.spawn mode).")
    g.add_argument("--seed",    type=int, default=NostalgiaConfig.seed)
    g.add_argument("--mode",    type=str, default=NostalgiaConfig.mode,
                   choices=["nostalgia", "finetune", "ewc", "l2sp"])
    g.add_argument("--log-dir", type=str, default=None)
    g.add_argument("--no-log-deltas", action="store_true")

    return p


def args_to_config(args: argparse.Namespace) -> NostalgiaConfig:
    cfg = NostalgiaConfig(
        root_dir                    = args.root_dir,
        batch_size                  = args.batch_size,
        batch_size_for_accumulation = args.batch_size_for_accumulation,
        num_workers                 = args.num_workers,
        lr                          = args.lr,
        downstream_lr               = args.downstream_lr,
        base_optimizer              = args.optimizer,
        weight_decay                = args.weight_decay,
        num_epochs                  = args.num_epochs,
        head_warmup_epochs          = args.head_warmup_epochs,
        warmup_steps                = args.warmup_steps,
        validate_after_steps        = args.validate_after_steps,
        grad_clip_norm              = args.grad_clip_norm,
        ewc_lambda                  = args.ewc_lambda,
        l2sp_lambda                 = args.l2sp_lambda,
        lora_r                      = args.lora_r,
        lora_alpha                  = args.lora_alpha,
        lora_dropout                = args.lora_dropout,
        reset_lora                  = not args.no_reset_lora,
        hessian_eigenspace_dim      = args.hessian_eigenspace_dim,
        iterations_of_accumulation  = args.iterations_of_accumulation,
        gamma                       = args.gamma,
        use_scaling                 = not args.no_scaling,
        accumulate_mode             = args.accumulate_mode,
        merge_tasks                 = args.merge_tasks,
        use_tpu                     = False,   # GPU run — always False
        world_size                  = args.world_size,
        seed                        = args.seed,
        mode                        = args.mode,
        log_deltas                  = not args.no_log_deltas,
    )
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Worker — called once per GPU by mp.spawn / torchrun
# ─────────────────────────────────────────────────────────────────────────────

def _worker(rank: int, config: NostalgiaConfig, world_sz: int, dist_url: str):
    """
    Initialise the process group for this rank, then run the experiment.

    When launched via `torchrun`, RANK / LOCAL_RANK / WORLD_SIZE are already
    set in the environment; dist.init_process_group discovers them automatically
    via the 'env://' init_method.

    When launched via mp.spawn (--no-distributed flag absent, torchrun absent),
    we fall back to a simple NCCL rendezvous on localhost.
    """
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_sz,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    experiment = NostalgiaExperiment(config, rank=rank)
    experiment.train()

    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    config = args_to_config(args)

    print("\n=== Nostalgia GPU Config ===")
    for field, value in vars(config).items():
        print(f"  {field:<35} {value}")
    print("============================\n")

    # ── Model prefetch ────────────────────────────────────────────────────
    hf_home = os.environ.get("HF_HOME", "/tmp/.cache/hf")
    os.environ["HF_HOME"] = hf_home
    prefetch_model(hf_home)

    if args.no_distributed:
        # ── Single-GPU mode ───────────────────────────────────────────────
        print("[main] Single-GPU mode (no torch.distributed).")
        # Fake a trivial process group so dist calls don't crash
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500",
                                world_size=1, rank=0)
        experiment = NostalgiaExperiment(config, rank=0)
        experiment.train()
        dist.destroy_process_group()

    else:
        # ── Multi-GPU mode ────────────────────────────────────────────────
        # Support both torchrun (env vars already set) and mp.spawn fallback.
        if "RANK" in os.environ:
            # Launched by torchrun — just run the worker directly in this process
            rank     = int(os.environ["RANK"])
            world_sz = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            experiment = NostalgiaExperiment(config, rank=rank)
            experiment.train()
            dist.destroy_process_group()
        else:
            # Fallback: use mp.spawn
            world_sz = min(args.world_size, torch.cuda.device_count())
            assert world_sz > 0, "No CUDA devices found."
            dist_url = "tcp://127.0.0.1:29500"
            mp.spawn(
                _worker,
                args=(config, world_sz, dist_url),
                nprocs=world_sz,
                join=True,
            )


if __name__ == "__main__":
    main()
