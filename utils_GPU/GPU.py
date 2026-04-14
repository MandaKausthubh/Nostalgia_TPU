"""
utils_GPU/GPU.py

GPU/NCCL equivalents of utils/TPU.py.

Replaces:
  xm.all_reduce  → dist.all_reduce
  xr.world_size  → dist.get_world_size
  xr.global_ordinal → dist.get_rank
  broadcast trick (zero-out + sum) → dist.broadcast
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_master() -> bool:
    return global_rank() == 0


def master_print(*args, **kwargs):
    if is_master():
        print(*args, **kwargs)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from `src` rank to all replicas via NCCL."""
    if world_size() == 1:
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_Q_Lambda(
    Q: Optional[torch.Tensor],
    Lambda: Optional[torch.Tensor],
    src: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Broadcast Q and Lambda from rank `src` to every other rank.

    All ranks MUST call this function.  On the source rank Q and Lambda
    must be valid tensors; on other ranks they may be None (they will be
    allocated here using the shape broadcast from the source).
    """
    if world_size() == 1:
        return Q, Lambda

    rank = global_rank()
    device = torch.device(f"cuda:{rank}")

    # ── Step 1: broadcast shape scalars ──────────────────────────────────
    if rank == src:
        assert Q is not None and Lambda is not None
        N, k = Q.shape
    else:
        N, k = 0, 0

    shape_t = torch.tensor([float(N), float(k)], device=device)
    dist.broadcast(shape_t, src=src)
    N_global = int(shape_t[0].item())
    k_global = int(shape_t[1].item())

    if N_global == 0 or k_global == 0:
        return Q, Lambda

    # ── Step 2: broadcast Q ───────────────────────────────────────────────
    if rank == src:
        Q_bcast = Q.to(device=device, dtype=torch.float32).contiguous()
    else:
        Q_bcast = torch.zeros(N_global, k_global, device=device, dtype=torch.float32)
    dist.broadcast(Q_bcast, src=src)

    # ── Step 3: broadcast Lambda ──────────────────────────────────────────
    if rank == src:
        L_bcast = Lambda.to(device=device, dtype=torch.float32).contiguous()
    else:
        L_bcast = torch.zeros(k_global, device=device, dtype=torch.float32)
    dist.broadcast(L_bcast, src=src)

    return Q_bcast, L_bcast


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce (SUM) then divide by world size → mean across ranks."""
    if world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return tensor


def mesh_reduce(name: str, value: float, reduce_fn) -> float:
    """
    Scalar all-reduce helper.  `reduce_fn` is only used to determine the
    op: pass `sum` for SUM reduction (then divide manually if you want mean).
    Unlike xm.mesh_reduce, this returns the summed value across all ranks.
    """
    t = torch.tensor(value, dtype=torch.float32,
                     device=torch.device(f"cuda:{global_rank()}"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()
