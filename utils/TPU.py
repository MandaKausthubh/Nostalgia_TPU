import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

from typing import Optional, Tuple
import torch


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from src rank to all replicas using all_reduce trick."""
    if xr.world_size() == 1:
        return tensor

    rank = xr.global_ordinal()

    # XLA doesn't have a direct broadcast; we zero out non-src ranks and all_reduce(SUM)
    if rank != src:
        tensor = torch.zeros_like(tensor)

    tensor = xm.all_reduce(xm.REDUCE_SUM, tensor)
    return tensor


def broadcast_Q_Lambda(
    Q: Optional[torch.Tensor],
    Lambda: Optional[torch.Tensor],
    src: int = 0
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Broadcast Q and Lambda tensors from source rank (src) to all other ranks.

    All ranks MUST call this function.  The source rank (rank 0 by default)
    holds valid tensors; every other rank contributes zeros that get summed
    away by all_reduce(SUM), leaving only the src values.

    Args:
        Q:      Eigenvector matrix [N, k] on src rank; can be None on others.
        Lambda: Eigenvalue tensor   [k]   on src rank; can be None on others.
        src:    Source rank (default 0).

    Returns:
        (Q, Lambda) – identical tensors on every rank.
    """
    world_size = xr.world_size()
    if world_size == 1:
        return Q, Lambda

    global_type = Q.dtype

    rank = xr.global_ordinal()
    device = xm.xla_device()

    # ------------------------------------------------------------------ #
    # Step 1 – broadcast shape metadata using scalar all_reduces           #
    # ------------------------------------------------------------------ #
    if rank == src:
        assert Q is not None and Lambda is not None, \
            "Source rank must have valid Q and Lambda tensors"
        N, k = Q.shape
    else:
        N, k = 0, 0

    # Exchange N and k via all_reduce (only src contributes the real values)
    N_t = torch.tensor(float(N), device=device, dtype=global_type)
    k_t = torch.tensor(float(k), device=device, dtype=global_type)
    N_t = xm.all_reduce(xm.REDUCE_SUM, N_t)
    k_t = xm.all_reduce(xm.REDUCE_SUM, k_t)
    xm.mark_step()

    N_global = int(N_t.item())
    k_global = int(k_t.item())

    if N_global == 0 or k_global == 0:
        # Nothing to broadcast (e.g. first task, Q/Lambda are None everywhere)
        return Q, Lambda

    # ------------------------------------------------------------------ #
    # Step 2 – broadcast Q                                                 #
    # ------------------------------------------------------------------ #
    if rank == src:
        Q_bcast = Q.to(device=device, dtype=global_type)
    else:
        Q_bcast = torch.zeros(N_global, k_global, device=device, dtype=global_type)

    Q_bcast = xm.all_reduce(xm.REDUCE_SUM, Q_bcast)
    xm.mark_step()

    # ------------------------------------------------------------------ #
    # Step 3 – broadcast Lambda                                            #
    # ------------------------------------------------------------------ #
    if rank == src:
        L_bcast = Lambda.to(device=device, dtype=global_type)
    else:
        L_bcast = torch.zeros(k_global, device=device, dtype=global_type)

    L_bcast = xm.all_reduce(xm.REDUCE_SUM, L_bcast)
    xm.mark_step()

    return Q_bcast, L_bcast
