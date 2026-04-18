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
    # CRITICAL: Clone tensor first to avoid modifying original on non-src ranks
    tensor = tensor.clone()
    if rank != src:
        tensor.zero_()

    tensor = xm.all_reduce(xm.REDUCE_SUM, tensor)
    xm.mark_step()  # Ensure broadcast completes before return
    return tensor


def broadcast_Q_Lambda(
    Q: Optional[torch.Tensor],
    Lambda: Optional[torch.Tensor],
    src: int = 0
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Broadcast Q and Lambda tensors from source rank (src) to all other ranks.

    All ranks MUST call this function. Uses all_gather for reliable broadcast.
    """
    if xr.world_size() <= 1:
        return Q, Lambda

    if Q is None or Lambda is None:
        return Q, Lambda

    rank = xr.global_ordinal()
    world_size = xr.world_size()

    # Debug: Check input validity
    q_finite = torch.isfinite(Q).all().item()
    l_finite = torch.isfinite(Lambda).all().item()
    if rank == 0:
        print(f"[broadcast_Q_Lambda] Input - Q finite: {q_finite}, Lambda finite: {l_finite}")

    # CRITICAL FIX: Ensure Q and Lambda are fully materialized before broadcast
    Q_input = Q.detach().clone().contiguous()
    Lambda_input = Lambda.detach().clone().contiguous()
    xm.mark_step()

    # METHOD: Use all_gather where we gather from all ranks, then select src's data
    # This is more reliable than all_reduce for broadcast on XLA/TPU

    # Gather Q from all ranks - each rank contributes its Q_input
    Q_gathered = xm.all_gather(Q_input, dim=0)  # shape: [world_size * n, k]
    xm.mark_step()

    # Reshape to [world_size, n, k] and select src rank
    n, k = Q_input.shape
    Q_gathered = Q_gathered.view(world_size, n, k)
    Q_bcast = Q_gathered[src].clone().contiguous()
    xm.mark_step()

    # Gather Lambda from all ranks
    Lambda_gathered = xm.all_gather(Lambda_input, dim=0)  # shape: [world_size * m]
    xm.mark_step()

    # Reshape and select src rank
    m = Lambda_input.shape[0]
    Lambda_gathered = Lambda_gathered.view(world_size, m)
    L_bcast = Lambda_gathered[src].clone().contiguous()
    xm.mark_step()

    # Debug: Check output validity
    q_out_finite = torch.isfinite(Q_bcast).all().item()
    l_out_finite = torch.isfinite(L_bcast).all().item()
    if not q_out_finite or not l_out_finite:
        print(f"[broadcast_Q_Lambda Rank {rank}] Output - Q finite: {q_out_finite}, Lambda finite: {l_out_finite}")

    return Q_bcast, L_bcast
