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

    All ranks MUST call this function. Uses all_reduce for reliable broadcast.
    """
    if xr.world_size() <= 1:
        return Q, Lambda

    if Q is None or Lambda is None:
        return Q, Lambda

    rank = xr.global_ordinal()

    # Debug: Check input validity
    q_finite = torch.isfinite(Q).all().item()
    l_finite = torch.isfinite(Lambda).all().item()
    if rank == 0:
        print(f"[broadcast_Q_Lambda] Input - Q finite: {q_finite}, Lambda finite: {l_finite}")

    # CRITICAL FIX: Create fresh tensors with explicit copy to break any XLA graph dependencies
    # This ensures the tensors are fully materialized before broadcast
    Q_input = torch.zeros_like(Q)
    Q_input.copy_(Q)

    Lambda_input = torch.zeros_like(Lambda)
    Lambda_input.copy_(Lambda)

    # Sync to ensure copies complete
    xm.mark_step()

    # Simple approach: src keeps its data, others contribute zeros
    if rank == src:
        Q_send = Q_input
        Lambda_send = Lambda_input
    else:
        Q_send = torch.zeros_like(Q_input)
        Lambda_send = torch.zeros_like(Lambda_input)

    # All-reduce sum - only src's data survives
    Q_reduced = xm.all_reduce(xm.REDUCE_SUM, Q_send)
    L_reduced = xm.all_reduce(xm.REDUCE_SUM, Lambda_send)

    # CRITICAL: Force execution and create fresh output tensors
    xm.mark_step()

    # Clone results to avoid any view/reference issues
    Q_bcast = Q_reduced.detach().clone().contiguous()
    L_bcast = L_reduced.detach().clone().contiguous()

    # Another sync to ensure clones complete
    xm.mark_step()

    # Debug: Check output validity
    q_out_finite = torch.isfinite(Q_bcast).all().item()
    l_out_finite = torch.isfinite(L_bcast).all().item()
    if rank == 0 or not q_out_finite:
        print(f"[broadcast_Q_Lambda Rank {rank}] Output - Q finite: {q_out_finite}, max abs: {Q_bcast.abs().max().item() if q_out_finite else float('nan')}")

    return Q_bcast, L_bcast
