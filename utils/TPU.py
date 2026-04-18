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

    All ranks MUST call this function. This relies on identical computational
    graphs across all ranks to maintain SPMD compliance for PyTorch/XLA.
    """
    if xr.world_size() <= 1:
        return Q, Lambda

    if Q is None or Lambda is None:
        return Q, Lambda

    device = Q.device
    dtype = Q.dtype
    rank = xr.global_ordinal()

    # CRITICAL FIX: Ensure Q and Lambda are fully materialized before broadcast
    # Clone and make contiguous to avoid view issues
    Q = Q.detach().clone().contiguous()
    Lambda = Lambda.detach().clone().contiguous()

    # SPMD safe masking: all ranks construct a mask, so the graph is exactly the same.
    # Non-source ranks multiply their Q by 0, source rank multiplies by 1.
    mask = torch.tensor(1.0 if rank == src else 0.0, device=device, dtype=dtype)

    Q_bcast = xm.all_reduce(xm.REDUCE_SUM, Q * mask)
    L_bcast = xm.all_reduce(xm.REDUCE_SUM, Lambda * mask)

    # Sync block to ensure propagation completes successfully
    xm.mark_step()

    return Q_bcast, L_bcast
