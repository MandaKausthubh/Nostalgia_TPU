import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

from typing import Optional, Tuple
import torch

def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast from master to all replicas (simple version)"""
    if xr.world_size() == 1:
        return tensor
    tensor = tensor.clone() if tensor is not None else torch.tensor(0.0, device=xm.xla_device())
    xm.broadcast(tensor, src=src)
    return tensor


def broadcast_Q_Lambda(
    Q: Optional[torch.Tensor],
    Lambda: Optional[torch.Tensor],
    src: int = 0
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Broadcast Q and Lambda tensors from source rank to all other ranks.

    All ranks must call this function. The source rank (typically rank 0)
    provides the data, and all other ranks receive it.

    Args:
        Q: Eigenvector matrix [N, k] on source rank, None on others
        Lambda: Eigenvalue tensor on source rank, None on others
        src: Source rank to broadcast from (default: 0)

    Returns:
        Tuple of (Q, Lambda) - all ranks receive the same tensors
    """
    world_size = xr.world_size()
    if world_size == 1:
        return Q, Lambda

    rank = xr.global_ordinal()

    # Determine shapes from source rank
    if rank == src:
        q_shape = torch.tensor(Q.shape if Q is not None else (0, 0), device=xm.xla_device(), dtype=torch.long)
        lambda_shape = torch.tensor(Lambda.shape if Lambda is not None else (0,), device=xm.xla_device(), dtype=torch.long)
        q_exists = torch.tensor(1 if Q is not None else 0, device=xm.xla_device(), dtype=torch.long)
        lambda_exists = torch.tensor(1 if Lambda is not None else 0, device=xm.xla_device(), dtype=torch.long)
    else:
        q_shape = torch.zeros(2, device=xm.xla_device(), dtype=torch.long)
        lambda_shape = torch.zeros(1, device=xm.xla_device(), dtype=torch.long)
        q_exists = torch.tensor(0, device=xm.xla_device(), dtype=torch.long)
        lambda_exists = torch.tensor(0, device=xm.xla_device(), dtype=torch.long)

    # Broadcast metadata first
    xm.broadcast(q_shape, src=src)
    xm.broadcast(lambda_shape, src=src)
    xm.broadcast(q_exists, src=src)
    xm.broadcast(lambda_exists, src=src)

    # Broadcast actual tensors if they exist
    if q_exists.item() == 1:
        if rank != src:
            Q = torch.zeros(q_shape[0].item(), q_shape[1].item(), device=xm.xla_device(), dtype=torch.float32)
        xm.broadcast(Q, src=src)

    if lambda_exists.item() == 1:
        if rank != src:
            Lambda = torch.zeros(lambda_shape[0].item(), device=xm.xla_device(), dtype=torch.float32)
        xm.broadcast(Lambda, src=src)

    return Q, Lambda
