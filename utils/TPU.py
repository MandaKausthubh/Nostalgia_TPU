import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

from typing import Optional
import torch

def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast from master to all replicas (simple version)"""
    if xr.world_size() == 1:
        return tensor
    tensor = tensor.clone() if tensor is not None else torch.tensor(0.0, device=xm.xla_device())
    xm.broadcast(tensor, src=src)
    return tensor

# For Q / Lambda broadcast (assuming they are flat or can be flattened)
def broadcast_Q_Lambda(Q: Optional[torch.Tensor], Lambda: Optional[torch.Tensor]) -> tuple:
    if Q is not None:
        Q = broadcast_tensor(Q)
    if Lambda is not None:
        Lambda = broadcast_tensor(Lambda)
    return Q, Lambda
