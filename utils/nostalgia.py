import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Any
from utils.hessians import *


class NostalgiaOptimizer(Optimizer):
    """
    Wraps a base optimizer and applies a Nostalgia-style gradient projection:
        g' = g - Q (Q^T g)

    Projection is applied ONLY to the specified projection_params.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        base_optimizer: Optimizer,
        device: torch.device,
        dtype: torch.dtype,
        writter: Optional[Any] = None,
        starting_step: int = 0,
        log_every: int = 50,
        weight_decay: float = 1e-4,
    ):
        super().__init__(params, {})  # Dummy call to satisfy Optimizer base class
        # Important: we do NOT pass params to super(); base_optimizer owns them
        self.base_optimizer = base_optimizer

        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype
        self.manual_weight_decay = weight_decay

        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.scaling: Optional[torch.Tensor] = None
        self.writter = writter

        self.log_every = log_every
        self.ema_beta = 0.98
        self.proj_ratio_ema: Optional[float] = None
        self.step_count = starting_step

        # Fixed parameter layout (ordering matters!)
        self.param_numels = [p.numel() for p in self.projection_params]
        self.num_params = sum(self.param_numels)
        self.k_max: Optional[int] = None

        # New list of hessian eigenvectors and eigenvalues
        # self.nostalgia_Q: List[torch.Tensor] = []
        # self.scaling: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_Q(self, Q: torch.Tensor, scaling: Optional[torch.Tensor] = None):
        """
        Q: [num_params, k] matrix of eigenvectors
        scaling: optional [k] or [k, k] eigenvalue-based scaling
        """
        if Q.shape[0] != self.num_params:
            raise ValueError(
                f"Q has {Q.shape[0]} rows, expected {self.num_params} "
                f"(sum of projection parameter sizes)."
            )
        self.nostalgia_Q = Q.to(self.device, self.dtype)
        self.scaling = scaling.to(self.device, self.dtype) if scaling is not None else None

    # ------------------------------------------------------------------
    def _flatten_grads(self) -> torch.Tensor:
        """
        Flatten gradients of ALL projection params in fixed order.
        Missing gradients are treated as zeros.
        """
        flat_grads = []
        for p in self.projection_params:
            if p.grad is None:
                flat_grads.append(torch.zeros(
                    p.numel(), device=self.device, dtype=self.dtype
                ))
            else:
                flat_grads.append(p.grad.view(-1))
        return torch.cat(flat_grads)

    # ------------------------------------------------------------------
    def _unflatten_to_grads(self, flat_grad: torch.Tensor):
        """
        Write projected flat gradient back into parameter .grad fields.
        """
        pointer = 0
        for p, n in zip(self.projection_params, self.param_numels):
            grad_slice = flat_grad[pointer:pointer + n].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            pointer += n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None): #type: ignore
        if self.nostalgia_Q is not None:

            # if self.base_optimizer.param_groups[0].get("weight_decay", 0.0) == 0.0:
            #     decay = self.manual_weight_decay
            #     if decay > 0:
            #         for p in self.projection_params:
            #             if p.grad is not None:
            #                 p.grad.add_(decay * p.data)

            g = self._flatten_grads()

            # Q^T g
            coeffs = self.nostalgia_Q.T @ g

            # Optional eigenvalue-aware scaling
            if self.scaling is not None:
                c_scaling = torch.median(self.scaling) + 1e-12
                if self.scaling.ndim == 1:
                    coeffs = coeffs * (self.scaling / (c_scaling + self.scaling))
                else:
                    coeffs = (self.scaling/(c_scaling + self.scaling)) @ coeffs

            # Q (Q^T g)
            projection = self.nostalgia_Q @ coeffs

            g_projected = g - projection

            self._unflatten_to_grads(g_projected)

            if self.writter is not None:
                grad_norm = torch.norm(g)
                proj_norm = torch.norm(g_projected)
                ratio = (proj_norm / (grad_norm + 1e-12)).item()

                if self.proj_ratio_ema is None:
                    self.proj_ratio_ema = ratio
                else:
                    self.proj_ratio_ema = (
                        self.ema_beta * self.proj_ratio_ema +
                        (1 - self.ema_beta) * ratio
                    )

                if self.step_count % self.log_every == 0:
                    self.writter.add_scalar(
                        'Nostalgia/Projection_Ratio',
                        ratio,
                        self.step_count
                    )
                    self.writter.add_scalar(
                        'Nostalgia/Projection_Ratio_EMA',
                        self.proj_ratio_ema,
                        self.step_count
                    )

        self.step_count += 1
        return self.base_optimizer.step(closure)

    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
