"""
utils_GPU/accumulate.py

GPU equivalent of utils/accumulate.py.

The math is identical.  The only changes:
  - _needs_cpu_offload: GPU tensors support all linalg ops natively,
    so we never need to offload (returns False for 'cuda' devices).
  - xm.master_print → utils_GPU.GPU.master_print (rank-0 conditional print).
"""

import torch
from typing import Optional, Tuple

from utils_GPU.GPU import master_print


def _needs_cpu_offload(tensor: torch.Tensor) -> bool:
    """
    On GPU, linalg ops (QR, eigh) run natively — no CPU offload required.
    We still offload for MPS which has limited linalg support.
    """
    return tensor.device.type == 'mps'


def _safe_qr(X: torch.Tensor) -> torch.Tensor:
    """Numerically stable QR, with CPU offload only when required (MPS)."""
    original_device = X.device
    original_dtype  = X.dtype

    if _needs_cpu_offload(X):
        X_cpu = X.detach().to("cpu", dtype=torch.float32)
        Q_cpu, _ = torch.linalg.qr(X_cpu, mode="reduced")
        return Q_cpu.to(device=original_device, dtype=original_dtype)

    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q


def _safe_eigh(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safe symmetric eigendecomposition, with CPU offload only for MPS."""
    original_device = S.device
    original_dtype  = S.dtype

    if _needs_cpu_offload(S):
        S_cpu = S.detach().to("cpu", dtype=torch.float32)
        eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(S_cpu)
        return (
            eigvals_cpu.to(device=original_device, dtype=original_dtype),
            eigvecs_cpu.to(device=original_device, dtype=original_dtype),
        )

    return torch.linalg.eigh(S)


def _diag_from_vector(v: torch.Tensor) -> torch.Tensor:
    if v.ndim == 2:
        return v
    return torch.diag(v)


def accumulate_hessian_eigenspace_stable(
    Q_old: Optional[torch.Tensor],
    Lambda_old: Optional[torch.Tensor],
    Q_new: torch.Tensor,
    Lambda_new: torch.Tensor,
    t: int,
    k: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stable ACCUMULATE step (GPU version).

    Maintains:
        H_bar_t = ((t-1)/t) * H_bar_{t-1} + (1/t) * H_t

    using a numerically stable low-rank eigenspace update.

    Args:
        Q_old:      (N, k_old) previous basis
        Lambda_old: (k_old,)   previous eigenvalues
        Q_new:      (N, k_new) new task basis
        Lambda_new: (k_new,)   new eigenvalues
        t:          task index (1-based)
        k:          rank cap
        eps:        numerical threshold

    Returns:
        Q_t:      (N, k_eff) orthonormal basis
        Lambda_t: (k_eff,)   eigenvalues
    """

    if Lambda_new.ndim == 2:
        Lambda_new = torch.diag(Lambda_new)

    Q_new = _safe_qr(Q_new)

    # First task
    if Q_old is None or Lambda_old is None:
        k_eff = min(k, Q_new.shape[1], Lambda_new.shape[0])

        qtq = Q_new[:, :k_eff].T @ Q_new[:, :k_eff]
        err = (
            qtq - torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
        ).abs().max()
        master_print(f"[ACCUMULATE GPU first] orth error = {err.item():.6e}")

        return Q_new[:, :k_eff], Lambda_new[:k_eff]

    if Lambda_old.ndim == 2:
        Lambda_old = torch.diag(Lambda_old)

    Q_old = _safe_qr(Q_old)

    alpha = (t - 1) / t
    beta  = 1.0 / t

    # Remove overlap
    overlap = Q_old.T @ Q_new
    Q_res   = Q_new - Q_old @ overlap
    res_norm = torch.norm(Q_res)

    if res_norm < eps:
        B = Q_old
    else:
        Q_res = _safe_qr(Q_res)
        B = torch.cat([Q_old, Q_res], dim=1)
        B = _safe_qr(B)

    # Small-space average Hessian
    A_old = B.T @ Q_old
    A_new = B.T @ Q_new

    Lambda_old_diag = _diag_from_vector(Lambda_old)
    Lambda_new_diag = _diag_from_vector(Lambda_new)

    S_old = A_old @ Lambda_old_diag @ A_old.T
    S_new = A_new @ Lambda_new_diag @ A_new.T
    S = alpha * S_old + beta * S_new
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = _safe_eigh(S)

    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    valid   = eigvals > eps
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    if eigvals.numel() == 0:
        master_print("[ACCUMULATE GPU warning] no valid eigvals, fallback to Q_old")
        return Q_old[:, :1], Lambda_old[:1]

    k_eff   = min(k, eigvals.shape[0])
    eigvals = eigvals[:k_eff]
    eigvecs = eigvecs[:, :k_eff]

    # Lift back to full space
    Q_t = B @ eigvecs
    Q_t = _safe_qr(Q_t)

    qtq = Q_t.T @ Q_t
    err = (
        qtq - torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
    ).abs().max()
    master_print(f"[ACCUMULATE GPU stable] orth error = {err.item():.6e}")

    return Q_t, eigvals
