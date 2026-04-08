import torch
from typing import Optional, Tuple


def _needs_cpu_offload(tensor: torch.Tensor) -> bool:
    """Check if tensor needs to be moved to CPU for numerical operations."""
    # TPU ('xla') and MPS may have issues with certain linalg operations
    return tensor.device.type in ('xla', 'mps')


def accumulate_hessian_eigenspace(
    Q_old: Optional[torch.Tensor],
    Lambda_old: Optional[torch.Tensor],
    Q_new: torch.Tensor,
    Lambda_new: torch.Tensor,
    t: int,
    k: int,
    alpha_scaling: float = 1.0,
    beta_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ACCUMULATE step from Nostalgia (Algorithm 2), flat-space version.

    Args:
        Q_old: (N, k_old) eigenvectors of previous average Hessian, or None
        Lambda_old: (k_old,) eigenvalues, or None
        Q_new: (N, k_new) eigenvectors of current task Hessian
        Lambda_new: (k_new,) eigenvalues of current task Hessian
        t: task index (1-based, t >= 1)
        k: rank to retain

    Returns:
        Q_t: (N, k) updated eigenvectors
        Lambda_t: (k,) updated eigenvalues
    """

    # --------------------------------------------------
    # First task: nothing to accumulate
    # --------------------------------------------------
    if Q_old is None or Lambda_old is None:
        return Q_new[:, :k], Lambda_new[:k]

    # Ensure Lambda are 1D eigenvalue vectors (not tridiagonal or diagonal matrices)
    if Lambda_old.ndim == 2:
        Lambda_old = Lambda_old.diag() if Lambda_old.shape[0] == Lambda_old.shape[1] else Lambda_old.squeeze()
    if Lambda_new.ndim == 2:
        Lambda_new = Lambda_new.diag() if Lambda_new.shape[0] == Lambda_new.shape[1] else Lambda_new.squeeze()

    # Convert to diagonal matrices for matrix operations
    Lambda_old_diag = torch.diag(Lambda_old)
    Lambda_new_diag = torch.diag(Lambda_new)

    # --------------------------------------------------
    # Weighting coefficients for running average
    # --------------------------------------------------
    alpha = alpha_scaling * (t - 1) / t
    beta = beta_scaling / t

    Lambda_old_diag = alpha * Lambda_old_diag
    Lambda_new_diag = beta * Lambda_new_diag

    # --------------------------------------------------
    # Merge subspaces
    # --------------------------------------------------
    # M = [Q_old, Q_new] ∈ R^{N × (k_old + k_new)}
    M = torch.cat([Q_old, Q_new], dim=1)

    # Orthonormal basis of merged subspace
    # B ∈ R^{N × r}, r ≤ k_old + k_new
    original_device = Q_new.device
    offload_to_cpu = _needs_cpu_offload(M)

    if offload_to_cpu:
        print(f"[Accumulate] Moving M to CPU for QR decomposition... {M.device.type} compatibility")
        M = M.detach().cpu()

    B, _ = torch.linalg.qr(M, mode="reduced")

    if offload_to_cpu:
        B = B.to(original_device)

    # --------------------------------------------------
    # Project both Hessians into merged basis
    # --------------------------------------------------
    A_old = Q_old.T @ B          # (k_old × r)
    A_new = Q_new.T @ B          # (k_new × r)

    # Small matrix S ∈ R^{r × r}
    S = A_old.T @ Lambda_old_diag @ A_old + A_new.T @ Lambda_new_diag @ A_new

    # --------------------------------------------------
    # Eigendecomposition in small space
    # --------------------------------------------------
    if offload_to_cpu:
        print(f"[Accumulate] Moving S to CPU for eigendecomposition... {original_device.type} compatibility")
        S = S.detach().cpu()

    eigvals, eigvecs = torch.linalg.eigh(S)

    if offload_to_cpu:
        eigvals = eigvals.to(original_device)
        eigvecs = eigvecs.to(original_device)

    # Take top-k components
    idx = torch.argsort(eigvals, descending=True)[:k]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Lift eigenvectors back to full space
    Q_t = B @ eigvecs            # (N × k)

    if offload_to_cpu:
        print(f"[Accumulate] Moving Q_t to CPU for QR decomposition [Normalisation]... {original_device.type} compatibility")
        Q_t = Q_t.detach().cpu()

    Q_t, _ = torch.linalg.qr(Q_t, mode="reduced")

    if offload_to_cpu:
        print(f"[Accumulate] Moving Q_t to Device for return... {original_device.type} compatibility")
        Q_t = Q_t.to(original_device)

    return Q_t, eigvals
