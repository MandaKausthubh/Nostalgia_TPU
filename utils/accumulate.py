import torch
from typing import Optional, Tuple


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

    # Ensure eigenvalues are diagonal matrices
    if Lambda_old.ndim == 1:
        Lambda_old = torch.diag(Lambda_old)
    if Lambda_new.ndim == 1:
        Lambda_new = torch.diag(Lambda_new)

    # --------------------------------------------------
    # Weighting coefficients for running average
    # --------------------------------------------------
    alpha = alpha_scaling *(t - 1) / t
    beta = beta_scaling / t

    Lambda_old = alpha * Lambda_old
    Lambda_new = beta * Lambda_new

    # --------------------------------------------------
    # Merge subspaces
    # --------------------------------------------------
    # M = [Q_old, Q_new] ∈ R^{N × (k_old + k_new)}
    M = torch.cat([Q_old, Q_new], dim=1)

    # Orthonormal basis of merged subspace
    # B ∈ R^{N × r}, r ≤ k_old + k_new
    if M.device.type == 'mps':
        # MPS backend has issues with "complete" mode
        print("[Accumulate] Moving M to CPU for QR decomposition... MPS compatibility issue")
        M = M.detach().cpu()

    print(M.device.type)

    B, _ = torch.linalg.qr(M, mode="reduced")

    B = B.to(Q_new.device)
    M = M.to(Q_new.device)

    # --------------------------------------------------
    # Project both Hessians into merged basis
    # --------------------------------------------------
    A_old = Q_old.T @ B          # (k_old × r)
    A_new = Q_new.T @ B          # (k_new × r)

    # Small matrix S ∈ R^{r × r}
    S = A_old.T @ Lambda_old @ A_old + A_new.T @ Lambda_new @ A_new

    # --------------------------------------------------
    # Eigendecomposition in small space
    # --------------------------------------------------
    if S.device.type == 'mps':
        print("[Accumulate] Moving S to CPU for eigendecomposition... MPS compatibility issue")
        S = S.detach().cpu()
    eigvals, eigvecs = torch.linalg.eigh(S)

    eigvals = eigvals.to(Q_new.device)
    eigvecs = eigvecs.to(Q_new.device)

    # Take top-k components
    idx = torch.argsort(eigvals, descending=True)[:k]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Lift eigenvectors back to full space
    Q_t = B @ eigvecs            # (N × k)

    return Q_t, eigvals
