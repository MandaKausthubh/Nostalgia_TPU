"""
utils_GPU/hessians.py

GPU equivalent of utils/hessians.py.

Key differences from the TPU version:
  - All xm.mark_step() calls removed (not needed on CUDA; graphs execute eagerly).
  - xm.all_reduce → dist.all_reduce via utils_GPU.GPU helpers.
  - xr.world_size / xr.global_ordinal → dist helpers.
  - Lanczos no longer offloads vectors to CPU as a graph-growth workaround
    (PyTorch eager mode doesn't suffer from XLA lazy-graph accumulation).
    Vectors are kept on GPU directly for speed.
  - recover_eigenspace_from_factor is pure PyTorch — unchanged.

The Hessian computation math is IDENTICAL to the TPU version.
"""

import gc
import math

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call

from utils_GPU.GPU import world_size, global_rank, master_print


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def flatten_params(params):
    return parameters_to_vector(params.values())


def unflatten(vec, params_template):
    new_params = {}
    pointer = 0
    for name, p in params_template.items():
        num = p.numel()
        new_params[name] = vec[pointer:pointer + num].view_as(p)
        pointer += num
    return new_params


# ---------------------------------------------------------------------------
# Hessian-vector product
# ---------------------------------------------------------------------------

def hvp_flat(vec, params, model, inputs, targets, loss_fn):
    """
    Computes Hessian-vector product Hv using double backward.

    Every call creates a FULLY INDEPENDENT computation graph:
    fresh params → forward → grad → dot → grad → detach.
    Nothing is retained across calls.
    """
    vec = vec.detach().clone()

    fresh_params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in params.items()
    }

    inputs_proc = model.preprocess_inputs(inputs.detach())
    representations = functional_call(model.backbone, fresh_params, (inputs_proc,))
    outputs = model.task_head_list[model.active_task](representations)
    loss = loss_fn(outputs, targets.detach())

    grads = torch.autograd.grad(
        loss,
        tuple(fresh_params.values()),
        create_graph=True,
        allow_unused=True,
    )

    grads = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, fresh_params.values())
    )

    grad_vec = parameters_to_vector(grads)
    grad_dot_vec = (grad_vec * vec).sum()

    hv = torch.autograd.grad(
        grad_dot_vec,
        tuple(fresh_params.values()),
        retain_graph=False,
        allow_unused=True,
    )

    hv = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(hv, fresh_params.values())
    )

    hv_vec = parameters_to_vector(hv).detach().clone()

    # Distributed reduction across GPU ranks
    if dist.is_available() and dist.is_initialized() and world_size() > 1:
        dist.all_reduce(hv_vec, op=dist.ReduceOp.SUM)
        hv_vec = hv_vec / world_size()

    return hv_vec


# ---------------------------------------------------------------------------
# Lanczos algorithm — GPU version
# ---------------------------------------------------------------------------

def lanczos(hvp_fn, dim, k, device):
    """
    Lanczos algorithm for GPU.

    Unlike the TPU version, we do NOT need to offload vectors to CPU between
    iterations — PyTorch eager mode does not accumulate a lazy computation
    graph.  All Lanczos vectors live on GPU for maximum throughput.

    Full reorthogonalization is applied at every step for numerical stability.
    """
    dtype = torch.float32
    rank = global_rank()

    # Lanczos basis stored directly on GPU
    Q_dev = torch.zeros(dim, k, dtype=dtype, device=device)
    alpha = torch.zeros(k, dtype=dtype)
    beta  = torch.zeros(k - 1, dtype=dtype)

    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm()
    Q_dev[:, 0] = q

    actual_k = k

    for j in range(k):
        if rank == 0:
            print(f"[Lanczos GPU] step {j}/{k}")

        q_j = Q_dev[:, j]
        v = hvp_fn(q_j)
        v = v.detach()

        alpha_j = torch.dot(q_j, v)
        alpha[j] = alpha_j.item()
        v = v - alpha_j * q_j

        if j > 0:
            v = v - beta[j - 1].item() * Q_dev[:, j - 1]

        # Full reorthogonalization
        for i in range(j + 1):
            coeff = torch.dot(Q_dev[:, i], v)
            v = v - coeff * Q_dev[:, i]

        if j < k - 1:
            beta_j = v.norm()
            beta_val = beta_j.item()
            beta[j] = beta_val

            if beta_val < 1e-10:
                if rank == 0:
                    print(f"[Lanczos GPU] Early exit at step {j}")
                actual_k = j + 1
                break

            Q_dev[:, j + 1] = (v / beta_j).detach()

        del v
        gc.collect()

    if rank == 0:
        print(f"[Lanczos GPU] effective rank: {actual_k}")

    Q_dev   = Q_dev[:, :actual_k]
    alpha   = alpha[:actual_k]
    beta    = beta[:max(1, actual_k - 1)]

    # Build tridiagonal matrix on CPU (tiny: k×k)
    T = torch.diag(alpha)
    for i in range(len(beta)):
        if i + 1 < actual_k:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

    return T, Q_dev   # T on CPU, Q_dev on GPU


# ---------------------------------------------------------------------------
# Full Q computation for a single task
# ---------------------------------------------------------------------------

def compute_Q_for_task(model, k, device, train_loader):
    model.eval()

    inputs, targets = next(iter(train_loader))
    inputs  = inputs.to(device)
    targets = targets.to(device)

    params = {
        name: p.detach()
        for name, p in model.get_backbone_params_dict().items()
    }
    param_dim = sum(p.numel() for p in params.values())

    if global_rank() == 0:
        print(f"[compute_Q_for_task] param_dim={param_dim:,}, k={k}")

    def hvp_operator(v):
        return hvp_flat(v, params, model, inputs, targets, model.criterion)

    T, Q_dev = lanczos(hvp_operator, dim=param_dim, k=k, device=device)

    del inputs, targets, params
    gc.collect()

    T_cpu = T.detach().cpu().float()
    T_cpu = 0.5 * (T_cpu + T_cpu.T)

    eps = 1e-6
    T_cpu += eps * torch.eye(T_cpu.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(T_cpu)

    eigvecs = eigvecs.to(device)
    eigvals = eigvals.to(device)

    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Lift Ritz vectors back to full parameter space
    Q_full = Q_dev @ eigvecs       # (param_dim, k)
    Q_full = Q_full.to(dtype=torch.float32)
    eigvals = eigvals.to(dtype=torch.float32)

    return Q_full, eigvals


# ---------------------------------------------------------------------------
# Eigenspace recovery from PSD factor  (math identical to TPU version)
# ---------------------------------------------------------------------------

def recover_eigenspace_from_factor(
    F_global: torch.Tensor,
    k: int,
    eps: float = 1e-8,
):
    """
    Recover low-rank eigenspace from PSD factor matrix.

    Given F ∈ R^{n × m}, returns the top-k eigenspace of H ≈ F F^T
    via the Gram trick:  G = F^T F  (cheap m×m matrix).
    """
    G = F_global.T @ F_global
    G = 0.5 * (G + G.T)

    eigvals, V = torch.linalg.eigh(G)

    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    V = V[:, idx]

    valid = eigvals > eps
    eigvals = eigvals[valid]
    V = V[:, valid]

    if eigvals.numel() == 0:
        raise RuntimeError("No valid eigenvalues found in factor recovery.")

    k_eff = min(k, eigvals.shape[0])
    eigvals = eigvals[:k_eff]
    V = V[:, :k_eff]

    singular_vals = torch.sqrt(eigvals.clamp_min(eps))
    Q = F_global @ V
    Q = Q / singular_vals.unsqueeze(0)

    Q, _ = torch.linalg.qr(Q, mode="reduced")
    return Q, eigvals
