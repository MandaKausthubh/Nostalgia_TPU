from torch.func import jvp, functional_call
from torch.nn.utils import parameters_to_vector
import torch_xla.core.xla_model as xm
import torch_xla
import torch_xla.runtime as xr

import torch
import gc

# Flag to track if we're in TPU distributed mode
_TPU_DISTRIBUTED = None

def _is_tpu_distributed():
    """Check if we're running in TPU distributed mode (xmp.spawn)."""
    global _TPU_DISTRIBUTED
    if _TPU_DISTRIBUTED is None:
        try:
            _TPU_DISTRIBUTED = xr.world_size() > 1
        except Exception:
            _TPU_DISTRIBUTED = False
    return _TPU_DISTRIBUTED


def flatten_params(params):
    return parameters_to_vector(params.values())


def unflatten(vec, params_template):
    new_params = {}
    pointer = 0
    for name, p in params_template.items():
        num = p.numel()
        new_params[name] = vec[pointer:pointer+num].view_as(p)
        pointer += num
    return new_params


def hvp_flat(vec, params, model, inputs, targets, loss_fn):
    """
    Computes Hessian-vector product Hv using double backward.

    Every call creates a FULLY INDEPENDENT computation graph:
    fresh params → forward → grad → dot → grad → detach.
    Nothing is retained across calls.
    """
    # Ensure vec is completely detached
    vec = vec.detach().clone()

    # Fresh params with grad enabled — no connection to prior graphs
    fresh_params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in params.items()
    }

    # Forward pass
    inputs_proc = model.preprocess_inputs(inputs.detach())
    representations = functional_call(model.backbone, fresh_params, (inputs_proc,))
    outputs = model.task_head_list[model.active_task](representations)
    loss = loss_fn(outputs, targets.detach())

    # First gradient (creates graph for second derivative)
    grads = torch.autograd.grad(
        loss,
        tuple(fresh_params.values()),
        create_graph=True,
        allow_unused=True
    )

    # Replace None grads with zeros
    grads = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, fresh_params.values())
    )

    grad_vec = torch.nn.utils.parameters_to_vector(grads)

    # Dot product with the query vector
    grad_dot_vec = (grad_vec * vec).sum()

    # Second gradient (HVP) — release graph completely
    hv = torch.autograd.grad(
        grad_dot_vec,
        tuple(fresh_params.values()),
        retain_graph=False,
        allow_unused=True
    )

    hv = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(hv, fresh_params.values())
    )

    # Detach and clone immediately to sever ALL graph references
    hv_vec = torch.nn.utils.parameters_to_vector(hv).detach().clone()

    # Distributed reduction
    if _is_tpu_distributed():
        hv_vec = xm.all_reduce(xm.REDUCE_SUM, hv_vec)
        world_size = xr.world_size()
        hv_vec = hv_vec / world_size

    return hv_vec


def lanczos(hvp_fn, dim, k, device):
    """
    Lanczos algorithm with XLA-safe memory management.

    Key insight: XLA builds a lazy computation graph. If we keep all
    intermediate tensors on XLA across iterations, the graph grows
    until it OOMs the device. The fix is to:
      1. xm.mark_step() to compile+execute the graph after each HVP
      2. Move Q, alpha, beta to CPU between iterations so XLA can't
         "see" them as part of the current graph
      3. Only bring the specific columns needed back to XLA for each step
    """
    dtype = torch.float32
    rank = xr.global_ordinal() if _is_tpu_distributed() else 0

    # Store Lanczos vectors on CPU to prevent XLA graph accumulation
    Q_cpu = torch.zeros(dim, k, dtype=dtype)       # CPU
    alpha_cpu = torch.zeros(k, dtype=dtype)         # CPU
    beta_cpu = torch.zeros(k - 1, dtype=dtype)      # CPU

    # Initial random vector — generate on device, normalize, move to CPU
    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm()
    xm.mark_step()
    Q_cpu[:, 0] = q.detach().cpu()
    del q

    actual_k = k  # Track actual rank (may be less if early exit)

    for j in range(k):
        if rank == 0:
            print(f"[Lanczos iteration]: {j}/{k}")

        # ── Step 1: bring current q_j to device ──────────────────────
        q_j = Q_cpu[:, j].to(device)

        # ── Step 2: compute HVP (fully self-contained graph) ─────────
        v = hvp_fn(q_j)

        # Force XLA to compile and execute THIS iteration's graph
        xm.mark_step()

        # ── Step 3: Lanczos recurrence (on device) ───────────────────
        # Detach v so subsequent ops don't extend the HVP graph
        v = v.detach()

        alpha_j = torch.dot(q_j, v)
        xm.mark_step()
        alpha_cpu[j] = alpha_j.item()

        v = v - alpha_j * q_j

        if j > 0:
            q_prev = Q_cpu[:, j - 1].to(device)
            v = v - beta_cpu[j - 1].item() * q_prev
            del q_prev

        # Full reorthogonalization — bring columns one at a time
        for i in range(j + 1):
            qi = Q_cpu[:, i].to(device)
            coeff = torch.dot(qi, v)
            v = v - coeff * qi
            del qi

        if j < k - 1:
            xm.mark_step()
            beta_j = v.norm()
            beta_val = beta_j.item()
            beta_cpu[j] = beta_val

            if beta_val < 1e-10:
                if rank == 0:
                    print(f"Early exit at step {j}")
                actual_k = j + 1
                break

            q_next = (v / beta_j).detach()
            xm.mark_step()
            Q_cpu[:, j + 1] = q_next.cpu()
            del q_next

        # Clean up device tensors from this iteration
        del v, q_j
        xm.mark_step()
        gc.collect()

    if rank == 0:
        print(f"Rank of Lanczos: {actual_k}")

    # Truncate to actual rank
    Q_cpu = Q_cpu[:, :actual_k]
    alpha_cpu = alpha_cpu[:actual_k]
    beta_cpu = beta_cpu[:max(1, actual_k - 1)]

    # Build tridiagonal matrix (on CPU — it's tiny: k×k)
    T = torch.diag(alpha_cpu)
    for i in range(len(beta_cpu)):
        if i + 1 < actual_k:
            T[i, i + 1] = beta_cpu[i]
            T[i + 1, i] = beta_cpu[i]

    return T, Q_cpu   # Both on CPU


def compute_Q_for_task(model, k, device, train_loader):
    model.eval()

    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Snapshot backbone params once — hvp_flat clones them each call
    params = {
        name: p.detach()
        for name, p in model.get_backbone_params_dict().items()
    }
    param_dim = sum(p.numel() for p in params.values())

    if xr.global_ordinal() == 0:
        print(f"[compute_Q_for_task] param_dim={param_dim:,}, k={k}")

    def hvp_operator(v):
        return hvp_flat(
            v,
            params,
            model,
            inputs,
            targets,
            model.criterion
        )

    # T and Q_cpu are returned on CPU
    T, Q_cpu = lanczos(
        hvp_operator,
        dim=param_dim,
        k=k,
        device=device
    )

    # Clean up the HVP closure references
    del inputs, targets, params
    xm.mark_step()
    gc.collect()

    T_cpu = T.detach().cpu().float()
    T_cpu = 0.5 * (T_cpu + T_cpu.T)

    # Eigendecomposition on CPU (T is small: k×k)
    eps = 1e-6
    T_cpu += eps * torch.eye(T_cpu.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(T_cpu)

    eigvals = eigvals.to(T.device)
    eigvecs = eigvecs.to(T.device)

    # Reorder to descending eigenvalue order
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Lift Ritz vectors back to full parameter space (CPU matmul — safe)
    Q_full = Q_cpu @ eigvecs      # (param_dim, k)

    # Move results to XLA device
    Q_full = Q_full.to(device=device, dtype=torch.float32)
    eigvals = eigvals.to(device=device, dtype=torch.float32)

    qtq = Q_full.T @ Q_full
    eye = torch.eye(qtq.shape[0], device=qtq.device)
    err = (qtq - eye).abs().max()
    xm.master_print(f"[compute_Q_for_task] Q_full orthogonality error: {err.item():.7e}")

    return Q_full, eigvals
