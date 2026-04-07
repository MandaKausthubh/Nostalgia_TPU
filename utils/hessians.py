from torch.func import jvp, functional_call
from torch.nn.utils import parameters_to_vector
import torch_xla.core.xla_model as xm
import torch_xla
import torch_xla.runtime as xr

import torch

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
    Computes Hessian-vector product Hv.

    Key fix: we re-detach and re-enable grad on 'vec' so that each
    Lanczos call starts with a clean graph.  We also avoid retain_graph
    between the two autograd.grad calls, which was causing graph
    corruption on XLA after ~3 iterations.
    """
    # Ensure vec is properly detached from any previous graph
    vec = vec.detach().requires_grad_(False)

    # ---- Collect fresh params with grad enabled ----
    fresh_params = {
        name: p.detach().requires_grad_(True)
        for name, p in params.items()
    }

    # ---- Forward pass ----
    inputs_proc = model.preprocess_inputs(inputs)
    representations = functional_call(model.backbone, fresh_params, (inputs_proc,))
    outputs = model.task_head_list[model.active_task](representations)
    loss = loss_fn(outputs, targets)

    # ---- First gradient ----
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

    # ---- Dot product with the Lanczos query vector ----
    # vec lives on XLA; grad_vec also on XLA — fine.
    grad_dot_vec = (grad_vec * vec.detach()).sum()

    # ---- Second gradient (HVP) ----
    hv = torch.autograd.grad(
        grad_dot_vec,
        tuple(fresh_params.values()),
        retain_graph=False,   # release graph completely
        allow_unused=True
    )

    hv = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(hv, fresh_params.values())
    )

    hv_vec = torch.nn.utils.parameters_to_vector(hv).detach()

    # ---- Distributed reduction ----
    if _is_tpu_distributed():
        hv_vec = xm.all_reduce(xm.REDUCE_SUM, hv_vec)
        world_size = xr.world_size()
        hv_vec = hv_vec / world_size

    return hv_vec


def lanczos(hvp_fn, dim, k, device):
    dtype = torch.float32

    Q = torch.zeros(dim, k, device=device, dtype=dtype)
    alpha = torch.zeros(k, device=device, dtype=dtype)
    beta = torch.zeros(k-1, device=device, dtype=dtype)

    # initial vector
    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm()
    Q[:, 0] = q

    is_tpu = device.type == 'xla'
    rank = xr.global_ordinal() if _is_tpu_distributed() else 0

    for j in range(k):
        if rank == 0:
            print(f"[Lanczos iteration]: {j}/{k}")

        q_j = Q[:, j].detach()

        v = hvp_fn(q_j)

        # Force XLA graph execution and memory cleanup each step
        if is_tpu:
            xm.mark_step()

        # Detach v before the dot products so we don't accumulate graph
        v = v.detach()

        alpha_j = torch.dot(q_j, v)
        alpha[j] = alpha_j.detach()

        v = v - alpha_j.detach() * q_j

        if j > 0:
            v = v - beta[j-1].detach() * Q[:, j-1].detach()

        # Full reorthogonalization (more stable)
        for i in range(j + 1):
            qi = Q[:, i].detach()
            v = v - torch.dot(qi, v) * qi

        if j < k - 1:
            beta_j = v.norm()
            beta[j] = beta_j.detach()

            if beta_j.item() < 1e-10:
                if rank == 0:
                    print(f"Early exit at step {j}")
                Q = Q[:, :j+1]
                alpha = alpha[:j+1]
                beta = beta[:j]
                break

            Q[:, j+1] = (v / beta_j).detach()

        # Force step after each iteration to bound graph size on XLA
        if is_tpu:
            xm.mark_step()

    if rank == 0:
        print(f"Rank of Lanczos: {Q.shape[1]}")

    # build tridiagonal
    T = torch.diag(alpha)
    for i in range(len(beta)):
        T[i, i+1] = beta[i]
        T[i+1, i] = beta[i]

    return T, Q


def compute_Q_for_task(model, k, device, train_loader):
    model.eval()

    inputs, targets = next(iter(train_loader))
    inputs  = inputs.to(device)
    targets = targets.to(device)

    # Snapshot backbone params once; hvp_flat re-detaches them each call
    params = {
        name: p.detach()
        for name, p in model.get_backbone_params_dict().items()
    }
    param_dim = sum(p.numel() for p in params.values())

    def hvp_operator(v):
        return hvp_flat(
            v,
            params,
            model,
            inputs,
            targets,
            model.criterion
        )

    T, Q_basis = lanczos(
        hvp_operator,
        dim=param_dim,
        k=k,
        device=device
    )

    # Mark step to free memory before eigendecomposition
    if device.type == 'xla':
        xm.mark_step()

    # Offload to CPU for TPU/MPS compatibility with eigh
    offload_to_cpu = T.device.type in ('xla', 'mps')
    original_device = T.device

    if offload_to_cpu:
        T       = T.detach().cpu()
        Q_basis = Q_basis.detach().cpu()

    eigvals, eigvecs = torch.linalg.eigh(T)

    # Reorder to descending eigenvalue order
    idx     = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Lift Ritz vectors back to full parameter space: Q_basis @ eigvecs
    Q_full = Q_basis @ eigvecs      # (param_dim, k)

    if offload_to_cpu:
        Q_full  = Q_full.to(original_device)
        eigvals = eigvals.to(original_device)

    return Q_full.to(torch.float32), eigvals.to(torch.float32)
