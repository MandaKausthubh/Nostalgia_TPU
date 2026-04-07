from torch.func import grad, jvp, functional_call
from torch.nn.utils import parameters_to_vector
import torch_xla.core.xla_model as xm
import torch_xla
import torch_xla.runtime as xr

from torch.func import functional_call
import torch

# Flag to track if we're in TPU distributed mode
_TPU_DISTRIBUTED = None

def _is_tpu_distributed():
    """Check if we're running in TPU distributed mode (xmp.spawn)."""
    global _TPU_DISTRIBUTED
    if _TPU_DISTRIBUTED is None:
        try:
            # In xmp.spawn, xr.world_size() > 1 indicates distributed mode
            _TPU_DISTRIBUTED = xr.world_size() > 1
        except Exception:
            _TPU_DISTRIBUTED = False
    return _TPU_DISTRIBUTED

def hvp(params, vec, grad_fn):
    _, hv = jvp(
        grad_fn,
        (params,),
        (vec,),
    )
    return hv

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


# def hvp_flat(vec, params, model, inputs, targets, loss_fn):

#     # Forward
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)

#     # First gradient
#     # grads = torch.autograd.grad(
#     #     loss,
#     #     params.values(),
#     #     create_graph=True
#     # )
#     grads = torch.autograd.grad(
#         loss,
#         params.values(),
#         create_graph=True
#     )
#     grads = [xm.all_reduce(xm.REDUCE_SUM, g) for g in grads]
#     world_size = xr.world_size()
#     grads = [g / world_size for g in grads]

#     grad_vec = parameters_to_vector(grads)

#     # Dot product
#     grad_dot_vec = torch.dot(grad_vec, vec)

#     # Second gradient
#     hv = torch.autograd.grad(
#         grad_dot_vec,
#         params.values(),
#         retain_graph=False
#     )
    
#     hv = [xm.all_reduce(xm.REDUCE_SUM, h) for h in hv]
#     hv = [h / world_size for h in hv]

#     hv_vec = parameters_to_vector(hv).detach()

#     return hv_vec


def hvp_flat(vec, params, model, inputs, targets, loss_fn):
    """
    Computes Hessian-vector product Hv using functional parameters
    and avoids distributed ops inside autograd graph.
    """

    # ---- Forward with explicit params (FIX #2) ----
    inputs = model.preprocess_inputs(inputs)
    representations = functional_call(model.backbone, params, (inputs,))
    outputs = model.task_head_list[model.active_task](representations)
    loss = loss_fn(outputs, targets)

    # ---- First gradient ----
    grads = torch.autograd.grad(
        loss,
        tuple(params.values()),
        create_graph=True
    )

    grad_vec = torch.nn.utils.parameters_to_vector(grads)

    # ---- Dot product with vector ----
    grad_dot_vec = torch.dot(grad_vec, vec)

    # ---- Second gradient (HVP) ----
    hv = torch.autograd.grad(
        grad_dot_vec,
        tuple(params.values()),
        retain_graph=False
    )

    hv_vec = torch.nn.utils.parameters_to_vector(hv)

    # ---- Distributed reduction using TPU-native xm.all_reduce ----
    if _is_tpu_distributed():
        hv_vec = hv_vec.detach()
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

    for j in range(k):
        print(f"[Lanczos iteration]: {j}/{k}")
        q_j = Q[:, j].detach()

        v = hvp_fn(q_j)

        alpha_j = torch.dot(q_j, v)
        alpha[j] = alpha_j

        v = v - alpha_j * q_j

        if j > 0:
            v = v - beta[j-1] * Q[:, j-1]

        # 🔁 FULL reorthogonalization (more stable)
        for i in range(j + 1):
            qi = Q[:, i]
            v = v - torch.dot(qi, v) * qi

        if j < k - 1:
            beta_j = v.norm()
            beta[j] = beta_j

            if beta_j < 1e-10:
                print(f"Early exit at step {j}")
                Q = Q[:, :j+1]
                alpha = alpha[:j+1]
                beta = beta[:j]
                break

            Q[:, j+1] = v / beta_j

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
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # params = model.get_backbone_params_dict()
    params = {
        k: v.detach().requires_grad_(True)
        for k, v in model.get_backbone_params_dict().items()
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
    # print(f"Lanczos compute time is: {t2-t1}")

    # Compute eigenvalues from the tridiagonal matrix T
    # The Ritz values (eigenvalues of T) approximate Hessian eigenvalues
    # Offload to CPU for TPU/MPS compatibility with eigh
    offload_to_cpu = T.device.type in ('xla', 'mps')
    original_device = T.device

    if offload_to_cpu:
        T = T.detach().cpu()

    eigvals, eigvecs = torch.linalg.eigh(T)

    if offload_to_cpu:
        eigvals = eigvals.to(original_device)
        eigvecs = eigvecs.to(original_device)

    # Reorder Q_basis columns to match descending eigenvalue order
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    Q_basis = Q_basis[:, idx]

    return Q_basis.to(torch.float32), eigvals.to(torch.float32)
