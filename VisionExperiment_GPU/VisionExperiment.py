"""
VisionExperiment_GPU/VisionExperiment.py

GPU equivalent of VisionExperiment.py.

Every torch_xla / XLA-specific call has been replaced:

  XLA / TPU                          GPU / NCCL
  ─────────────────────────────────  ─────────────────────────────────────
  import torch_xla.*                 import torch.distributed as dist
  xm.xla_device()                    torch.device("cuda:<rank>")
  xm.optimizer_step(opt)             opt.step()
  xm.mark_step()                     (removed — not needed in eager mode)
  xm.master_print(...)               master_print(...)  (rank-0 guard)
  xm.mesh_reduce(name, val, sum)     dist.all_reduce then /world_size
  xm.is_master_ordinal()             dist.get_rank() == 0
  xm.all_reduce(REDUCE_SUM, t)       dist.all_reduce(t, SUM)
  xr.world_size()                    dist.get_world_size()
  xr.global_ordinal()                dist.get_rank()
  pl.MpDeviceLoader(loader, device)  plain DataLoader (CUDA handles transfer)
  broadcast_Q_Lambda (XLA trick)     broadcast_Q_Lambda (dist.broadcast)

The Hessian math, Nostalgia projection, and training logic are UNCHANGED.
"""

import os
import math
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

from pytorch_adapt.datasets import DomainNet

from utils_GPU.accumulate import accumulate_hessian_eigenspace_stable
from utils_GPU.hessians import compute_Q_for_task, recover_eigenspace_from_factor
from utils_GPU.logging import WandbLogger
from utils_GPU.nostalgia import NostalgiaOptimizer
from utils_GPU.GPU import (
    world_size, global_rank, is_master, master_print,
    barrier, broadcast_Q_Lambda, mesh_reduce,
)

from models.model import ContinualLearnerViT, NostalgiaConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_orthogonality(Q: torch.Tensor) -> float:
    qtq = Q.T @ Q
    eye = torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
    return (qtq - eye).abs().max().item()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig, rank: int):
        self.config = config
        self.rank   = rank
        torch.manual_seed(config.seed + rank)  # per-rank seed diversity

        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ])

        master_print(f"Using device: {self.device}, world_size={world_size()}")

        self.model = ContinualLearnerViT(
            lr=config.lr,
            downstream_lr=config.downstream_lr,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_peft=True,
            lora_modules=None,
            device=self.device,
            optimizer_type=config.base_optimizer,
        ).to(self.device)

        # Wrap with DDP for gradient synchronisation
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[rank],
            find_unused_parameters=True,
        )
        self._module = self.model.module   # unwrapped model for direct access

        # Datasets (created once; samplers/loaders built per-domain on demand)
        self.domains = ['clipart', 'quickdraw', 'sketch', 'infograph', 'painting']
        self.train_loaders:  dict = {}
        self.test_loaders:   dict = {}
        self.train_samplers: dict = {}
        self.test_samplers:  dict = {}
        self.train_datasets: dict = {}
        self.test_datasets:  dict = {}

        root = config.root_dir
        for domain in self.domains:
            master_print(f"Preparing dataset for: {domain}")
            self.train_datasets[domain] = DomainNet(
                root, domain, train=True,  transform=self.transform
            )
            self.test_datasets[domain] = DomainNet(
                root, domain, train=False, transform=self.transform
            )

        self.current_train_loader:   Optional[DataLoader] = None
        self.current_test_loader:    Optional[DataLoader] = None
        self.current_train_sampler:  Optional[DistributedSampler] = None
        self.current_test_sampler:   Optional[DistributedSampler] = None

        for domain in self.domains:
            self._module.add_task(domain, 345)

        # Logger (W&B on master only)
        if is_master():
            self.writer = WandbLogger(
                project='Nostalgia-GPU',
                config=asdict(config),
            )
        else:
            self.writer = None

        self.finished_domains: list = []

        self.ema_loss:     Optional[float] = None
        self.ema_accuracy: Optional[float] = None
        self.ema_beta: float = 0.9

        master_print(f"Initialized on device: {self.device}")

    # -----------------------------------------------------------------------
    # Transform
    # -----------------------------------------------------------------------

    def transform(self, image):
        image = self.augment(image)
        pixel_values = self._module._processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)
        return pixel_values

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------

    def prepare_dataloaders_for_domain(self, domain: str):
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")

        if domain in self.train_loaders:
            self.current_train_loader  = self.train_loaders[domain]
            self.current_test_loader   = self.test_loaders[domain]
            self.current_train_sampler = self.train_samplers[domain]
            self.current_test_sampler  = self.test_samplers[domain]
            return

        ws = world_size()
        per_core_bs = self.config.batch_size // ws

        train_sampler = DistributedSampler(
            self.train_datasets[domain],
            num_replicas=ws,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            self.test_datasets[domain],
            num_replicas=ws,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )

        train_loader = DataLoader(
            self.train_datasets[domain],
            batch_size=per_core_bs,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            self.test_datasets[domain],
            batch_size=per_core_bs,
            sampler=test_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.current_train_loader  = train_loader
        self.current_test_loader   = test_loader
        self.current_train_sampler = train_sampler
        self.current_test_sampler  = test_sampler

        self.train_loaders[domain]  = train_loader
        self.test_loaders[domain]   = test_loader
        self.train_samplers[domain] = train_sampler
        self.test_samplers[domain]  = test_sampler

        master_print(f"Prepared loaders for '{domain}' (per-GPU bs={per_core_bs})")

    # -----------------------------------------------------------------------
    # Forward / loss
    # -----------------------------------------------------------------------

    def compute_loss_accuracy(self, domain_name, inputs, targets, criterion):
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        loss     = criterion(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        return loss, accuracy

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_all_seen(self, criterion):
        self.model.eval()
        results = {}

        for dom in self.finished_domains:
            self.prepare_dataloaders_for_domain(dom)
            assert self.current_test_loader is not None

            total_loss, total_acc, count = 0.0, 0.0, 0
            self._module.set_active_task(dom)

            for images, labels in self.current_test_loader:
                loss, acc = self.compute_loss_accuracy(dom, images, labels, criterion)
                total_loss += loss.item() * labels.size(0)
                total_acc  += acc.item()  * labels.size(0)
                count      += labels.size(0)

            # All-reduce scalars across GPUs
            global_loss_sum = mesh_reduce(f"{dom}_loss_sum", total_loss, sum)
            global_acc_sum  = mesh_reduce(f"{dom}_acc_sum",  total_acc,  sum)
            global_count    = mesh_reduce(f"{dom}_count",    count,      sum)

            results[dom] = {
                'Test_Loss':     global_loss_sum / global_count,
                'Test_Accuracy': (global_acc_sum / global_count) * 100,
            }

        self.model.train()

        if not results:
            return {}, 0.0, 0.0

        avg_acc  = sum(r['Test_Accuracy'] for r in results.values()) / len(results)
        avg_loss = sum(r['Test_Loss']     for r in results.values()) / len(results)
        master_print(f"After {self.finished_domains[-1]} → Avg seen acc: {avg_acc:.2f}%")
        return results, avg_acc, avg_loss

    # -----------------------------------------------------------------------
    # Q / Lambda computation  (single domain)
    # -----------------------------------------------------------------------

    def update_Q_Lambda_for_single_domain(self, domain: str):
        self.prepare_dataloaders_for_domain(domain)
        factor_blocks = []
        k = self.config.hessian_eigenspace_dim

        master_print(f"\n=========== Computing Q/Lambda for {domain} ===========")
        assert self.current_train_loader  is not None
        assert self.current_train_sampler is not None

        for epoch in range(self.config.iterations_of_accumulation):
            master_print(f"  [Hessian epoch] {epoch}/{self.config.iterations_of_accumulation}")
            self.current_train_sampler.set_epoch(epoch)

            Q_epoch, Lambda_epoch = compute_Q_for_task(
                model=self._module,
                device=self.device,
                k=k,
                train_loader=self.current_train_loader,
            )

            Lambda_epoch = Lambda_epoch.clamp_min(0)
            sqrt_lambda  = torch.sqrt(Lambda_epoch)
            F_epoch      = Q_epoch * sqrt_lambda.unsqueeze(0)
            factor_blocks.append(F_epoch)

        F_local  = torch.cat(factor_blocks, dim=1)
        F_local  = F_local / math.sqrt(len(factor_blocks))
        G_local  = F_local.T @ F_local

        # Average Gram matrix across GPUs
        dist.all_reduce(G_local, op=dist.ReduceOp.SUM)
        G_global = G_local / world_size()

        eigvals, V = torch.linalg.eigh(G_global)
        idx    = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        V       = V[:, idx]

        eps   = 1e-9
        valid  = eigvals > eps
        eigvals = eigvals[valid]
        V       = V[:, valid]

        if eigvals.numel() == 0:
            raise RuntimeError(f"[{domain}] No valid eigenvalues found")

        k_eff   = min(k, eigvals.shape[0])
        eigvals = eigvals[:k_eff]
        V       = V[:, :k_eff]

        singular_vals = torch.sqrt(eigvals)
        Q = F_local @ V
        Q = Q / singular_vals.unsqueeze(0)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        Lambda = eigvals

        if self.rank == 0:
            err = check_orthogonality(Q)
            master_print(
                f"[MASTER] {domain}: Q={Q.shape}, Lambda={Lambda.shape}, "
                f"max_eig={Lambda.max().item():.4e}, orth_err={err:.4e}"
            )

        return Q, Lambda

    # -----------------------------------------------------------------------
    # Q / Lambda computation  (all past domains)
    # -----------------------------------------------------------------------

    def update_Q_Lambda_for_all_past_domains(self, past_domains):
        Q_memory      = None
        Lambda_memory = None
        k = self.config.hessian_eigenspace_dim

        for i, domain in enumerate(past_domains):
            master_print(f"[Hessian] Processing domain {domain}")

            Q_new, Lambda_new = self.update_Q_Lambda_for_single_domain(domain)

            if Q_memory is None or Lambda_memory is None:
                Q_memory      = Q_new
                Lambda_memory = Lambda_new
            else:
                t     = i + 1
                alpha = (t - 1) / t
                beta  = 1.0 / t

                sqrt_old = torch.sqrt(Lambda_memory.clamp_min(0))
                sqrt_new = torch.sqrt(Lambda_new.clamp_min(0))

                F_old    = math.sqrt(alpha) * Q_memory * sqrt_old.unsqueeze(0)
                F_new    = math.sqrt(beta)  * Q_new    * sqrt_new.unsqueeze(0)
                F_global = torch.cat([F_old, F_new], dim=1)

                Q_memory, Lambda_memory = recover_eigenspace_from_factor(
                    F_global=F_global,
                    k=k,
                )

            if self.rank == 0:
                err = check_orthogonality(Q_memory)
                master_print(
                    f"[MASTER] After {domain}: Q={Q_memory.shape}, "
                    f"Lambda={Lambda_memory.shape}, orth_err={err:.4e}"
                )

        return Q_memory, Lambda_memory

    # -----------------------------------------------------------------------
    # Task-head warmup
    # -----------------------------------------------------------------------

    def train_taskhead(self, domain: str, epochs: int):
        self.prepare_dataloaders_for_domain(domain)
        assert self.current_train_loader is not None

        self._module.set_active_task(domain)
        criterion = self._module.criterion
        self._module.task_head_list[domain].train()
        self._module.backbone.eval()

        optimizer = torch.optim.AdamW(
            self._module.task_head_list[domain].parameters(),
            lr=self.config.downstream_lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.current_train_loader) * epochs,
            eta_min=1e-6,
        )

        master_print(f"\n======= Task-head warmup: {domain} =======")
        for epoch in range(epochs):
            master_print(f"  [Warmup epoch] {epoch}")
            for step, (inputs, targets) in enumerate(self.current_train_loader):
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = self._module.backbone(inputs)
                outputs = self._module.task_head_list[domain](features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self):
        Q_curr, Lambda_curr = None, None

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain)

        global_step = 1
        domain_list = []

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain)
            criterion = self._module.criterion

            # Activate task BEFORE building optimizer so head params are included
            self._module.set_active_task(domain)

            master_print(f"\n====== Training domain: {domain} ======")
            domain_list.append(domain)

            # Rebuild optimizer each domain (fresh head params, possibly new Q)
            self._module.set_Q(Q_curr, scaling=None)
            optimizer = self._module.configure_optimizers(
                writter=self.writer,
                iteration=global_step,
            )
            if Q_curr is not None:
                optimizer.set_Q(Q_curr, None)

            self.finished_domains.append(domain)
            self.train_taskhead(domain, self.config.head_warmup_epochs)

            # Switch backbone back to train mode after task-head warmup
            self._module.backbone.train()
            self._module.set_active_task(domain)

            # Reset EMA for new domain
            self.ema_loss     = None
            self.ema_accuracy = None

            steps_per_epoch = len(self.current_train_loader)
            total_steps     = steps_per_epoch * self.config.num_epochs
            warmup_steps    = min(self.config.warmup_steps, total_steps // 10)

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer.base_optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=self.config.lr * 0.01,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer.base_optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

            master_print(
                f"\n========== Full Training [{domain}]: {total_steps} steps | "
                f"warmup={warmup_steps} | lr={self.config.lr:.1e} =========="
            )

            log_interval = max(1, steps_per_epoch // 50)

            for epoch in range(self.config.num_epochs):
                self.current_train_sampler.set_epoch(epoch)

                epoch_loss_sum  = 0.0
                epoch_acc_sum   = 0.0
                epoch_step_count = 0

                for step, batch in enumerate(self.current_train_loader):
                    images, labels = batch

                    optimizer.zero_grad()
                    loss, accuracy = self.compute_loss_accuracy(domain, images, labels, criterion)
                    loss.backward()

                    all_params = (
                        list(self._module.backbone.parameters()) +
                        list(self._module.task_head_list[domain].parameters())
                    )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params,
                        max_norm=self.config.grad_clip_norm,
                    )

                    # GPU: plain optimizer.step() — no xm.optimizer_step needed
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    loss_value = loss.item()
                    acc_value  = accuracy.item()
                    epoch_loss_sum   += loss_value
                    epoch_acc_sum    += acc_value
                    epoch_step_count += 1

                    if self.ema_loss is None:
                        self.ema_loss     = loss_value
                        self.ema_accuracy = acc_value
                    else:
                        self.ema_loss     = self.ema_beta * self.ema_loss     + (1 - self.ema_beta) * loss_value
                        self.ema_accuracy = self.ema_beta * self.ema_accuracy + (1 - self.ema_beta) * acc_value

                    if step % log_interval == 0:
                        global_ema_loss  = mesh_reduce('ema_loss',  self.ema_loss,                          sum) / world_size()
                        global_ema_acc   = mesh_reduce('ema_acc',   self.ema_accuracy,                      sum) / world_size()
                        global_avg_loss  = mesh_reduce('avg_loss',  epoch_loss_sum / epoch_step_count,      sum) / world_size()
                        global_avg_acc   = mesh_reduce('avg_acc',   epoch_acc_sum  / epoch_step_count,      sum) / world_size()
                        global_step_loss = mesh_reduce('step_loss', loss_value,                              sum) / world_size()
                        global_step_acc  = mesh_reduce('step_acc',  acc_value,                              sum) / world_size()
                        global_grad_norm = mesh_reduce('grad_norm', grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, sum) / world_size()

                        if is_master() and self.writer is not None:
                            current_lr = scheduler.get_last_lr()[0]
                            self.writer.add_scalars(domain, {
                                'Training_Loss_Step': global_step_loss,
                                'Training_Acc_Step':  global_step_acc,
                                'Training_Loss_EMA':  global_ema_loss,
                                'Training_Acc_EMA':   global_ema_acc,
                                'Training_Loss_Avg':  global_avg_loss,
                                'Training_Acc_Avg':   global_avg_acc,
                                'Grad_Norm':          global_grad_norm,
                                'LR':                 current_lr,
                            }, global_step)

                    if step % (log_interval * 10) == 0 and step > 0:
                        result, acc, loss_val = self.evaluate_all_seen(criterion)
                        if result:
                            master_print(
                                f"Validation | Loss: {result[domain]['Test_Loss']:.4f} "
                                f"| Acc: {result[domain]['Test_Accuracy']:.2f}%"
                            )
                        if is_master() and self.writer is not None:
                            for eval_domain, metrics in result.items():
                                self.writer.add_scalars(eval_domain, metrics, global_step)
                        self._module.set_active_task(domain)

            # ── After all epochs for this domain: update Q/Lambda ──────────
            Q_curr, Lambda_curr = self.update_Q_Lambda_for_all_past_domains(domain_list)

            # Broadcast Q/Lambda from rank 0 to all GPUs
            Q_curr, Lambda_curr = broadcast_Q_Lambda(Q_curr, Lambda_curr, src=0)
            barrier()

            assert Q_curr is not None and Lambda_curr is not None

            if is_master():
                orth_err = check_orthogonality(Q_curr)
                master_print(f"[Q sync] max orth error = {orth_err:.4e}")

            master_print(f"Total steps so far: {global_step}")

        master_print("All domains completed.")
