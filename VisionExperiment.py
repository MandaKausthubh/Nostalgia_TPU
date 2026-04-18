import os, time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torchvision.transforms as transforms

from dataclasses import dataclass, asdict
from tqdm.notebook import tqdm
from pytorch_adapt.datasets import DomainNet

import math

from utils.accumulate import *
from utils.hessians import *
from utils.logging import WandbLogger
from utils.nostalgia import *
from utils.TPU import *

from models.model import ContinualLearnerViT, NostalgiaConfig



def check_orthogonality(Q):
    qtq = Q.T @ Q
    eye = torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
    orth_err = (qtq - eye).abs().max()
    return orth_err.item()



class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config
        torch.manual_seed(self.config.seed)

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ])
        self.config.world_size = xr.world_size()

        if self.config.use_tpu:
            self.device = xm.xla_device()
        else:
            self.device = torch.device(self.config.device)

        print("Using device: ", self.device)

        self.model = ContinualLearnerViT(
            lr=self.config.lr,
            downstream_lr=self.config.downstream_lr,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,   # BUG FIX: was hardcoded 0.1
            use_peft=True,
            lora_modules=None,
            device=self.device,
            optimizer_type=self.config.base_optimizer,  # BUG FIX: was hardcoded "adamw"
        ).to(self.device)

        # Create datasets once — shared across epochs/domains
        self.domains = ['clipart', 'quickdraw', 'sketch', 'infograph', 'painting']
        self.train_loaders = {}
        self.test_loaders  = {}
        self.train_samplers = {}
        self.test_samplers  = {}

        self.train_datasets, self.test_datasets = {}, {}

        root = self.config.root_dir
        for domain in self.domains:
            xm.master_print(f"Preparing the dataset for: {domain}")
            self.train_datasets[domain] = DomainNet(
                root, domain, train=True, transform=self.transform
            )
            self.test_datasets[domain] = DomainNet(
                root, domain, train=False, transform=self.transform
            )

        # We'll create samplers/loaders dynamically per the current domain
        self.current_train_loader = None
        self.current_test_loader  = None
        self.current_train_sampler = None
        self.current_test_sampler   = None

        for domain in self.domains:
            self.model.add_task(domain, 345)

        self.log_global = "/kaggle/working/"
        # self.writer = SummaryWriter(logdir=os.path.join(self.log_global, self.config.log_dir))

        if xm.is_master_ordinal():
            self.writer = WandbLogger(
                project='Nostalgia',
                config=asdict(self.config)
            )
        else:
            self.writer = None
        self.finished_domains = []

        # EMA smoothing for training metrics
        self.ema_loss: Optional[float] = None
        self.ema_accuracy: Optional[float] = None
        self.ema_beta: float = 0.9  # Higher = smoother

        xm.master_print(f"Initialized on device: {self.device}, world_size={self.config.world_size}")


    def transform(self, image):
        image = self.augment(image)
        # BUG FIX: backbone.processor disappears after get_peft_model() wraps the backbone;
        # use the cached _processor reference set before PEFT is applied.
        pixel_values = self.model._processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values

    def update_Q_Lambda_for_single_domain(
        self, domain, rank
    ):
        self.prepare_dataloaders_for_domain(domain, rank)
        factor_blocks = []
        k = self.config.hessian_eigenspace_dim

        xm.master_print(f"[Rank {rank}] world_size = {xr.world_size()}")
        xm.master_print(f"[Rank {rank}] device = {self.device}")
        xm.master_print(f"\n\n=========== Computing Q Lambda for {domain} =================")

        assert self.current_train_loader is not None, "Train loader not prepared"
        assert self.current_train_sampler is not None, "Train sampler not prepared"

        for epoch in range(self.config.iterations_of_accumulation):
            xm.master_print(f"[Rank {rank}] Epoch Starting: {epoch}")
            self.current_train_sampler.set_epoch(epoch)
            Q_epoch, Lambda_epoch = compute_Q_for_task(
                model=self.model,
                device=self.device,
                k=self.config.hessian_eigenspace_dim,
                train_loader=self.current_train_loader,
            )

            Lambda_epoch = Lambda_epoch.clamp_min(0)
            sqrt_lambda = torch.sqrt(Lambda_epoch)
            F_epoch = Q_epoch * sqrt_lambda.unsqueeze(0)

            factor_blocks.append(F_epoch)
            xm.mark_step()

        F_local = torch.cat(factor_blocks, dim=1)
        F_local = F_local / math.sqrt(len(factor_blocks))
        G_local = F_local.T @ F_local
        G_global = xm.all_reduce(xm.REDUCE_SUM, G_local)/self.config.world_size
        xm.mark_step()

        eigvals, V = torch.linalg.eigh(G_global)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        V = V[:, idx]

        eps = 1e-9
        valid = eigvals > eps
        eigvals = eigvals[valid]
        V = V[:, valid]

        if eigvals.numel() == 0:
            xm.master_print("[WARNING] No valid eigenvalues found, falling back to Q_old")
            raise RuntimeError("No valid eigenvalues found")

        k_eff = min(k, eigvals.shape[0])
        eigvals = eigvals[:k_eff]
        V = V[:, :k_eff]

        singular_vals = torch.sqrt(eigvals)
        Q = F_local @ V
        Q = Q / singular_vals.unsqueeze(0)

        # CRITICAL FIX: Sync before QR to ensure Q is fully computed
        xm.mark_step()

        Q, _ = torch.linalg.qr(Q, mode="reduced")
        Lambda = eigvals

        # CRITICAL FIX: Ensure Q/Lambda are valid and synced before returning
        Q = Q.detach().clone().contiguous()
        Lambda = Lambda.detach().clone().contiguous()

        xm.mark_step()

        # Validation: check for NaN/Inf
        q_is_finite = torch.isfinite(Q).all().item()
        l_is_finite = torch.isfinite(Lambda).all().item()

        if rank == 0:
            if not (q_is_finite and l_is_finite):
                xm.master_print(f"[WARNING] Q or Lambda has non-finite values after single domain compute!")
                xm.master_print(f"  Q finite: {q_is_finite}")
                xm.master_print(f"  Lambda finite: {l_is_finite}")

            qtq = Q.T @ Q
            err = (
                qtq
                - torch.eye(
                    qtq.shape[0],
                    device=qtq.device,
                    dtype=qtq.dtype,
                )
            ).abs().max()

            xm.master_print(
                f"[MASTER] FINAL RESULTS\n"
                f"Q shape: {Q.shape}\n"
                f"Lambda shape: {Lambda.shape}\n"
                f"Max eigenvalue: {Lambda.max().item():.6e}\n"
                f"Orthogonality error: {err.item():.6e}\n"
                f"Q is finite: {q_is_finite}"
            )
        return Q, Lambda


    def update_Q_Lambda_for_all_past_domains(
        self,
        past_domains,
        rank,
    ):
        """
        Stable across-domain Hessian memory accumulation.

        Computes running average:
            H_bar_t = ((t-1)/t) H_bar_{t-1} + (1/t) H_t

        using weighted PSD factor merge.
        """

        Q_memory = None
        Lambda_memory = None

        k = self.config.hessian_eigenspace_dim

        for i, domain in enumerate(past_domains):

            xm.master_print(
                f"[Rank {rank}] Processing domain {domain}"
            )

            Q_new, Lambda_new = self.update_Q_Lambda_for_single_domain(
                domain, rank
            )

            if Q_memory is None or Lambda_memory is None:
                Q_memory = Q_new
                Lambda_memory = Lambda_new
            else:
                t = i + 1

                alpha = (t - 1) / t
                beta = 1.0 / t
                sqrt_old = torch.sqrt(Lambda_memory.clamp_min(0))
                sqrt_new = torch.sqrt(Lambda_new.clamp_min(0))

                F_old = (math.sqrt(alpha) * Q_memory * sqrt_old.unsqueeze(0))

                F_new = (math.sqrt(beta) * Q_new * sqrt_new.unsqueeze(0))
                F_global = torch.cat( [F_old, F_new], dim=1)

                Q_memory, Lambda_memory = recover_eigenspace_from_factor(
                    F_global=F_global,
                    k=k,
                )

            # CRITICAL FIX: Ensure Q_memory/Lambda_memory are valid after merge
            Q_memory = Q_memory.detach().clone().contiguous()
            Lambda_memory = Lambda_memory.detach().clone().contiguous()

            xm.mark_step()

            # Validation: check for NaN/Inf
            q_is_finite = torch.isfinite(Q_memory).all().item()
            l_is_finite = torch.isfinite(Lambda_memory).all().item()

            if rank == 0:
                if not (q_is_finite and l_is_finite):
                    xm.master_print(f"[WARNING] Q_memory or Lambda_memory has non-finite values after merge!")
                    xm.master_print(f"  Q finite: {q_is_finite}, Lambda finite: {l_is_finite}")

                err = check_orthogonality(Q_memory)

                xm.master_print(
                    f"[MASTER] After domain {domain}\n"
                    f"Q shape: {Q_memory.shape}\n"
                    f"Lambda shape: {Lambda_memory.shape}\n"
                    f"Orthogonality error: {err}\n"
                    f"Q is finite: {q_is_finite}"
                )

        # CRITICAL FIX: Final validation before returning
        if Q_memory is not None and Lambda_memory is not None:
            Q_memory = Q_memory.detach().clone().contiguous()
            Lambda_memory = Lambda_memory.detach().clone().contiguous()
            xm.mark_step()

            # All-rank validation
            q_is_finite = torch.isfinite(Q_memory).all().item()
            l_is_finite = torch.isfinite(Lambda_memory).all().item()
            if not (q_is_finite and l_is_finite):
                xm.master_print(f"[Rank {rank}] ERROR: Q/Lambda has non-finite values BEFORE returning from update_Q_Lambda_for_all_past_domains!")

        return Q_memory, Lambda_memory




    def prepare_dataloaders_for_domain(self, domain, rank):
        """Call this when switching to a new domain in continual training"""
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")

        if domain in self.train_loaders:
            self.current_train_loader = self.train_loaders[domain]
            self.current_test_loader  = self.test_loaders[domain]
            self.current_train_sampler = self.train_samplers[domain]
            self.current_test_sampler   = self.test_samplers[domain]
            return

        world_size = self.config.world_size

        train_ds = self.train_datasets[domain]
        test_ds  = self.test_datasets[domain]

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )

        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        per_core_bs = self.config.batch_size // world_size

        train_loader = pl.MpDeviceLoader(
            DataLoader(
                train_ds,
                batch_size=per_core_bs,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=False,
                drop_last=True
            ), self.device
        )

        test_loader = pl.MpDeviceLoader(
            DataLoader(
                test_ds,
                batch_size=per_core_bs,
                sampler=test_sampler,
                num_workers=self.config.num_workers,
                pin_memory=False,
                drop_last=False
            ), self.device
        )

        self.current_train_loader = train_loader
        self.current_test_loader  = test_loader
        self.current_train_sampler = train_sampler
        self.current_test_sampler   = test_sampler

        self.train_loaders[domain] = self.current_train_loader
        self.test_loaders[domain] = self.current_test_loader
        self.train_samplers[domain] = self.current_train_sampler
        self.test_samplers[domain] = self.current_test_sampler

        xm.master_print(f"Prepared loaders for domain '{domain}' (per-core bs={per_core_bs})")


    def compute_loss_accuracy(
        self, domain_name, inputs, targets, criterion
    ):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        # inputs  = self.model.preprocess_inputs(inputs)
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()

        return loss, accuracy


    @torch.no_grad()
    def evaluate_all_seen(self, criterion, rank):
        self.model.eval()
        results = {}
        for dom in self.finished_domains:
            self.prepare_dataloaders_for_domain(dom, rank) # you'll need to implement or cache

            assert self.current_test_loader is not None, f"Test loader not prepared for domain {dom}"

            total_loss, total_acc, count = 0., 0., 0
            self.model.set_active_task(dom)
            for batch in self.current_test_loader:  # or your parallel loader
                images, labels = batch
                loss, acc = self.compute_loss_accuracy(dom, images, labels, criterion)
                total_loss += loss.item() * labels.size(0)
                total_acc  += acc.item()  * labels.size(0)
                count += labels.size(0)

            global_loss_sum = xm.mesh_reduce(
                f"{dom}_loss_sum",
                total_loss,
                sum
            )
            global_acc_sum = xm.mesh_reduce(
                f"{dom}_acc_sum",
                total_acc,
                sum
            )
            global_count = xm.mesh_reduce(
                f"{dom}_count",
                count,
                sum
            )

            results[dom] = {
                'Test_Loss': global_loss_sum / global_count,
                'Test_Accuracy': (global_acc_sum / global_count) * 100
            }

        self.model.train()
        if not results:  # BUG FIX: guard against zero-division when no domains finished
            return {}, 0.0, 0.0
        avg_acc = sum(r['Test_Accuracy'] for r in results.values()) / len(results)
        avg_loss = sum(r['Test_Loss'] for r in results.values()) / len(results)
        xm.master_print(f"After domain {self.finished_domains[-1]} → Avg seen acc: {avg_acc:.2f}%")
        # log to tensorboard
        return results, avg_acc, avg_loss


    def train_taskhead(self, domain, epochs, rank):
        self.prepare_dataloaders_for_domain(domain, rank)

        assert self.current_train_loader is not None, "Train loader not prepared for task head training"

        self.model.set_active_task(domain)
        criterion = self.model.criterion
        self.model.task_head_list[domain].train()
        self.model.backbone.eval()

        optimizer = torch.optim.AdamW(
            self.model.task_head_list[domain].parameters(),
            lr=self.config.downstream_lr,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.current_train_loader) * epochs,
            eta_min=1e-6
        )

        xm.master_print(f"\n======= Training task head for {domain} ========")
        for epoch in range(epochs):
            step_iter = 0
            xm.master_print(f"\nStarting epoch: {epoch}")
            for inputs, targets in self.current_train_loader:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = self.model.backbone(inputs)
                outputs = self.model.task_head_list[domain](features)
                loss = criterion(outputs, targets)
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()
                if step_iter%10==0:
                    # xm.master_print(f'\tTask headtraining loss at step {step_iter}: {loss.item():.4f}')
                    pass
                step_iter+=1


    def train(self, rank):
        Q_curr, Lambda_curr = None, None

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain, rank)

        global_step = 1
        domain_list = []

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain, rank)
            criterion = self.model.criterion

            # BUG FIX: activate task BEFORE building the optimizer so that
            # head params have requires_grad=True and are included in head_params.
            self.model.set_active_task(domain)

            xm.master_print(f"\n====== Starting training on domain: {domain} =======")
            domain_list.append(domain)

            # BUG FIX: rebuild optimizer each domain so the newly-active head
            # params are tracked; also apply Q before constructing so the
            # NostalgiaOptimizer receives the correct subspace from the start.
            self.model.set_Q(Q_curr, scaling=None)
            optimizer = self.model.configure_optimizers(
                writter=self.writer,
                iteration=global_step,
            )
            if Q_curr is not None:
                print(
                    f"Q stats before setting in optimizer for domain {domain}:\n"
                    f"\tQ max abs: {Q_curr.abs().max().item()}\n"
                    f"\tQ norm: {Q_curr.norm().item()}\n"
                    f"\tQ is finite: {torch.isfinite(Q_curr).all().item()}\n"
                    f"\tQ max ortho error: {(Q_curr.T @ Q_curr - torch.eye(Q_curr.shape[1], device=Q_curr.device)).abs().max().item()}\n"
                    f"\tQ shape: {Q_curr.shape}\n"
                    f"\tQ device: {Q_curr.device}\n"
                    f"\tQ dtype: {Q_curr.dtype}\n"
                )
                optimizer.set_Q(Q_curr, None)

            self.finished_domains.append(domain)
            self.train_taskhead(domain, self.config.head_warmup_epochs, rank)

            # Switch backbone back to train() mode after task-head warmup
            # (train_taskhead puts backbone in eval() to freeze BN/dropout)
            self.model.backbone.train()
            self.model.set_active_task(domain)   # re-enable head grad after warmup

            # Reset EMA for new domain
            self.ema_loss = None
            self.ema_accuracy = None

            steps_per_epoch = len(self.current_train_loader)
            total_steps     = steps_per_epoch * self.config.num_epochs
            warmup_steps    = min(self.config.warmup_steps, total_steps // 10)

            # Linear warmup → Cosine decay schedule
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer.base_optimizer,
                start_factor = 0.1,
                end_factor   = 1.0,
                total_iters  = warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer,
                T_max   = max(1, total_steps - warmup_steps),
                eta_min = self.config.lr * 0.01,   # decay to 1% of peak LR
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer.base_optimizer,
                schedulers  = [warmup_scheduler, cosine_scheduler],
                milestones  = [warmup_steps],
            )

            xm.master_print(
                f"\n========== Full Training [{domain}]: "
                f"{total_steps} steps | warmup={warmup_steps} | "
                f"lr={self.config.lr:.1e} → {self.config.lr*0.01:.1e} =========="
            )

            # Calculate dynamic logging interval for ~50 logs per epoch
            steps_per_epoch = len(self.current_train_loader)
            log_interval = max(1, steps_per_epoch // 50)
            xm.master_print(f"Steps per epoch: {steps_per_epoch}, Logging every {log_interval} steps")

            for epoch in range(self.config.num_epochs):
                self.current_train_sampler.set_epoch(epoch)

                # Track running sums for this epoch (for efficient averaging)
                epoch_loss_sum = 0.0
                epoch_acc_sum = 0.0
                epoch_step_count = 0

                for step, batch in enumerate(self.current_train_loader):
                    images, labels = batch

                    optimizer.zero_grad()
                    loss, accuracy = self.compute_loss_accuracy(domain, images, labels, criterion)
                    loss.backward()

                    # ── Gradient clipping ─────────────────────────────────
                    # Collect all backbone + active head params with grads.
                    all_params = list(self.model.backbone.parameters()) + \
                                 list(self.model.task_head_list[domain].parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params,
                        max_norm = self.config.grad_clip_norm,
                    )

                    xm.optimizer_step(optimizer)
                    scheduler.step()

                    global_step += 1

                    # Update local running averages
                    loss_value = loss.item()
                    acc_value = accuracy.item()
                    epoch_loss_sum += loss_value
                    epoch_acc_sum += acc_value
                    epoch_step_count += 1

                    # Update local EMA
                    if self.ema_loss is None:
                        self.ema_loss = loss_value
                        self.ema_accuracy = acc_value
                    else:
                        self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * loss_value
                        self.ema_accuracy = self.ema_beta * self.ema_accuracy + (1 - self.ema_beta) * acc_value

                    if step % log_interval == 0:
                        # Aggregate EMA, running averages, and per-step values across all TPU cores
                        global_ema_loss  = xm.mesh_reduce('ema_loss',  self.ema_loss,                                    sum) / self.config.world_size
                        global_ema_acc   = xm.mesh_reduce('ema_acc',   self.ema_accuracy,                               sum) / self.config.world_size
                        global_avg_loss  = xm.mesh_reduce('avg_loss',  epoch_loss_sum / epoch_step_count,               sum) / self.config.world_size
                        global_avg_acc   = xm.mesh_reduce('avg_acc',   epoch_acc_sum  / epoch_step_count,               sum) / self.config.world_size
                        global_step_loss = xm.mesh_reduce('step_loss', loss_value,                                       sum) / self.config.world_size
                        global_step_acc  = xm.mesh_reduce('step_acc',  acc_value,                                        sum) / self.config.world_size
                        global_grad_norm = xm.mesh_reduce('grad_norm', grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, sum) / self.config.world_size

                        if xm.is_master_ordinal():
                            current_lr = scheduler.get_last_lr()[0]
                            # print(
                            #     f"Domain {domain} | Ep {epoch} | Step {step}/{steps_per_epoch} | "
                            #     f"Loss(step/ema) {global_step_loss:.4f}/{global_ema_loss:.4f} | "
                            #     f"Acc(step/ema) {global_step_acc*100:.2f}/{global_ema_acc*100:.2f}% | "
                            #     f"GradNorm {global_grad_norm:.3f} | LR {current_lr:.2e}"
                            # )
                            if self.writer is not None:
                                self.writer.add_scalars(domain, {
                                    'Training_Loss_Step'  : global_step_loss,
                                    'Training_Acc_Step'   : global_step_acc,
                                    'Training_Loss_EMA'   : global_ema_loss,
                                    'Training_Acc_EMA'    : global_ema_acc,
                                    'Training_Loss_Avg'   : global_avg_loss,
                                    'Training_Acc_Avg'    : global_avg_acc,
                                    'Grad_Norm'           : global_grad_norm,
                                    'LR'                  : current_lr,
                                }, global_step)

                    if step % (log_interval * 10) == 0 and step > 0:
                        # Run evaluation on all test datasets (less frequent)
                        result, acc, loss = self.evaluate_all_seen(criterion, rank)
                        xm.master_print(f"Validation Score | Loss: {result[domain]['Test_Loss']:.4f} | Accuracy: {result[domain]['Test_Accuracy']:.4f}%")
                        for eval_domain, metrics in result.items():
                            if self.writer is not None:
                                self.writer.add_scalars(eval_domain, metrics, global_step)
                        self.model.set_active_task(domain)   # Evaluation possible changes the head, so setting it back to the current task

            # ── Recompute Q/Lambda once, AFTER all epochs for this domain ──
            # This runs outside the epoch loop so Q is computed only once per
            # domain, then broadcast from rank 0 to every TPU core so all
            # ranks use the exact same subspace for the next domain.
            Q_curr, Lambda_curr = self.update_Q_Lambda_for_all_past_domains(domain_list, rank)

            # CRITICAL FIX: Ensure Q/Lambda are valid before broadcast
            xm.mark_step()

            # Validate Q before broadcast (only on rank 0 to avoid spam)
            if rank == 0:
                q_is_finite = torch.isfinite(Q_curr).all().item()
                l_is_finite = torch.isfinite(Lambda_curr).all().item()
                if not q_is_finite or not l_is_finite:
                    xm.master_print(f"[ERROR] Q/Lambda has non-finite values BEFORE broadcast!")
                    xm.master_print(f"  Q finite: {q_is_finite}, max abs: {Q_curr.abs().max().item()}")
                    xm.master_print(f"  Lambda finite: {l_is_finite}, max: {Lambda_curr.max().item()}")

            # Broadcast from rank 0 → all ranks so every core has the same Q.
            Q_curr, Lambda_curr = broadcast_Q_Lambda(Q_curr, Lambda_curr, src=0)

            # CRITICAL FIX: Sync after broadcast to ensure all ranks received the data
            xm.mark_step()

            # Validate Q after broadcast (all ranks should have the same value now)
            q_is_finite = torch.isfinite(Q_curr).all().item()
            if not q_is_finite:
                xm.master_print(f"[Rank {rank}] ERROR: Q has non-finite values AFTER broadcast!")

            # Re-orthogonalise after broadcast (numerical safety).
            Q_curr, _ = torch.linalg.qr(Q_curr, mode="reduced")

            # CRITICAL FIX: Final sync and validation before using Q in next domain
            Q_curr = Q_curr.detach().clone().contiguous()
            Lambda_curr = Lambda_curr.detach().clone().contiguous()

            xm.mark_step()

            assert Q_curr is not None and Lambda_curr is not None, "Q/Lambda computation failed"

            qtq = Q_curr.T @ Q_curr
            eye = torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
            orth_err = (qtq - eye).abs().max()
            xm.master_print("max orthogonality error =", orth_err.item())

            xm.master_print("Number of steps at the end of training: ", {global_step})

        xm.master_print("All domains completed")
