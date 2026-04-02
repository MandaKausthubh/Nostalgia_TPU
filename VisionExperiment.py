import os, time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torchvision.transforms as transforms
import torch_xla

from dataclasses import dataclass, asdict
from tqdm.notebook import tqdm
from pytorch_adapt.datasets import DomainNet

from utils.accumulate import *
from utils.hessians import *
from utils.logging import WandbLogger
from utils.nostalgia import *
from utils.TPU import *

from models.model import ContinualLearnerViT, NostalgiaConfig

class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config
        torch.manual_seed(self.config.seed)

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            # transforms.ToTensor(),
        ])
        self.config.world_size = xr.world_size()

        if self.config.use_tpu:
            self.device = xm.xla_device()
        else:
            self.device = torch.device(self.config.device)

        xm.master_print(self.device)

        self.model = ContinualLearnerViT(
            lr=self.config.lr, downstream_lr=self.config.downstream_lr, 
            lora_r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=0.1,
            use_peft=True, lora_modules = None, device=self.device, optimizer_type="adamw"
        ).to(self.device)

        def vit_transform(image):
            image = self.augment(image)
            pixel_values = self.model.backbone.processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

            return pixel_values

        self.transform = vit_transform

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
        xm.master_print(f"Initialized on device: {self.device}, world_size={self.config.world_size}")



    def update_Q_Lambda_for_single_domain(
        self, domain, rank
    ):
        self.prepare_dataloaders_for_domain(domain, rank)
        Q, Lambda = None, None

        xm.master_print(f"[Rank {rank}] world_size = {xr.world_size()}")
        xm.master_print(f"[Rank {rank}] device = {self.device}")
        xm.master_print(f"\n\n=========== Computing Q Lambda for {domain} =================")

        for epoch in range(self.config.iterations_of_accumulation):

            xm.master_print(f"[Rank {rank}] Epoch Starting: {epoch}")
            self.current_train_sampler.set_epoch(epoch)
            t1 = time.time()
            Q_new, Lambda_new = compute_Q_for_task(
                model=self.model, device=self.device,
                k = self.config.hessian_eigenspace_dim,
                train_loader = self.current_train_loader,
            )
            t2 = time.time()

            if rank == 0:
                Q, Lambda = accumulate_hessian_eigenspace(
                    Q_old=Q, Lambda_old=Lambda,
                    Q_new=Q_new.to(device=self.device), Lambda_new=Lambda_new.to(device=self.device),
                    t=(epoch+1), k=self.config.hessian_eigenspace_dim,
                )

            broadcast_Q_Lambda(Q, Lambda)
            torch_xla.sync()
            t3 = time.time()
            if rank==0:
                xm.master_print(
                    f"[MASTER]Q Lambda calculation domain: {domain} | epoch: {epoch} | "
                    f"Computing Q/L = {t2-t1:.6f} | Accumulate = {t3-t2:.6f}"
                    f"Q shape: {Q.shape}"
                )
        return Q, Lambda


    def update_Q_Lambda_for_all_past_domains(
        self, past_domains, rank
    ):
        Q, Lambda = None, None
        for i, domain in enumerate(past_domains):
            Q_new, Lambda_new = self.update_Q_Lambda_for_single_domain(domain, rank)
            Q, Lambda = accumulate_hessian_eigenspace(
                Q_old=Q, Lambda_old=Lambda,
                Q_new=Q_new, Lambda_new=Lambda_new,
                t = (i+1), alpha_scaling=1.0+1e-2, beta_scaling=1.0-1e-2,
                k = self.config.hessian_eigenspace_dim,
            )
        return Q, Lambda


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
        avg_acc = sum(r['Test_Accuracy'] for r in results.values()) / len(results)
        avg_loss = sum(r['Test_Loss'] for r in results.values()) / len(results)
        xm.master_print(f"After domain {self.finished_domains[-1]} → Avg seen acc: {avg_acc:.2f}%")
        # log to tensorboard
        return results, avg_acc, avg_loss


    def train_taskhead(self, domain, epochs, rank):
        self.prepare_dataloaders_for_domain(domain, rank)
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
                    xm.master_print(f'\tTask headtraining loss at step {step_iter}: {loss.item():.4f}')
                step_iter+=1


    def train(self, rank):
        Q_curr, Lambda_curr = None, None

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain, rank)

        global_step = 1
        self.model.set_Q(Q_curr, scaling=None)

        optimizer = self.model.configure_optimizers(
            writter = self.writer,
            iteration = global_step,   # REQUIRED FOR LOGGING
        )

        domain_list = []

        for domain in self.domains:
            self.prepare_dataloaders_for_domain(domain, rank)
            criterion = self.model.criterion

            self.model.set_active_task(domain)

            xm.master_print(f"\n====== Starting training on domain: {domain} =======")
            domain_list.append(domain)

            if Q_curr is not None:
                optimizer.set_Q(Q_curr, None)   # Ignoring scaling for now

            self.finished_domains.append(domain)
            self.train_taskhead(domain, self.config.head_warmup_epochs, rank)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer,
                T_max=len(self.current_train_loader) * self.config.num_epochs,
                eta_min=1e-6
            )

            xm.master_print(f"\n========== Full Training: ===========")

            for epoch in range(self.config.num_epochs):
                self.current_train_sampler.set_epoch(epoch)

                for step, batch in enumerate(self.current_train_loader):
                    images, labels = batch
                    loss, accuracy = self.compute_loss_accuracy(domain, images, labels, criterion)

                    optimizer.zero_grad()
                    loss.backward()
                    xm.optimizer_step(optimizer)
                    scheduler.step()

                    global_step += 1

                    if step % 100 == 0 and xm.is_master_ordinal():
                        print(f"Domain {domain} | Ep {epoch} | Step {global_step} | Loss {loss.item():.4f} | Acc {accuracy.item()*100:.4f}%")
                        if self.writer is not None:
                            self.writer.add_scalars(domain, {
                                f'Training_Loss'    : loss.item(),
                                f'Training_Accuracy': accuracy.item(),
                                f'LR'               : scheduler.get_last_lr()[0]
                            }, global_step)

                    if step % 200 == 0:
                        # Run evaluation on all test datasets (past, current, and future)
                        # if xm.is_master_ordinal():
                        result, acc, loss = self.evaluate_all_seen(criterion, rank)
                        xm.master_print(f"Validation Score | Loss: {result[domain]['Test_Loss']:.4f} | Accuracy: {result[domain]['Test_Accuracy']:.4f}%")
                        for eval_domain, metrics in result.items():
                            if self.writer is not None:
                                self.writer.add_scalars(eval_domain, metrics, global_step)
                        self.model.set_active_task(domain)   # Evaluation possible changes the head, so setting it back to the current task

            # if xm.is_master_ordinal() or True:

                # Full recomputation
            Q_curr, Lambda_curr = self.update_Q_Lambda_for_all_past_domains(domain_list, rank)

                # For EMA style updates
                # Q_curr, Lambda_curr = self.update_Q_Lambda_for_all_past_domains(domain_list, rank)
                # Q_curr = (self.config.gamma * Q_new) + ((1-self.config.gamma) * Q_curr)
                # Lamda_curr = (self.config.gamma * Lambda_new) + ((1-self.config.gamma) * Lambda_curr)

                # For single merging
                # Q_new, Lambda_new = update_Q_Lambda_for_single_domain(domain, rank)
                # Q_curr, Lambda_curr = accumulate_hessian_eigenspace(
                #     Q_old=Q_curr, Lambda_old = Lambda_curr,
                #     Q_new=Q_new,  Lambda_new = Lambda_new,
                #     t=len(self.finished_domains), k=self.config.hessian_eigenspace_dim
                # )

                # Q_curr = broadcast_tensor(Q_curr)
                # Lambda_curr = broadcast_tensor(Lambda_curr)
            xm.master_print("Number of steps at the end of training: ", {global_step})

        xm.master_print("All domains completed")
