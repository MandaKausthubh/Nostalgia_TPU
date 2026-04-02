from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.nn.utils.stateless import functional_call
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.hessians import *
from utils.TPU import *


@dataclass
class NostalgiaConfig:
    mode: str = 'nostalgia'
    seed: int = 42
    root_dir: str = '/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset'
    batch_size: int = 512
    batch_size_for_accumulation: int = 16
    lr: float = 1e-5
    downstream_lr: float = 1e-3
    base_optimizer: str = 'sgd'
    weight_decay: float = 1e-4
    num_epochs: int = 5 # 25
    device: str = 'mps'
    validate_after_steps: int = 10
    log_deltas: bool = True
    num_workers: int = 4

    hessian_eigenspace_dim: int = 16
    moving_average_hessians_epochs: int = 5
    gamma: torch.float32 = 0.9

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_modules: Optional[list] = None

    ewc_lambda: float = 1e-4
    l2sp_lambda: float = 1e-4
    reset_lora: bool = True
    head_warmup_epochs: int = 1 # 10

    accumulate_mode: str = 'accumulate'  # or 'union'
    merge_tasks: str = 'union'  # or 'accumulate'
    iterations_of_accumulation: int = 4
    use_scaling: bool = True
    adapt_downstream_tasks: bool = False
    log_dir: str = f'logs/nostalgia_vision_experiment/{mode}/{lr}/{lora_r}/{hessian_eigenspace_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    use_tpu: bool = True
    world_size: int = 8



class ViTClassifier(nn.Module):
    def __init__(self):
        super(ViTClassifier, self).__init__()
        model_id = "google/vit-base-patch32-224-in21k"
        config = AutoConfig.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.vit = AutoModel.from_pretrained(model_id, config=config)

    def forward(self, pixel_values):
        # Extract the CLS token representation
        outputs = self.vit(pixel_values=pixel_values)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return cls_representation



class ContinualLearnerViT(nn.Module):

    def __init__(
        self, device,
        lr=1e-4, downstream_lr=1e-3, 
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        use_peft=True, lora_modules = None,
        optimizer_type="adamw"
    ):
        super().__init__()
        self.lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=lora_modules if lora_modules is not None else ["query", "value", "key"],
            lora_dropout=lora_dropout, bias="none"
        )
        self.backbone = ViTClassifier().to(device)
        self.rep_dim  = 768
        self.optimizer_type = optimizer_type

        self.use_peft = use_peft
        self.is_peft_on = True
        if self.use_peft:
            self._apply_peft()

        self.use_nostalgia = True
        self.use_preprocessor = False
        self.weight_decay = 0.0

        self.task_head_list = torch.nn.ModuleDict()
        self.active_task  = None

        self.device = device

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.learning_rate = lr
        self.downstream_learning_rate = downstream_lr

        self.nostalgia_Q = None
        self.nostalgia_scaling = None


    def preprocess_inputs(self, x):
        if self.use_preprocessor:
            x = self.backbone.processor(
                images=x,
                return_tensors="pt"
            )["pixel_values"].to(x.device)

        return x
        

    def _apply_peft(self):
        if self.use_peft:
            self.backbone = get_peft_model(self.backbone, self.lora_config) #type: ignore
            self.is_peft_on = True

    
    def _merge_and_unload_peft(self):
        if self.is_peft_on:
            self.backbone = self.backbone.merge_and_unload() #type: ignore

            if hasattr(self.backbone, "peft_config"):
                print("Deleting peft_config attribute...")
                delattr(self.backbone, "peft_config")

            self.is_peft_on = False


    def add_task(self, task_name, num_classes):
        self.task_head_list[task_name] = nn.Linear(self.rep_dim, num_classes, device=self.device)   # .to(self.device)
        nn.init.trunc_normal_(self.task_head_list[task_name].weight, std=0.02)  # type: ignore
        nn.init.zeros_(self.task_head_list[task_name].bias)                     # type: ignore

        for param in self.task_head_list[task_name].parameters():
            param.requires_grad = False


    def set_active_task(self, task_name):
        if self.active_task is not None:
            for param in self.task_head_list[self.active_task].parameters():
                param.requires_grad = False

        self.active_task = task_name
        for param in self.task_head_list[task_name].parameters():
            param.requires_grad = True

    
    def forward(self, x):
        # print("Input shape:", x.shape)
        features = self.backbone(x)
        if self.active_task is None:
            raise ValueError("Active task is not set.")
        logits = self.task_head_list[self.active_task](features)
        return logits

    
    def set_Q(self, Q: Optional[torch.Tensor], scaling: Optional[torch.Tensor] = None):
        self.nostalgia_Q = Q
        self.nostalgia_scaling = scaling
        

    def get_backbone_params(self):
        return [
            p for _, p in self.backbone.named_parameters()
            if p.requires_grad
        ]

    def get_backbone_params_dict(self):
        return {
            name: p
            for name, p in self.backbone.named_parameters()
            if p.requires_grad
        }

        
    def configure_optimizers(
            self,
            writter: Optional[SummaryWriter] = None,
            iteration: int = 0,
    ):
        # Backbone params (shared, projected)
        backbone_params = self.get_backbone_params()

        # ALL head params (we will freeze/unfreeze via requires_grad)
        head_params = []
        for head in self.task_head_list.values():
            head_params.extend(
                p for p in head.parameters() if p.requires_grad
            )

        if self.optimizer_type == "sgd":
            base_optimizer = torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                momentum=0.9,
                weight_decay=0.0,
                nesterov=True,
            )
        elif self.optimizer_type == "adamw":
            base_optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                weight_decay=0.0,
            )
        elif self.optimizer_type == "adam":
            base_optimizer = torch.optim.Adam(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


        # base_optimizer = torch.optim.SGD(
        #     [
        #         {"params": backbone_params},
        #         {"params": head_params},
        #     ],
        #     lr=self.learning_rate,
        #     momentum=0.9,
        #     weight_decay=1e-3,
        # )

        if not self.use_nostalgia:
            return base_optimizer

        assert backbone_params and len(backbone_params) > 0, "No backbone parameters to optimize for Nostalgia."

        nostalgia_opt = NostalgiaOptimizer(
            params=backbone_params,
            base_optimizer=base_optimizer,
            device=self.device,
            dtype=backbone_params[0].dtype,
            writter=writter,
            starting_step=iteration,
            weight_decay=self.weight_decay,
        )

        if self.nostalgia_Q is not None:
            nostalgia_opt.set_Q(self.nostalgia_Q, scaling=self.nostalgia_scaling)

        return nostalgia_opt

