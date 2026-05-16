import torch

import torch_xla.core.xla_model as xm
import torchvision.transforms as transforms
from utils.accumulate import *
from utils.hessians import *
from utils.nostalgia import *
from utils.TPU import *

from models.model import ContinualLearnerViT, NostalgiaConfig




def check_orthogonality(Q):
    qtq = Q.T @ Q
    eye = torch.eye(qtq.shape[0], device=qtq.device, dtype=qtq.dtype)
    orth_err = (qtq - eye).abs().max()
    return orth_err.item()



class NostalgiaExperiment:
    def __init__(self, config:NostalgiaConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)

        self.augment = transforms.Compose([ 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.config.world_size = xr.world_size()
        if self.config.use_tpu:
            self.device = xm.xla_device()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ContinualLearnerViT( 
            lr = self.config.lr, 
            downstream_lr= self.config.downstream_lr, 
            lora_r = self.config.lora_r, 
            lora_alpha = self.config.lora_alpha,
            lora_dropout = self.config.lora_dropout,
            use_peft=True,
            lora_modules=self.config.lora_modules,
            device=self.device,
            optimizer_type=self.config.optimizer_type,
        ).to(self.device)







