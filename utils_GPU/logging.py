"""
utils_GPU/logging.py

GPU equivalent of utils/logging.py.

Changes from the TPU version:
  - xm.is_master_ordinal() → dist.get_rank() == 0 (or rank == 0 in single-GPU).
  - Interface is identical so VisionExperiment_GPU can use it transparently.
"""

import wandb
import torch.distributed as dist


def _is_master() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True   # single-GPU / non-distributed run: always master


class WandbLogger:
    def __init__(self, project: str = "my-project", config=None):
        self.is_master = _is_master()

        if self.is_master:
            wandb.init(
                project=project,
                config=config,
                reinit=True,
            )

    def log(self, step: int, metrics: dict):
        if self.is_master:
            wandb.log(metrics, step=step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if not self.is_master:
            return
        formatted = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        wandb.log(formatted, step=step)

    def close(self):
        if self.is_master:
            wandb.finish()
