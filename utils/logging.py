import wandb
import torch_xla.core.xla_model as xm

class WandbLogger:
    def __init__(self, project="my-project", config=None):
        self.is_master = xm.is_master_ordinal()

        if self.is_master:
            wandb.init(
                project=project,
                config=config,
                reinit=True
            )

    def log(self, step, metrics: dict):
        if self.is_master:
            wandb.log(metrics, step=step)

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        if not self.is_master:
            return

        formatted = {
            f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()
        }

        wandb.log(formatted, step=step)

    def close(self):
        if self.is_master:
            wandb.finish()
