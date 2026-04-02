from .VisionExperiment import NostalgiaExperiment
import torch_xla.distributed.xla_multiprocessing as xmp
from models.model import NostalgiaConfig

def _mp_fn(rank):
    config = NostalgiaConfig()
    experiment = NostalgiaExperiment(config)
    experiment.train(rank)

if __name__ == '__main__':
    xmp.spawn(_mp_fn, start_method='fork')
    # _mp_fn(0)
