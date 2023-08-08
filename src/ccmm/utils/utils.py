import copy
import logging
from typing import List

import hydra
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

pylogger = logging.getLogger(__name__)


def flatten_params(model):
    return model.state_dict()


def linear_interpolation(lam, t1, t2):
    t3 = copy.deepcopy(t2)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


# class ChangeScheduler(Callback):

#     def on_train_epoch_start(self, trainer. pl_module, scheduler):
#         if trainer.current_epoch = 1:
#             trainer.lr_schedulers = trainer.configure_schedulers([new_schedulers])
#             trainer.optimizer_frequencies = []
