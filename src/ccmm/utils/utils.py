import copy
import json
import logging
from pathlib import Path
from pydoc import locate
from typing import Any, Dict, List

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nn_core.serialization import load_model

ModelParams = Dict[str, torch.Tensor]


pylogger = logging.getLogger(__name__)

MODEL_SEED_TO_SYMBOL = {
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    5: "e",
    "dummy_a": "x",
    "dummy_b": "y",
    "dummy_c": "z",
}


def map_model_seed_to_symbol(seed):
    return MODEL_SEED_TO_SYMBOL[seed]


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


def block(i, j, n):
    return slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n)


def load_model_from_info(model_info_path, seed):
    model_info_path_seed = model_info_path + f"_{seed}.json"
    model_info = json.load(open(model_info_path_seed))
    model_class = locate(model_info["class"])

    model = load_model(model_class, checkpoint_path=Path(model_info["path"] + ".zip"))
    model.eval()

    return model


def save_permutations(permutations, path):
    for source, targets in permutations.items():
        for target, source_target_perms in targets.items():
            for perm_name, perm in source_target_perms.items():
                if perm is not None:
                    permutations[source][target][perm_name] = perm.tolist()
    with open(path, "w+") as f:
        json.dump(permutations, f)


def load_permutations(path):
    with open(path, "r") as f:
        permutations = json.load(f)

    for source, targets in permutations.items():
        for target, source_target_perms in targets.items():
            for perm_name, perm in source_target_perms.items():
                if perm is not None:
                    permutations[source][target][perm_name] = torch.tensor(perm)

    return permutations


class OnSaveCheckpointCallback(Callback):
    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        metadata = getattr(pl_module, "metadata", None)
        if metadata is not None:
            checkpoint["metadata"] = metadata
