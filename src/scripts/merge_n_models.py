import copy
import json
import logging
from typing import Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import NNCheckpointIO

from ccmm.matching.utils import restore_original_weights
from ccmm.utils.utils import load_model_from_info, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)

# ASSUMPTION: P[a][b] refers to the permutation/params to map b -> a


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds

    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    restore_original_weights(models, model_orig_weights)

    seed_index_everything(cfg)

    model_merger = instantiate(cfg.merger, permutation_spec=permutation_spec)

    merged_model = model_merger(models)

    pylogger.info(f"Successfully merged {len(model_seeds)} models.")

    save_merged_model(merged_model, cfg)


def save_merged_model(merged_model: LightningModule, cfg: DictConfig):
    trainer = pl.Trainer(
        plugins=[NNCheckpointIO(jailing_dir=cfg.output_path)],
    )

    trainer.strategy.connect(merged_model)
    ckpt_path = cfg.output_path / "merged_model.ckpt"

    trainer.save_checkpoint(ckpt_path)
    model_class = merged_model.__class__.__module__ + "." + merged_model.__class__.__qualname__

    model_info = {
        "path": str(ckpt_path),
        "class": model_class,
    }

    json.dump(model_info, open(cfg.out_model_info_path, "w+"))


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="merge_n_models")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()