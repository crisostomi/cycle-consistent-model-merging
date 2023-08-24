import json
import logging
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import NNCheckpointIO

from ccmm.matching.weight_matching import create_artificially_permuted_models
from ccmm.pl_modules.pl_module import MyLightningModule
from ccmm.utils.utils import OnSaveCheckpointCallback, load_model_from_info, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds
    cfg.results_path = Path(cfg.results_path / f"{len(model_seeds)}")

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    pylogger.info("Using artificial models obtained via permutations of the first model")
    seed_model = load_model_from_info(cfg.model_info_path, model_seeds[0])
    artificial_models: List[MyLightningModule] = create_artificially_permuted_models(
        seed_model, permutation_spec, num_models=len(model_seeds) - 1
    )

    models = {map_model_seed_to_symbol(seed): model for seed, model in zip(model_seeds[1:], artificial_models)}

    trainer = pl.Trainer(plugins=[NNCheckpointIO()], callbacks=[OnSaveCheckpointCallback()])
    for model_id, model in models.items():
        trainer.strategy.connect(model)

        model_path = str(cfg.dummy_permutations_path) + f"{cfg.model_identifier}_{model_id}"
        ckpt_path = f"{model_path}.ckpt"
        trainer.save_checkpoint(ckpt_path)

        model_info = {
            "path": ckpt_path,
            "class": str(model.__class__.__module__ + "." + model.__class__.__qualname__),
        }

        model_info_path = Path(f"{model_path}.json")
        model_info_path.parent.mkdir(parents=True, exist_ok=True)

        json.dump(model_info, open(model_info_path, "w+"))


@hydra.main(config_path=str(PROJECT_ROOT / "conf/matching"), config_name="match_then_sync")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
