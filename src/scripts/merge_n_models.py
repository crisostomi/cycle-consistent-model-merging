import copy
import logging
import os
from typing import Dict

import hydra
import omegaconf
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import NNCheckpointIO

from ccmm.matching.utils import restore_original_weights
from ccmm.utils.utils import load_model_from_artifact, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)

# ASSUMPTION: P[a][b] refers to the permutation/params to map b -> a


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds

    run = wandb.init(project=core_cfg.core.project_name, entity=core_cfg.core.entity, job_type="matching")

    artifact_path = (
        lambda seed: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.model.model_identifier}_{seed}:latest"
    )

    # {a: model_a, b: model_b, c: model_c, ..}
    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_artifact(run, artifact_path(seed)) for seed in cfg.model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    restore_original_weights(models, model_orig_weights)

    transform = instantiate(core_cfg.dataset.test.transform)

    train_dataset = instantiate(core_cfg.dataset.train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    seed_index_everything(cfg)

    model_merger = instantiate(cfg.merger, permutation_spec=permutation_spec)

    merged_model, repaired_model = model_merger(models, train_loader=train_loader)

    pylogger.info(f"Successfully merged {len(model_seeds)} models.")

    upload_model_to_wandb(merged_model, run, core_cfg)
    upload_model_to_wandb(repaired_model, run, core_cfg, suffix="_repaired")


def upload_model_to_wandb(merged_model: LightningModule, run, cfg: DictConfig, suffix=""):
    trainer = pl.Trainer(
        plugins=[NNCheckpointIO(jailing_dir="./tmp")],
    )

    # Create a temporary file name
    temp_path = "temp_checkpoint.ckpt"

    # Save checkpoint
    trainer.strategy.connect(merged_model)
    trainer.save_checkpoint(temp_path)

    model_class = merged_model.__class__.__module__ + "." + merged_model.__class__.__qualname__

    num_models = len(cfg.matching.model_seeds)

    artifact_name = f"{cfg.model.model_identifier}_{cfg.matching.merger.name}_N{num_models}{suffix}"

    model_artifact = wandb.Artifact(
        name=artifact_name,
        type="merged model",
        metadata={"model_identifier": cfg.model.model_identifier, "model_class": model_class},
    )

    model_artifact.add_file(temp_path + ".zip", name="merged.ckpt.zip")
    run.log_artifact(model_artifact)

    os.remove(temp_path + ".zip")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="merge_n_models", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
