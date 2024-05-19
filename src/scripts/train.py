import logging
import os
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

import ccmm  # noqa
from ccmm.data.datamodule import MetaData
from ccmm.utils.utils import build_callbacks

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    pylogger.info(f"Using dataset [bold yellow]{cfg.dataset.name}[/bold yellow]")

    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False, metadata=metadata)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

    upload_model_to_wandb(model, logger.experiment, cfg)

    if "test" in cfg.nn.data.dataset and trainer.checkpoint_callback.best_model_path is not None:
        pylogger.info("Starting testing!")
        trainer.test(datamodule=datamodule)

    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


def upload_model_to_wandb(model: LightningModule, run, cfg: DictConfig):
    trainer = pl.Trainer(
        plugins=[NNCheckpointIO(jailing_dir="./tmp")],
    )

    temp_path = "temp_checkpoint.ckpt"

    trainer.strategy.connect(model)
    trainer.save_checkpoint(temp_path)

    model_class = model.__class__.__module__ + "." + model.__class__.__qualname__

    artifact_name = f"{cfg.dataset.name}_{cfg.nn.module.model_name}_{cfg.train.seed_index}"

    model_artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        metadata={"model_identifier": cfg.nn.module.model_name, "model_class": model_class},
    )

    model_artifact.add_file(temp_path + ".zip", name="trained.ckpt.zip")
    run.log_artifact(model_artifact)

    os.remove(temp_path + ".zip")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mlp", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
