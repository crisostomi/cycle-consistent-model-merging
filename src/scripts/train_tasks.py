import logging
import os
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
import wandb
from datasets import disable_caching
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

import ccmm  # noqa
from ccmm.utils.utils import build_callbacks

disable_caching()
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

    num_tasks = datamodule.data["metadata"]["num_tasks"]
    num_classes = datamodule.data["metadata"]["num_classes"]

    for task_ind in range(num_tasks + 1):
        seed_index_everything(cfg.train)

        pylogger.info(f"Instantiating <{cfg.nn.module.model}>")
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.module,
            _recursive_=False,
            num_classes=num_classes,
            model=cfg.nn.module.model,
            input_dim=datamodule.img_size,
            metadata=datamodule.metadata,
        )

        datamodule.task_ind = task_ind
        datamodule.transform_func = hydra.utils.instantiate(cfg.dataset.transform_func, _recursive_=True)
        datamodule.setup()

        template_core: NNTemplateCore = NNTemplateCore(
            restore_cfg=cfg.train.get("restore", None),
        )
        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        pylogger.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=template_core.trainer_ckpt_path,
        )

        # best_model_path = get_checkpoint_callback(callbacks).best_model_path
        artifact_name = f"{cfg.nn.module.model_name}_T{task_ind}_{cfg.train.seed_index}"

        upload_model_to_wandb(model, logger.experiment, cfg, artifact_name)

        if trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(model, datamodule.test_dataloader()[0])

        if logger is not None:
            logger.experiment.finish()

    return logger.run_dir


def upload_model_to_wandb(model: LightningModule, run, cfg: DictConfig, artifact_name):
    trainer = pl.Trainer(
        plugins=[NNCheckpointIO(jailing_dir="./tmp")],
    )

    temp_path = "temp_checkpoint.ckpt"

    trainer.strategy.connect(model)
    trainer.save_checkpoint(temp_path)

    model_class = model.__class__.__module__ + "." + model.__class__.__qualname__

    model_artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        metadata={"model_identifier": cfg.nn.module.model_name, "model_class": model_class},
    )

    model_artifact.add_file(temp_path + ".zip", name="trained.ckpt.zip")
    run.log_artifact(model_artifact)

    os.remove(temp_path + ".zip")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="train_tasks", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
