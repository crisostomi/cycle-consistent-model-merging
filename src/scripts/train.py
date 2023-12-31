import json
import logging
from pathlib import Path
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

import ccmm  # noqa
from ccmm.data.datamodule import MetaData
from ccmm.utils.utils import build_callbacks, get_checkpoint_callback

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

    best_model_path = get_checkpoint_callback(callbacks).best_model_path

    best_model_info = {
        "path": best_model_path,
        "class": str(model.__class__.__module__ + "." + model.__class__.__qualname__),
    }

    model_identifier = f"{cfg.nn.module.model_name}_{cfg.train.seed_index}"
    model_info_path = Path(cfg.nn.output_path) / cfg.dataset.name / f"{model_identifier}.json"
    model_info_path.parent.mkdir(parents=True, exist_ok=True)

    json.dump(best_model_info, open(model_info_path, "w+"))

    if "test" in cfg.nn.data.dataset and trainer.checkpoint_callback.best_model_path is not None:
        pylogger.info("Starting testing!")
        trainer.test(datamodule=datamodule)

    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mlp", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
