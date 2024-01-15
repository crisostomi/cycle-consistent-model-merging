import logging

import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from ccmm.utils.utils import load_model_from_info

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg
    cfg = cfg.matching

    seed_index_everything(cfg)

    merged_model = load_model_from_info(cfg.out_model_info_path)

    pylogger.info("Loaded merged model.")

    template_core: NNTemplateCore = NNTemplateCore(restore_cfg=None)

    logger: NNLogger = NNLogger(logging_cfg=core_cfg.logging, cfg=core_cfg, resume_id=template_core.resume_id)

    transform = instantiate(core_cfg.dataset.test.transform)

    train_dataset = instantiate(core_cfg.dataset.train, transform=transform)
    test_dataset = instantiate(core_cfg.dataset.test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    results = {}
    trainer = instantiate(cfg.trainer)

    train_results = trainer.test(merged_model, train_loader)[0]
    train_results["acc/train"] = train_results["acc/test"]
    train_results["loss/train"] = train_results["loss/test"]

    test_results = trainer.test(merged_model, test_loader)[0]

    results = {"train": train_results, "test": test_results}

    metrics = ["acc", "loss"]
    for metric in metrics:
        for split in ["train", "test"]:
            logger.experiment.log({f"{metric}/{split}": results[split][f"{metric}/{split}"]})

    if logger is not None:
        logger.log_configuration(model=merged_model, cfg=core_cfg)
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="merge_n_models", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
