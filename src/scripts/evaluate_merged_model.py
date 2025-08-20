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

from ccmm.utils.utils import load_model_from_artifact

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg
    cfg = cfg.matching

    seed_index_everything(cfg)

    num_models = len(cfg.model_seeds)

    artifact_path = (
        lambda suffix: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.model.model_identifier}_{core_cfg.matching.merger.name}_N{num_models}{suffix}:latest"
    )

    suffix = "_repaired" if cfg.repaired else ""
    tag = "repaired" if cfg.repaired else "merged"
    core_cfg.core.tags.append(tag)
    core_cfg.core.tags.append(core_cfg.matching.merger.name)

    pylogger.info("Loaded merged model.")

    template_core: NNTemplateCore = NNTemplateCore(restore_cfg=None)

    logger: NNLogger = NNLogger(
        logging_cfg=core_cfg.logging, cfg=core_cfg, resume_id=template_core.resume_id
    )

    merged_model = load_model_from_artifact(logger.experiment, artifact_path(suffix))

    transform = instantiate(core_cfg.dataset.test.transform)

    train_dataset = instantiate(core_cfg.dataset.train, transform=transform)
    test_dataset = instantiate(core_cfg.dataset.test, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    results = {}
    trainer = instantiate(cfg.train)

    train_results = trainer.test(merged_model, train_loader)[0]
    train_results["acc/train"] = train_results["acc/test"]
    train_results["loss/train"] = train_results["loss/test"]

    test_results = trainer.test(merged_model, test_loader)[0]
    results = {"train": train_results, "test": test_results}

    metrics = ["acc", "loss"]
    for metric in metrics:
        for split in ["train", "test"]:
            logger.experiment.log(
                {f"{metric}/{split}": results[split][f"{metric}/{split}"]}
            )

    # trainer = instantiate(cfg.train, max_epochs=100)
    # trainer.fit(merged_model, train_loader)
    # test_results = trainer.test(merged_model, test_loader)[0]
    # print(test_results)

    if logger is not None:
        logger.log_configuration(model=merged_model, cfg=core_cfg)
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="merge_n_models",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
