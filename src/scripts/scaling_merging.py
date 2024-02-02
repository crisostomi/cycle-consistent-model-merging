import logging
from time import sleep

import hydra
import omegaconf
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from scripts.evaluate_merged_model import run as evaluate_merged_model
from scripts.merge_n_models import run as merge_n_models

pylogger = logging.getLogger(__name__)


def scaling_exp(cfg: DictConfig) -> str:

    seed_index_everything(cfg)

    next_seed = 9
    model_seeds = list(range(1, next_seed))

    for seed in range(next_seed, cfg.total_num_models + 1):

        model_seeds.append(seed)
        pylogger.info(f"Running with seeds {model_seeds}")
        cfg.matching.model_seeds = model_seeds

        merge_n_models(cfg)

        sleep(30)

        evaluate_merged_model(cfg)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="merge_n_models_scaling", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    scaling_exp(cfg)


if __name__ == "__main__":
    main()
