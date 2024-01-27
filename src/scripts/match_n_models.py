import logging
from typing import Dict

import hydra
import omegaconf
import torch  # noqa
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

import ccmm  # noqa
from ccmm.matching.utils import get_all_symbols_combinations, plot_permutation_history_animation
from ccmm.utils.utils import load_model_from_artifact, map_model_seed_to_symbol, save_factored_permutations

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed: Dict[int, str] = {map_model_seed_to_symbol(seed): seed for seed in cfg.model_seeds}

    run = wandb.init(project=core_cfg.core.project_name, entity=core_cfg.core.entity, job_type="matching")

    artifact_path = (
        lambda seed: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.model.model_identifier}_{seed}:v0"
    )

    # {a: model_a, b: model_b, c: model_c, ..}
    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_artifact(run, artifact_path(seed)) for seed in cfg.model_seeds
    }

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    ref_model = list(models.values())[0]
    assert set(permutation_spec.layer_and_axes_to_perm.keys()) == set(ref_model.model.state_dict().keys())

    # always permute the model having larger character order, i.e. c -> b, b -> a and so on ...
    symbols = set(symbols_to_seed.keys())
    sorted_symbols = sorted(symbols, reverse=False)

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)
    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    canonical_combinations = [(source, target) for (source, target) in all_combinations if source < target]

    matcher = instantiate(cfg.matcher, permutation_spec=permutation_spec)

    permutations, perm_history = matcher(models, symbols=sorted_symbols, combinations=canonical_combinations)

    save_factored_permutations(permutations, cfg.permutations_path / "permutations.json")

    if cfg.plot_perm_history:
        plot_permutation_history_animation(perm_history, cfg)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching_n_models", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
