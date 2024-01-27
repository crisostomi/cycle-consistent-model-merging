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
from ccmm.matching.utils import get_inverse_permutations, plot_permutation_history_animation
from ccmm.utils.utils import flatten_params, load_model_from_artifact, map_model_seed_to_symbol, save_permutations

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    run = wandb.init(project=core_cfg.core.project_name, entity=core_cfg.core.entity, job_type="matching")

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed: Dict[int, str] = {map_model_seed_to_symbol(seed): seed for seed in cfg.model_seeds}

    # models: Dict[str, LightningModule] = {
    #     map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in cfg.model_seeds
    # }

    artifact_path = (
        lambda seed: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.model.model_identifier}_{seed}:v0"
    )

    # {a: model_a, b: model_b, c: model_c, ..}
    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_artifact(run, artifact_path(seed)) for seed in cfg.model_seeds
    }

    ref_model = models["a"]

    # data structure that specifies the permutations acting on each layer and on what axis
    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    ref_model = list(models.values())[0]
    assert set(permutation_spec.layer_and_axes_to_perm.keys()) == set(ref_model.model.state_dict().keys())

    # always permute the model having larger character order, i.e. c -> b, b -> a and so on ...
    symbols = set(symbols_to_seed.keys())
    sorted_symbols = sorted(symbols, reverse=False)
    fixed_symbol, permutee_symbol = sorted_symbols
    fixed_model, permutee_model = models[fixed_symbol], models[permutee_symbol]

    # dicts for permutations and permuted params, D[a][b] refers to the permutation/params to map b -> a
    permutations = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    matcher = instantiate(cfg.matcher, permutation_spec=permutation_spec)
    permutations[fixed_symbol][permutee_symbol], perm_history = matcher(
        fixed=flatten_params(fixed_model.model), permutee=flatten_params(permutee_model.model)
    )

    permutations[permutee_symbol][fixed_symbol] = get_inverse_permutations(permutations[fixed_symbol][permutee_symbol])

    # save models as well
    for symbol, model in models.items():
        torch.save(model.model.state_dict(), cfg.permutations_path / f"model_{symbol}.pt")

    save_permutations(permutations, cfg.permutations_path / "permutations.json")

    if cfg.plot_perm_history:
        plot_permutation_history_animation(perm_history, cfg)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
