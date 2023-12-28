import copy
import logging
from pathlib import Path
from typing import Dict

import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

from ccmm.matching.weight_matching import synchronized_weight_matching, weight_matching
from ccmm.utils.matching_utils import (
    check_permutations_are_valid,
    get_all_symbols_combinations,
    get_inverse_permutations,
    restore_original_weights,
)
from ccmm.utils.utils import load_model_from_info, map_model_seed_to_symbol, save_permutations

pylogger = logging.getLogger(__name__)

# ASSUMPTION: P[a][b] refers to the permutation/params to map b -> a


def run(cfg: DictConfig) -> str:

    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    if not cfg.sync_method:
        pylogger.warning("Only using naive and git re-basin methods.")

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds
    cfg.results_path = Path(cfg.results_path) / f"{len(model_seeds)}"

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed: Dict[int, str] = {map_model_seed_to_symbol(seed): seed for seed in model_seeds}

    # {a, b, c, ..}
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ... sorted first by first element, then by second element
    all_combinations = get_all_symbols_combinations(symbols)

    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    # dicts for permutations and permuted params, D[a][b] refers to the permutation/params to map b -> a
    permutations = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    # assert set(permutation_spec.axes_to_perm.keys()) == set(models["a"].model.state_dict().keys())

    restore_original_weights(models, model_orig_weights)

    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    canonical_combinations = [(fixed, permutee) for (fixed, permutee) in all_combinations if fixed < permutee]

    for fixed, permutee in canonical_combinations:
        permutations[fixed][permutee] = weight_matching(
            permutation_spec,
            fixed=model_orig_weights[fixed],
            to_permute=model_orig_weights[permutee],
            alternate_diffusion_params=None,
        )

        permutations[permutee][fixed] = get_inverse_permutations(permutations[fixed][permutee])

        restore_original_weights(models, model_orig_weights)

        check_permutations_are_valid(permutations[fixed][permutee], permutations[permutee][fixed])

    symbols_seq = sorted(list(symbols))

    if cfg.sync_method is not None:
        seed_index_everything(cfg)

        pylogger.info(f"Using synchronization method {cfg.sync_method}")
        improved_permutations = synchronized_weight_matching(
            models, permutation_spec, method=cfg.sync_method, symbols=symbols_seq, combinations=all_combinations
        )
    elif cfg.use_alternate_diffusion:
        seed_index_everything(cfg)

        pylogger.info("Using alternating diffusion")
        improved_permutations = {
            symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols
        }

        for fixed, permutee in canonical_combinations:
            improved_permutations[fixed][permutee] = weight_matching(
                permutation_spec,
                fixed=model_orig_weights[fixed],
                to_permute=model_orig_weights[permutee],
                alternate_diffusion_params=cfg.alternate_diffusion_params,
            )

            improved_permutations[permutee][fixed] = get_inverse_permutations(improved_permutations[fixed][permutee])

            restore_original_weights(models, model_orig_weights)

            check_permutations_are_valid(improved_permutations[fixed][permutee], improved_permutations[permutee][fixed])
    else:
        pylogger.info("Not using any improved method")
        improved_permutations = copy.deepcopy(permutations)

    save_permutations(permutations, cfg.permutations_path / "permutations.json")
    save_permutations(improved_permutations, cfg.permutations_path / "improved_permutations.json")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
