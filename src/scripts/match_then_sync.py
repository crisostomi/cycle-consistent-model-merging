import copy
import itertools
import logging
from typing import Dict, Set

import hydra
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

from ccmm.matching.weight_matching import construct_uber_matrix, optimize_synchronization, weight_matching
from ccmm.utils.matching_utils import (
    PermutationIndices,
    check_permutations_are_valid,
    get_all_symbols_combinations,
    get_inverse_permutations,
    parse_sync_matrix,
    perm_indices_to_perm_matrix,
    perm_matrix_to_perm_indices,
    restore_original_weights,
)
from ccmm.utils.utils import load_model_from_info, map_model_seed_to_symbol, save_permutations

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed = {map_model_seed_to_symbol(seed): seed for seed in model_seeds}

    # {a, b, c, ..}
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    # dicts for permutations and permuted params, D[a][b] refers to the permutation/params to map a -> b
    permutations = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    assert set(permutation_spec.axes_to_perm.keys()) == set(models["a"].model.state_dict().keys())

    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    canonical_combinations = [(source, target) for (source, target) in all_combinations if source < target]

    for source, target in canonical_combinations:
        permutations[source][target] = weight_matching(
            permutation_spec,
            target=models[target].model.state_dict(),
            to_permute=models[source].model.state_dict(),
        )

        permutations[target][source] = get_inverse_permutations(permutations[source][target])

        restore_original_weights(models, model_orig_weights)

        check_permutations_are_valid(permutations[source][target], permutations[target][source])

    sync_permutations = synchronize_permutations(permutations, method=cfg.sync_method, symbols=symbols)

    save_permutations(permutations, cfg.permutations_path / "permutations.json")
    save_permutations(sync_permutations, cfg.permutations_path / "sync_permutations.json")


def synchronize_permutations(permutations: Dict[str, Dict[str, PermutationIndices]], method: str, symbols: Set[str]):
    """
    :param permutations: permutations[a][b] refers to the permutation to map a -> b
    :param method: stiefel, spectral or nmfSync
    :param symbols: set of model identifiers, e.g. {a, b, c, ..}
    """

    pylogger.info(f"Synchronizing permutations with method {method}")

    ref_symbol_a, ref_symbol_b = list(symbols)[0], list(symbols)[1]
    ref_perms = permutations[ref_symbol_a][ref_symbol_b]
    perm_names = list(ref_perms.keys())

    sync_permutations = {
        symb: {other_symb: {perm_name: None for perm_name in perm_names} for other_symb in symbols.difference(symb)}
        for symb in symbols
    }

    # (a, b), (a, c), (b, c), ...
    combinations = list(itertools.permutations(permutations.keys(), 2))

    # layer for layer synchronization
    for layer_perm_name, layer_perm in ref_perms.items():
        perm_dim = len(layer_perm)

        perm_matrices = {
            (source, target): perm_indices_to_perm_matrix(permutations[source][target][layer_perm_name])
            for (source, target) in combinations
        }

        uber_matrix = construct_uber_matrix(perm_matrices, perm_dim, combinations, symbols)

        sync_matrix = optimize_synchronization(uber_matrix, perm_dim, method)

        sync_perm_matrices = parse_sync_matrix(sync_matrix, perm_dim, symbols, combinations)

        for comb in combinations:
            sync_perm_matrix_comb = sync_perm_matrices[comb]
            sync_permutations[comb[0]][comb[1]][layer_perm_name] = perm_matrix_to_perm_indices(sync_perm_matrix_comb)

            # safety check
            inv_comb = (comb[1], comb[0])
            assert torch.all(sync_perm_matrices[comb] == sync_perm_matrices[inv_comb].T)

    return sync_permutations


@hydra.main(config_path=str(PROJECT_ROOT / "conf/matching"), config_name="match_then_sync")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
