import copy
import itertools
from typing import Dict, List, Set, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from ccmm.utils.utils import block

# shape (n, n), contains the permutation matrix, e.g. [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
PermutationMatrix = Tensor

# shape (n), contains the indices of the target permutation, e.g. [0, 3, 2, 1]
PermutationIndices = Tensor


def restore_original_weights(models: Dict[str, LightningModule], original_weights: Dict[str, Dict[str, Tensor]]):
    """
    Restores the original weights of the models, i.e. the weights before the matching procedure.
    Both models and original_weights are dictionaries indexed by a model_id (e.g. "a", "b", "c", ..).

    :param models: The models to restore the weights of.
    :param original_weights: The original weights of the models.
    """

    for model_id, model in models.items():
        model.model.load_state_dict(copy.deepcopy(original_weights[model_id]))


def get_all_symbols_combinations(symbols: Set[str]) -> List[Tuple[str, str]]:
    """
    Given a set of symbols, returns all possible permutations of two symbols.

    :param symbols: The set of symbols, e.g. {"a", "b", "c"}.
    :return: A list of all possible permutations of two symbols, e.g. [("a", "b"), ("a", "c"), ("b", "a"), ("b", "c"), ("c", "a"), ("c", "b")].
    """
    combinations = list(itertools.permutations(symbols, 2))
    sorted_combinations = sorted(combinations)
    return sorted_combinations


def get_inverse_permutations(permutations: Dict[str, PermutationIndices]) -> Dict[str, PermutationIndices]:
    """
    Given a dictionary of permutations, returns a dictionary of the inverse permutations.
    """

    inv_permutations = {}

    for perm_name, perm in permutations.items():
        perm_matrix = perm_indices_to_perm_matrix(perm)
        inv_perm_matrix = perm_matrix.T
        inv_permutations[perm_name] = perm_matrix_to_perm_indices(inv_perm_matrix)

    return inv_permutations


def perm_indices_to_perm_matrix(perm_indices: PermutationIndices):
    n = len(perm_indices)
    perm_matrix = torch.eye(n)[perm_indices.long()]
    return perm_matrix


def perm_matrix_to_perm_indices(perm_matrix: PermutationMatrix):
    return perm_matrix.nonzero()[:, 1]


def check_permutations_are_valid(permutation, inv_permutation):
    for layer_perm, layer_perm_inv in zip(permutation.values(), inv_permutation.values()):
        perm_matrix = perm_indices_to_perm_matrix(layer_perm)
        inv_perm_matrix = perm_indices_to_perm_matrix(layer_perm_inv).T

        assert is_valid_permutation_matrix(perm_matrix) and is_valid_permutation_matrix(inv_perm_matrix)
        assert torch.all(perm_matrix == inv_perm_matrix)


def is_valid_permutation_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False

    row_sums = torch.sum(matrix, dim=1)
    col_sums = torch.sum(matrix, dim=0)

    ones_tensor = torch.ones_like(row_sums)

    return (
        torch.all(row_sums == ones_tensor)
        and torch.all(col_sums == ones_tensor)
        and torch.all((matrix == 0) | (matrix == 1))
    )


# TODO: check if this still makes sense to keep, not used at the moment
def construct_gt_uber_matrix(
    perm_matrices: Dict[Tuple[str, str], PermutationMatrix],
    perm_dim: int,
    combinations: List[Tuple[str, str]],
    num_models: int,
) -> torch.Tensor:
    """
    :param perm_matrices: dictionary of permutation matrices, e.g. {(a, b): Pab, (a, c): Pac, (b, c): Pbc, ...}
    :param perm_dim: dimension of the permutation matrices
    :param combinations: list of combinations, e.g. [(a, b), (a, c), (b, c), ...]
    """
    perms_to_consider = []

    sorted_combinations = sorted(combinations, key=lambda x: (x[0], x[1]))
    target_source = "a"
    for source, target in sorted_combinations:
        if target == target_source:
            perms_to_consider.append(perm_matrices[(source, target)])

    Paa = torch.eye(perm_dim)

    # (Paa, Pab, Pac, ...)
    U_gt = torch.cat([Paa, *perms_to_consider], axis=0)

    gt_uber_matrix = U_gt @ U_gt.T

    assert gt_uber_matrix.shape == (perm_dim * num_models, perm_dim * num_models)

    return gt_uber_matrix


def three_models_uber_matrix(PAB, PAC, PBC, perm_dim):
    # P_gt2 = [speye(n), P_AB, P_AC;
    #           P_BA, speye(n), P_BC;
    #           P_CA, P_CB, speye(n)];

    uber_matrix = torch.zeros((3 * perm_dim, 3 * perm_dim))
    uber_matrix[block(0, 1, perm_dim)] = PAB
    uber_matrix[block(1, 0, perm_dim)] = PAB.T

    uber_matrix[block(0, 2, perm_dim)] = PAC
    uber_matrix[block(2, 0, perm_dim)] = PAC.T

    uber_matrix[block(1, 2, perm_dim)] = PBC
    uber_matrix[block(2, 1, perm_dim)] = PBC.T

    uber_matrix[block(0, 0, perm_dim)] = torch.eye(perm_dim)
    uber_matrix[block(1, 1, perm_dim)] = torch.eye(perm_dim)
    uber_matrix[block(2, 2, perm_dim)] = torch.eye(perm_dim)

    return uber_matrix


def construct_uber_matrix(
    perm_matrices: Dict[Tuple[str, str], PermutationMatrix], perm_dim: int, combinations: List, symbols: Set[str]
) -> torch.Tensor:
    """
    :param perm_matrices: dictionary of permutation matrices, e.g. {(a, b): Pab, (a, c): Pac, (b, c): Pbc, ...}
    :param perm_dim: dimension of the permutation matrices
    :param combinations: list of combinations, e.g. [(a, b), (a, c), (b, c), ...]
    """

    num_models = len(symbols)

    uber_matrix = torch.zeros((num_models * perm_dim, num_models * perm_dim))

    order = {symb: i for i, symb in enumerate(sorted(symbols))}

    # fill in diagonal blocks with identity matrices
    for i in range(num_models):
        uber_matrix[block(i, i, perm_dim)] = torch.eye(perm_dim)

    # fill in off-diagonal blocks with permutation matrices

    for source, target in combinations:
        P = perm_matrices[(source, target)]
        i, j = order[source], order[target]
        uber_matrix[block(i, j, perm_dim)] = P
        uber_matrix[block(j, i, perm_dim)] = P.T

    return uber_matrix


def parse_sync_matrix(sync_matrix, n, symbols, combinations):
    """
    Parse the large synchronization block matrix into the individual permutation matrices.
    """
    num_models = len(symbols)
    assert sync_matrix.shape == (num_models * n, n)

    ordered_symbols = sorted(symbols)

    to_universe, from_universe = {}, {}
    for order, symbol in enumerate(ordered_symbols):
        to_universe[symbol] = sync_matrix[n * order : n * (order + 1), :]
        from_universe[symbol] = to_universe[symbol].T

    sync_perm_matrices = {comb: None for comb in combinations}
    for source, target in combinations:
        # source B, target A: P_AB = P_UA.T @ P_UB
        new_perm_matrix_comb = from_universe[target] @ to_universe[source]

        assert is_valid_permutation_matrix(new_perm_matrix_comb)

        sync_perm_matrices[(source, target)] = new_perm_matrix_comb

    return sync_perm_matrices


def parse_three_models_sync_matrix(sync_matrix, n, symbols, combinations):
    """
    Parse the large synchronization block matrix into the individual permutation matrices.
    """
    num_models = len(symbols)
    assert sync_matrix.shape == (num_models * n, num_models * n)

    sync_perm_matrices = {comb: None for comb in combinations}

    a = symbols[0]
    b = symbols[1]
    c = symbols[2]

    sync_perm_matrices[(a, b)] = sync_matrix[block(0, 1, n)]
    sync_perm_matrices[(b, a)] = sync_matrix[block(1, 0, n)]

    sync_perm_matrices[(a, c)] = sync_matrix[block(0, 2, n)]
    sync_perm_matrices[(c, a)] = sync_matrix[block(2, 0, n)]

    sync_perm_matrices[(b, c)] = sync_matrix[block(1, 2, n)]
    sync_perm_matrices[(c, b)] = sync_matrix[block(2, 1, n)]

    P_BC = sync_perm_matrices[(b, c)]
    P_CA = sync_perm_matrices[(c, a)]
    P_AB = sync_perm_matrices[(a, b)]

    assert torch.all(P_BC @ P_CA @ P_AB == torch.eye(n))

    return sync_perm_matrices
