import copy
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy
import torch
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.utils import (
    PermutationMatrix,
    get_permuted_param,
    is_valid_permutation_matrix,
    perm_matrix_to_perm_indices,
)
from ccmm.matching.weight_matching import compute_weights_similarity
from ccmm.utils.utils import block

pylogger = logging.getLogger(__name__)


def synchronized_weight_matching(models, ps: PermutationSpec, method, symbols, combinations, max_iter=100):
    """
    Find a permutation to make the params in `models` match.

    :param ps: PermutationSpec
    :param target: the parameters to match
    :param to_permute: the parameters to permute
    """
    params = {s: m.model.state_dict() for s, m in models.items()}
    a, b, c = list(symbols)[0], list(symbols)[1], list(symbols)[2]
    pylogger.info(f"a: {a}, b: {b}, c: {c}")

    # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
    perm_sizes = {p: params[a][axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    # {'a': {'b': 'P0': P_AB_0, 'c': ....}, .... }. N.B. it's unsorted. P[a][b] refers to the permutation to map b -> a
    # i.e. P[fixed][permutee] maps permutee to fixed target
    symbol_set = set(symbols)
    perm_indices = {
        symb: {
            other_symb: {p: torch.arange(n) for p, n in perm_sizes.items()}
            for other_symb in symbol_set.difference(symb)
        }
        for symb in symbol_set
    }

    # e.g. P0, P1, ..
    perm_names = list(perm_indices[a][b].keys())

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        progress = False

        # iterate over the permutation matrices in random order
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]

            # a, b in similarities[a][b] are such that the permutation maps B to A
            similarities = {
                symb: {other_symb: torch.zeros((n, n)) for other_symb in symbol_set.difference(symb)}
                for symb in symbol_set
            }

            # all the params that are permuted by this permutation matrix, together with the axis on which it acts
            # e.g. ('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..
            params_and_axes = ps.perm_to_axes[p]

            for params_name, axis in params_and_axes:
                w_a = copy.deepcopy(params[a][params_name])
                w_b = copy.deepcopy(params[b][params_name])
                w_c = copy.deepcopy(params[c][params_name])

                assert w_a.shape == w_b.shape and w_b.shape == w_c.shape

                perms_to_apply = ps.axes_to_perm[params_name]

                # ASSUMPTION: given A, B, we always permute B to match A

                # w_b_a are the weights of A permuted to match B
                w_a_b = get_permuted_param(w_b, perms_to_apply, perm_indices[a][b], except_axis=axis)
                w_a_c = get_permuted_param(w_c, perms_to_apply, perm_indices[a][c], except_axis=axis)
                w_b_c = get_permuted_param(w_c, perms_to_apply, perm_indices[b][c], except_axis=axis)

                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
                w_c = torch.moveaxis(w_c, axis, 0).reshape((n, -1))
                w_a_b = torch.moveaxis(w_a_b, axis, 0).reshape((n, -1))
                w_a_c = torch.moveaxis(w_a_c, axis, 0).reshape((n, -1))
                w_b_c = torch.moveaxis(w_b_c, axis, 0).reshape((n, -1))

                # similarity of the original weights (e.g. w_a) with the permuted ones (e.g. w_a_b)
                similarities[a][b] += w_a @ w_a_b.T
                similarities[a][c] += w_a @ w_a_c.T
                similarities[b][c] += w_b @ w_b_c.T

            similarities[b][a] = similarities[a][b].T
            similarities[c][a] = similarities[a][c].T
            similarities[c][b] = similarities[b][c].T

            canonical_combinations = [(a, b), (a, c), (b, c)]

            old_similarity = 0.0

            pylogger.info("old pairwise similarities")
            for fixed, permutee in canonical_combinations:
                # e.g. canonical combination is (a, b) then b is permutee and a is fixed, i.e. b -> a
                pairwise_sim = compute_weights_similarity(
                    similarities[fixed][permutee], perm_indices[fixed][permutee][p]
                )
                pylogger.info(f"\t fixed: {fixed}, permutee: {permutee}, pairwise_sim: {pairwise_sim}")
                old_similarity += pairwise_sim

            uber_matrix = three_models_uber_matrix(
                similarities[a][b], similarities[a][c], similarities[b][c], perm_dim=n
            )

            sync_matrix = optimize_synchronization(uber_matrix, n, method)

            sync_perm_matrices = parse_three_models_sync_matrix(sync_matrix, n, symbols, combinations)

            for fixed, permutee in canonical_combinations:
                P_fixed_permutee = sync_perm_matrices[(fixed, permutee)]
                perm_indices[fixed][permutee][p] = perm_matrix_to_perm_indices(P_fixed_permutee)
                perm_indices[permutee][fixed][p] = perm_matrix_to_perm_indices(P_fixed_permutee.T)

            new_similarity = 0.0
            pylogger.info("new pairwise similarities")
            for fixed, permutee in canonical_combinations:
                pairwise_sim = compute_weights_similarity(
                    similarities[fixed][permutee], perm_indices[fixed][permutee][p]
                )
                pylogger.info(f"\t fixed: {fixed}, permutee: {permutee}, pairwise_sim: {pairwise_sim}")
                new_similarity += pairwise_sim

            pylogger.info(f"Iteration {iteration}, Permutation {p}: {(new_similarity - old_similarity)}")

            progress = progress or new_similarity > old_similarity + 1e-12

        if not progress:
            break

    return perm_indices


def optimize_synchronization(uber_matrix, n, method="stiefel"):
    import matlab.engine

    m = uber_matrix.shape[0] // n

    eng = matlab.engine.start_matlab()
    matlab_path = PROJECT_ROOT / "matlab" / "SparseStiefelOpt"
    eng.addpath(str(matlab_path), nargout=0)

    dimVector = np.array([n] * m)
    W = np.float64(uber_matrix.detach().numpy())
    d = n

    inputs_path = matlab_path / "inputs.mat"
    outputs_path = matlab_path / "outputs.mat"

    inputs = {
        "W": W,
        "dimVector": dimVector,
        "d": d,
        "outputs_path": str(outputs_path),
    }

    if method == "stiefel":
        inputs["vis"] = 0
        scipy.io.savemat(inputs_path, inputs)

        eng.SparseStiefelSync(nargout=1)

        # sync_matrix = scipy.io.loadmat(str(outputs_path))["Uproj"]
        sync_matrix = scipy.io.loadmat(str(outputs_path))["Wout"]

    elif method == "spectral":
        inputs["eigMode"] = "eigs"
        scipy.io.savemat(inputs_path, inputs)

        eng.mmatch_spectral(nargout=1)

        sync_matrix = scipy.io.loadmat(str(outputs_path))["X"]

    elif method == "nmfSync":
        inputs["dimVector"] = dimVector.reshape(m, 1)
        inputs["eigMode"] = "eigs"
        inputs["theta"] = []
        inputs["verbose"] = 0

        scipy.io.savemat(inputs_path, inputs)

        eng.nmfSync(nargout=1)

        sync_matrix = scipy.io.loadmat(str(outputs_path))["U"]

    else:
        raise ValueError(f"Unknown method {method}")

    eng.quit()

    return torch.tensor(sync_matrix)


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


def construct_uber_matrix_2(
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
