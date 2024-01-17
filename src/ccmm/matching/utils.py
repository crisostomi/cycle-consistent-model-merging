import copy
import itertools
import logging
from typing import Dict, List, Set, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from ccmm.matching.permutation_spec import PermutationSpec

# shape (n, n), contains the permutation matrix, e.g. [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
PermutationMatrix = Tensor

# shape (n), contains the indices of the target permutation, e.g. [0, 3, 2, 1]
PermutationIndices = Tensor

pylogger = logging.getLogger(__name__)


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
    return perm_matrix.nonzero()[:, 1].long()


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


def create_artificially_permuted_models(
    seed_model: LightningModule, permutation_spec: PermutationSpec, num_models: int
) -> List[LightningModule]:
    artificial_models = []

    for _ in range(num_models):
        artificial_model = copy.deepcopy(seed_model)

        orig_params = seed_model.model.state_dict()

        # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
        perm_sizes = {
            p: orig_params[axes[0][0]].shape[axes[0][1]] for p, axes in permutation_spec.perm_to_layers_and_axes.items()
        }

        # initialize with identity permutation if none given
        perm_indices = {p: torch.arange(n) for p, n in perm_sizes.items()}
        # e.g. P0, P1, ..
        perm_names = list(perm_indices.keys())

        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            perm_indices[p] = torch.randperm(n)

        permuted_params = apply_permutation_to_statedict(permutation_spec, perm_indices, orig_params)
        artificial_model.model.load_state_dict(permuted_params)

        artificial_models.append(artificial_model)

    return artificial_models


def get_permuted_param(param, perms_to_apply, perm_matrices, except_axis=None):
    """
    Apply to the parameter `param` all the permutations computed until the current step.

    :param param: the parameter to permute
    :param perms_to_apply: the list of permutations to apply to the parameter
    :param perm_matrices: the list of permutation matrices
    :param except_axis: axis to skip
    """
    for axis, perm_id in enumerate(perms_to_apply):
        # skip the axis we're trying to permute
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis
        if perm_id is not None:
            param = torch.index_select(param, axis, perm_matrices[perm_id].int())

    return param


def apply_permutation_to_statedict(ps: PermutationSpec, perm_matrices, all_params):
    """Apply a `perm` to `params`."""

    permuted_params = {}

    for param_name, param in all_params.items():
        param = copy.deepcopy(param)
        perms_to_apply = ps.layer_and_axes_to_perm[param_name]

        param = get_permuted_param(param, perms_to_apply, perm_matrices)
        permuted_params[param_name] = param

    return permuted_params


# def frank_wolfe_with_sdp_penalty(W, X0, get_gradient_fn, get_objective_fn):
#     """
#     Maximise an objective f over block permutation matrices.

#     Args:
#         W: Data involved in the objective function.
#         X0: Initialisation matrix.

#     Returns:
#         X: Optimised matrix.
#         obj_vals: Objective values at each iteration.
#     """
#     n_max_fw_iters = 1000
#     convergence_threshold = 1e-6
#     X_old = X0

#     obj_vals = [get_objective_fn(X_old, W)]

#     for jj in range(n_max_fw_iters):

#         grad_f = get_gradient_fn(X_old)  # Function that computes gradient of f w.r.t. X_old.

#         grad_f_scaled = -grad_f  # Flip sign since we want to maximise.

#         # Project gradient onto set of permutation matrices
#         D = grad_f_scaled  # project_onto_partial_perm_blockwise(grad_f_scaled)

#         # Line search to find step size (convex combination of X_old and D)
#         D_minus_X_old = D - X_old

#         def fun(t):
#             return get_objective_fn(X_old + t * D_minus_X_old)

#         eta = fminbound(fun, 0, 1)

#         X = X_old + eta * D_minus_X_old

#         # Check convergence
#         obj_val = get_objective_fn(X, W)
#         obj_vals.append(obj_val)

#         if abs(obj_val / obj_vals[-2] - 1) < convergence_threshold:
#             break

#         X_old = X

#     return X, obj_vals


# def apply_perm(perm, x, axis):
#     assert perm.shape[0] == perm.shape[1]
#     assert x.shape[axis] == perm.shape[0]

#     # Bring the specified axis to the front
#     x = x.moveaxis(axis, 0)

#     # Store the original shape and reshape for matrix multiplication
#     original_shape = x.shape
#     x = x.reshape(x.shape[0], -1)

#     # Apply the permutation
#     x_permuted = perm @ x

#     # Reshape back to the expanded original shape
#     x_permuted = x_permuted.reshape(original_shape)

#     # Move the axis back to its original position
#     x_permuted = x_permuted.moveaxis(0, axis)

#     return x_permuted
