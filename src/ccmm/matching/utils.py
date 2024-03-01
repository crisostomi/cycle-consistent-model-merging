import copy
import itertools
import json
import logging
from typing import Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from pytorch_lightning import LightningModule
from torch import Tensor

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.utils.utils import to_np

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
        if perm.dim() == 1:
            perm_matrix = perm_indices_to_perm_matrix(perm)
        else:
            perm_matrix = perm

        inv_perm_matrix = perm_matrix.T

        if perm.dim() == 1:
            inv_permutations[perm_name] = perm_matrix_to_perm_indices(inv_perm_matrix)
        else:
            inv_permutations[perm_name] = inv_perm_matrix

    return inv_permutations


def perm_indices_to_perm_matrix(perm_indices: PermutationIndices):
    n = len(perm_indices)
    perm_matrix = torch.eye(n, device=perm_indices.device)[perm_indices.long()]
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


def perm_rows(x, perm):
    """
    X ~ (n, d0) or (n, d0, d1) or (n, d0, d1, d2)
    perm ~ (n, n)
    """
    assert x.shape[0] == perm.shape[0]
    assert perm.dim() == 2 and perm.shape[0] == perm.shape[1]

    input_dims = "jklm"[: x.dim()]
    output_dims = "iklm"[: x.dim()]

    ein_string = f"ij,{input_dims}->{output_dims}"

    return torch.einsum(ein_string, perm, x)


def perm_cols(x, perm):
    """
    X ~ (d0, n) or (d0, d1, n) or (d0, d1, d2, n)
    perm ~ (n, n)
    """
    assert x.shape[1] == perm.shape[0]
    assert perm.shape[0] == perm.shape[1]

    x = x.transpose(1, 0)
    perm = perm.transpose(1, 0)

    permuted_x = perm_rows(x=x, perm=perm)

    return permuted_x.transpose(1, 0)


def get_permuted_param(param, perms_to_apply, perm_matrices, except_axis=None):
    """
    Apply to the parameter `param` all the permutations computed until the current step.

    :param param: the parameter to permute
    :param perms_to_apply: the list of permutations to apply to the parameter
    :param perm_matrices: the list of permutation matrices
    :param except_axis: axis to skip
    """

    for axis, perm_id in enumerate(perms_to_apply):

        if axis == except_axis or perm_id is None:
            continue

        perm = perm_matrices[perm_id].cpu()
        if perm.dim() == 1:
            # permute by indices
            param = torch.index_select(param, axis, perm.int())

        else:
            # permute by matrix
            param = perm_tensor_by_perm_matrix(param, perm, axis)

        if param.dim() == 2 and perm.dim() == 1:
            assert torch.allclose(
                torch.index_select(param, axis, perm.int()),
                perm_tensor_by_perm_matrix(param, perm_indices_to_perm_matrix(perm), axis),
                atol=1e-3,
            )

    return param


def perm_tensor_by_perm_matrix(tens, perm, axis):
    assert axis == 0 or axis == 1
    if axis == 0:
        tens = perm_rows(tens, perm)
    else:
        tens = perm_cols(tens, perm.T)

    return tens


def apply_permutation_to_statedict(ps: PermutationSpec, perm_matrices, all_params):
    """Apply a `perm` to `params`."""

    permuted_params = {}

    for param_name, param in all_params.items():

        param_name_in_perm_dict = param_name

        if "num_batches_tracked" in param_name:
            permuted_params[param_name] = param
            continue

        if "running_mean" in param_name or "running_var" in param_name:
            layer_name = ".".join(param_name.split(".")[:-1])
            param_name_in_perm_dict = layer_name + ".weight"

        assert (
            param_name_in_perm_dict in ps.layer_and_axes_to_perm
        ), f"param_name {param_name} not found in ps.layer_and_axes_to_perm"

        param = copy.deepcopy(param)
        perms_to_apply = ps.layer_and_axes_to_perm[param_name_in_perm_dict]

        param = get_permuted_param(param, perms_to_apply, perm_matrices)
        permuted_params[param_name] = param

    return permuted_params


def plot_permutation_history_animation(perm_history, cfg):
    def update(frame, ax, perm_history, perm_index):
        perm = to_np(perm_history[frame][perm_index])
        ax.imshow(perm, cmap="gray")

    for perm_index in perm_history[0].keys():
        fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
        ani = FuncAnimation(fig, update, fargs=(ax, perm_history, perm_index), frames=len(perm_history), interval=500)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        animation_path = cfg.permutations_path / f"animation_{perm_index}.mp4"
        ani.save(animation_path)
        plt.close(fig)


def unfactor_permutations(permutations, matrix_format=False):
    if matrix_format:
        raise NotImplementedError

    symbols = set(permutations.keys())

    unfactored_permutations = {
        symbol: {
            permutee: {perm: None for perm in permutations[symbol].keys()} for permutee in symbols.difference(symbol)
        }
        for symbol in symbols
    }
    for symbol, perms in permutations.items():
        for perm_name, perm in perms.items():
            if perm is not None:
                permutations[symbol][perm_name] = torch.tensor(perm)

    combinations = get_all_symbols_combinations(symbols)
    for fixed, permutee in combinations:
        for perm in permutations[fixed].keys():
            res = (
                perm_indices_to_perm_matrix(permutations[fixed][perm])
                @ perm_indices_to_perm_matrix(permutations[permutee][perm]).T
            )

            unfactored_permutations[fixed][permutee][perm] = perm_matrix_to_perm_indices(res)

    return unfactored_permutations


def load_permutations(
    path, factored=False, matrix_format=False
) -> Dict[str, Union[PermutationIndices, PermutationMatrix]]:
    with open(path, "r") as f:
        permutations = json.load(f)

    if factored:
        return unfactor_permutations(permutations, matrix_format)

    if matrix_format:
        for source, targets in permutations.items():
            for target, source_target_perms in targets.items():
                for perm_name, perm in source_target_perms.items():
                    if perm is not None:
                        permutations[source][target][perm_name] = torch.tensor(perm)

        return permutations
    else:
        for source, targets in permutations.items():
            for target, source_target_perms in targets.items():
                for perm_name, perm in source_target_perms.items():
                    if perm is not None:
                        permutations[source][target][perm_name] = torch.tensor(perm)

        return permutations

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


def permute_batchnorm(model, perm, perm_dict, map_param_index):

    for name, module in model.named_modules():

        if "BatchNorm" in str(type(module)):

            if name + ".weight" in map_param_index:

                if module.running_mean is None and module.running_var is None:
                    continue

                i = perm_dict[map_param_index[name + ".weight"]]

                index = torch.argmax(perm[i], dim=1) if i is not None else torch.arange(module.running_mean.shape[0])

                module.running_mean.copy_(module.running_mean[index, ...])
                module.running_var.copy_(module.running_var[index, ...])


def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    if not isinstance(v0, np.ndarray):
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = v0 / (np.linalg.norm(v0) + 1e-6)
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return res

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res
