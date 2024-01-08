import copy
import logging
from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.utils import (
    PermutationIndices,
    PermutationMatrix,
    perm_indices_to_perm_matrix,
    perm_matrix_to_perm_indices,
)
from ccmm.utils.utils import ModelParams

pylogger = logging.getLogger(__name__)

LAYER_TO_VAR = {
    "P_bg0": 1.5,
    "P_bg1": 7.5,
    "P_bg2": 4.0,
    "P_bg3": 4.0,
    "P_layer1.0_inner": 0.3,
    "P_layer1.1_inner": 0.3,
    "P_layer1.2_inner": 0.25,
    "P_layer2.0_inner": 0.1,
    "P_layer2.1_inner": 0.3,
    "P_layer2.2_inner": 0.4,
    "P_layer3.0_inner": 0.4,
    "P_layer3.1_inner": 0.4,
    "P_layer3.2_inner": 0.05,
}


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


def apply_permutation(ps: PermutationSpec, perm_matrices, all_params):
    """Apply a `perm` to `params`."""

    permuted_params = {}

    for param_name, param in all_params.items():
        param = copy.deepcopy(param)
        perms_to_apply = ps.axes_to_perm[param_name]

        param = get_permuted_param(param, perms_to_apply, perm_matrices)
        permuted_params[param_name] = param

    return permuted_params


def weight_matching(
    ps: PermutationSpec,
    fixed: ModelParams,
    permutee: ModelParams,
    max_iter=100,
    init_perm=None,
    alternate_diffusion_params=None,
):
    """
    Find a permutation of params_b to make them match params_a.

    :param ps: PermutationSpec
    :param target: the parameters to match
    :param to_permute: the parameters to permute
    """
    params_a, params_b = fixed, permutee

    # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
    perm_sizes = {
        p: params_a[params_and_axes[0][0]].shape[params_and_axes[0][1]]
        for p, params_and_axes in ps.perm_to_axes.items()
    }

    # initialize with identity permutation if none given
    all_perm_indices = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    # e.g. P0, P1, ..
    perm_names = list(all_perm_indices.keys())

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        progress = False

        # iterate over the permutation matrices in random order
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            num_neurons = perm_sizes[p]

            sim_matrix = torch.zeros((num_neurons, num_neurons))
            dist_aa = torch.zeros((num_neurons, num_neurons))
            dist_bb = torch.zeros((num_neurons, num_neurons))

            # all the params that are permuted by this permutation matrix, together with the axis on which it acts
            # e.g. ('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..
            params_and_axes: List[Tuple[str, int]] = ps.perm_to_axes[p]

            for params_name, axis in params_and_axes:
                w_a = params_a[params_name]
                w_b = params_b[params_name]

                assert w_a.shape == w_b.shape

                perms_to_apply = ps.axes_to_perm[params_name]

                w_b = get_permuted_param(w_b, perms_to_apply, all_perm_indices, except_axis=axis)

                w_a = torch.moveaxis(w_a, axis, 0).reshape((num_neurons, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((num_neurons, -1))

                sim_matrix += w_a @ w_b.T

                dist_aa += torch.cdist(w_a, w_a)
                dist_bb += torch.cdist(w_b, w_b)

            perm_indices = solve_linear_assignment_problem(sim_matrix)

            if alternate_diffusion_params:
                perm_matrix = perm_indices_to_perm_matrix(perm_indices)
                perm_indices = alternating_diffusion(perm_matrix, dist_aa, dist_bb, alternate_diffusion_params, p)

            old_similarity = compute_weights_similarity(sim_matrix, all_perm_indices[p])

            all_perm_indices[p] = perm_indices

            new_similarity = compute_weights_similarity(sim_matrix, all_perm_indices[p])

            pylogger.info(f"Iteration {iteration}, Permutation {p}: {new_similarity - old_similarity}")

            progress = progress or new_similarity > old_similarity + 1e-12

        if not progress:
            break

    return all_perm_indices


def compute_weights_similarity(similarity_matrix, perm_indices):

    """
    similarity_matrix: matrix s.t. S[i, j] = w_a[i] @ w_b[j]

    we sum over the cells identified by perm_indices, i.e. S[i, perm_indices[i]] for all i
    """

    n = len(perm_indices)

    similarity = torch.sum(similarity_matrix[torch.arange(n), perm_indices.long()])

    return similarity


def alternating_diffusion(
    initial_perm: PermutationMatrix, dist_aa: Tensor, dist_bb: Tensor, alternate_diffusion_params, param_name
) -> PermutationIndices:
    """

    :param initial_perm: initial permutation matrix obtained from LAP
    """
    sim_matrix = initial_perm
    initial_perm_indices = perm_matrix_to_perm_indices(initial_perm)

    var_percentage = alternate_diffusion_params.var_percentage
    K = alternate_diffusion_params.num_diffusion_steps

    dist_bb_perm = dist_bb @ initial_perm.T

    var_a = dist_aa.max() * var_percentage
    var_b = dist_bb.max() * var_percentage

    kernel_A = torch.exp(-(dist_aa).pow(2) / (2 * var_a))
    kernel_B = torch.exp(-(dist_bb_perm).pow(2) / (2 * var_b))

    sim_matrix = kernel_A @ kernel_B.T

    old_sim_value = compute_weights_similarity(sim_matrix, perm_matrix_to_perm_indices(initial_perm))

    old_perm_indices = initial_perm_indices

    original_kernel_sim_matrix = sim_matrix.clone()

    for k in range(K):
        perm_indices = solve_linear_assignment_problem(sim_matrix)

        new_sim_value = compute_weights_similarity(original_kernel_sim_matrix, perm_indices)

        if new_sim_value <= old_sim_value:
            pylogger.info(f"Exiting after {k} steps, difference: {new_sim_value - old_sim_value}")
            return old_perm_indices

        # a large var will have a large smoothing effect, i.e. it will make all values equal
        # a small var will have a small smoothing effect, i.e. it will preserve the original values
        # radius_a = calculate_global_radius(dist_aa, target_percentage=0.8)
        # radius_b = calculate_global_radius(dist_bb, target_percentage=0.8)
        # var_a = radius_a *  max(torch.log10(dist_aa.mean()), 0.1) * ((K - k) / K)
        # var_b = radius_b *  max(torch.log10(dist_bb.mean()), 0.1) * ((K - k) / K)
        var_a = dist_aa.max() * var_percentage * ((K - k) / K)
        var_b = dist_bb.max() * var_percentage * ((K - k) / K)

        perm_matrix = perm_indices_to_perm_matrix(perm_indices)
        dist_bb_perm = dist_bb @ perm_matrix.T

        kernel_A = torch.exp(-(dist_aa).pow(2) / (2 * var_a))
        kernel_B = torch.exp(-(dist_bb_perm).pow(2) / (2 * var_b))

        sim_matrix = kernel_A @ kernel_B.T

        old_sim_value = new_sim_value.clone()
        old_perm_indices = perm_indices.clone()

    pylogger.info(f"Did all {K} iterations.")

    return perm_indices


def solve_linear_assignment_problem(sim_matrix: torch.Tensor):
    ri, ci = linear_sum_assignment(sim_matrix.detach().numpy(), maximize=True)

    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    return torch.tensor(ci)
