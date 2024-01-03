import copy
import logging
from collections import defaultdict
from typing import Dict, List, NamedTuple, Set, Tuple

import numpy as np
import scipy
import torch
from pytorch_lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

from ccmm.utils.matching_utils import (
    PermutationIndices,
    PermutationMatrix,
    parse_three_models_sync_matrix,
    perm_indices_to_perm_matrix,
    perm_matrix_to_perm_indices,
    three_models_uber_matrix,
)
from ccmm.utils.utils import ModelParams, block

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


def conv_axes(name, p_in, p_out):
    return {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }


def norm_layer_axes(name, p):
    return {f"{name}.weight": (p,), f"{name}.bias": (p,)}


def dense_layer_axes(name, p_in, p_out, bias=True):
    # it's (p_in , p_out) in git-rebasin (due to jax)
    return {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}


def easyblock_axes(name, p):
    """Easy blocks that use a residual connection, without any change in the number of channels."""
    return {
        **conv_axes(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm_layer_axes(f"{name}.bn1", p),
        **conv_axes(f"{name}.conv2", f"P_{name}_inner", p),
        **norm_layer_axes(f"{name}.bn2", p),
    }


def shortcut_block_axes(name, p_in, p_out):
    """This is for blocks that use a residual connection, but change the number of channels via a Conv."""
    return {
        **conv_axes(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm_layer_axes(f"{name}.bn1", f"P_{name}_inner"),
        **conv_axes(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm_layer_axes(f"{name}.bn2", p_out),
        **conv_axes(f"{name}.shortcut.0", p_in, p_out),
        **norm_layer_axes(f"{name}.shortcut.1", p_out),
    }


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


class PermutationSpecBuilder:
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> list:
        pass

    def permutation_spec_from_axes_to_perm(self, axes_to_perm: dict) -> PermutationSpec:
        perm_to_axes = defaultdict(list)

        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))

        return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


class MLPPermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers

    def create_permutation(self) -> PermutationSpec:
        L = self.num_hidden_layers
        assert L >= 1

        axes_to_perm = {
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": (f"P_{i}", f"P_{i-1}") for i in range(1, L)},
            **{f"layer{i}.bias": (f"P_{i}",) for i in range(L)},
            f"layer{L}.weight": (None, f"P_{L-1}"),
            f"layer{L}.bias": (None,),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class ResNet20PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        axes_to_perm = {
            **conv_axes("conv1", None, "P_bg0"),
            **norm_layer_axes("bn1", "P_bg0"),
            #
            **shortcut_block_axes("layer1.0", "P_bg0", "P_bg1"),
            **easyblock_axes(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock_axes("layer1.2", "P_bg1"),
            #
            **shortcut_block_axes("layer2.0", "P_bg1", "P_bg2"),
            **easyblock_axes(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock_axes("layer2.2", "P_bg2"),
            #
            **shortcut_block_axes("layer3.0", "P_bg2", "P_bg3"),
            **easyblock_axes(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock_axes("layer3.2", "P_bg3"),
            **norm_layer_axes("out_bn", "P_bg3"),
            #
            **dense_layer_axes("linear", "P_bg3", None),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class ResNet50PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        # TODO: invert conv and batch norm as in ResNet20

        return self.permutation_spec_from_axes_to_perm(
            {
                **conv_axes("conv1", None, "P_bg0"),
                **norm_layer_axes("bn1", "P_bg0"),
                **shortcut_block_axes("layer1.0", "P_bg0", "P_bg1"),
                **easyblock_axes(
                    "layer1.1",
                    "P_bg1",
                ),
                **easyblock_axes("layer1.2", "P_bg1"),
                **easyblock_axes("layer1.3", "P_bg1"),
                **easyblock_axes("layer1.4", "P_bg1"),
                **easyblock_axes("layer1.5", "P_bg1"),
                **easyblock_axes("layer1.6", "P_bg1"),
                **easyblock_axes("layer1.7", "P_bg1"),
                **shortcut_block_axes("layer2.0", "P_bg1", "P_bg2"),
                **easyblock_axes(
                    "layer2.1",
                    "P_bg2",
                ),
                **easyblock_axes("layer2.2", "P_bg2"),
                **easyblock_axes("layer2.3", "P_bg2"),
                **easyblock_axes("layer2.4", "P_bg2"),
                **easyblock_axes("layer2.5", "P_bg2"),
                **easyblock_axes("layer2.6", "P_bg2"),
                **easyblock_axes("layer2.7", "P_bg2"),
                **shortcut_block_axes("layer3.0", "P_bg2", "P_bg3"),
                **easyblock_axes(
                    "layer3.1",
                    "P_bg3",
                ),
                **easyblock_axes("layer3.2", "P_bg3"),
                **easyblock_axes("layer3.3", "P_bg3"),
                **easyblock_axes("layer3.4", "P_bg3"),
                **easyblock_axes("layer3.5", "P_bg3"),
                **easyblock_axes("layer3.6", "P_bg3"),
                **easyblock_axes("layer3.7", "P_bg3"),
                **norm_layer_axes("out_bn", "P_bg3"),
                **dense_layer_axes("linear", "P_bg3", None),
            }
        )


class VGG16PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
        layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
        return self.permutation_spec_from_axes_to_perm(
            {
                # first features
                "features.0.weight": ("P_Conv_0", None, None, None),
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),
                "features.1.running_mean": ("P_Conv_0", None),
                "features.1.running_var": ("P_Conv_0", None),
                "features.1.num_batches_tracked": (),
                **{
                    f"features.{layers_with_conv[i]}.weight": (
                        f"P_Conv_{layers_with_conv[i]}",
                        f"P_Conv_{layers_with_conv_b4[i]}",
                        None,
                        None,
                    )
                    for i in range(len(layers_with_conv))
                },
                **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
                # bn
                **{
                    f"features.{layers_with_bn[i]}.weight": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.bias": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.running_mean": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.running_var": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{f"features.{layers_with_bn[i]}.num_batches_tracked": () for i in range(len(layers_with_bn))},
                **dense_layer_axes("classifier", "P_Conv_40", "P_Dense_0", False),
            }
        )


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
    # var_a = var_b = LAYER_TO_VAR[param_name]

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


def create_artificially_permuted_models(
    seed_model: LightningModule, permutation_spec: PermutationSpec, num_models: int
) -> List[LightningModule]:
    artificial_models = []

    for _ in range(num_models):
        artificial_model = copy.deepcopy(seed_model)

        orig_params = seed_model.model.state_dict()

        # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
        perm_sizes = {p: orig_params[axes[0][0]].shape[axes[0][1]] for p, axes in permutation_spec.perm_to_axes.items()}

        # initialize with identity permutation if none given
        perm_indices = {p: torch.arange(n) for p, n in perm_sizes.items()}
        # e.g. P0, P1, ..
        perm_names = list(perm_indices.keys())

        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            perm_indices[p] = torch.randperm(n)

        permuted_params = apply_permutation(permutation_spec, perm_indices, orig_params)
        artificial_model.model.load_state_dict(permuted_params)

        artificial_models.append(artificial_model)

    return artificial_models
