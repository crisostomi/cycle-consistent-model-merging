import copy
import itertools
import logging
from typing import Dict, List, Set, Tuple

import torch
from pytorch_lightning import LightningModule
from scipy.optimize import fminbound
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
        perm_sizes = {p: orig_params[axes[0][0]].shape[axes[0][1]] for p, axes in permutation_spec.perm_to_axes.items()}

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
        perms_to_apply = ps.axes_to_perm[param_name]

        param = get_permuted_param(param, perms_to_apply, perm_matrices)
        permuted_params[param_name] = param

    return permuted_params


def weight_matching_gradient_fn(
    params_a, params_b, P_curr, P_curr_name, perm_to_axes, not_visited_params, perm_names, all_perm_indices, gradients
):
    """
    Compute gradient of the weight matching objective function w.r.t. P_curr and P_prev.
    sim = <Wa_i, Pi Wb_i P_{i-1}^T>_f where f is the Frobenius norm, rewrite it as < A, xBy^T>_f where A = Wa_i, x = Pi, B = Wb_i, y = P_{i-1}

    Returns:
        grad_P_curr: Gradient of objective function w.r.t. P_curr.
        grad_P_prev: Gradient of objective function w.r.t. P_prev.
    """

    # all the params that are permuted by this permutation matrix, together with the axis on which it acts
    # e.g. ('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..
    params_and_axes: List[Tuple[str, int]] = perm_to_axes[P_curr_name]

    P_prev_name = None

    for params_name, axis in params_and_axes:

        # axis != 0 will be considered when P_curr will be P_prev for some next layer
        if axis == 0:

            not_visited_params[params_name].remove(0)

            Wa, Wb = params_a[params_name], params_b[params_name]
            assert Wa.shape == Wa.shape

            num_neurons = P_curr.shape[0]

            P_prev_name, P_prev = get_prev_permutation(perm_names, params_name, perm_to_axes, all_perm_indices)

            if P_prev is not None:

                not_visited_params[params_name].remove(1)

                grad_P_curr = compute_gradient_P_curr(Wa, Wb, P_prev)

                grad_P_prev = compute_gradient_P_prev(Wa, Wb, P_curr)

            else:
                grad_P_curr = Wa.reshape((num_neurons, -1)) @ Wb.reshape((num_neurons, -1)).T

            gradients[P_curr_name] += grad_P_curr

            if P_prev is not None:
                gradients[P_prev_name] += grad_P_prev


def compute_obj_function(params_a, params_b, P_curr, P_curr_name, perm_to_axes, perm_names, all_perm_indices):
    """
    Compute gradient of the weight matching objective function w.r.t. P_curr and P_prev.
    sim = <Wa_i, Pi Wb_i P_{i-1}^T>_f where f is the Frobenius norm, rewrite it as < A, xBy^T>_f where A = Wa_i, x = Pi, B = Wb_i, y = P_{i-1}

    Returns:
        grad_P_curr: Gradient of objective function w.r.t. P_curr.
        grad_P_prev: Gradient of objective function w.r.t. P_prev.
    """

    # all the params that are permuted by this permutation matrix, together with the axis on which it acts
    # e.g. ('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..
    params_and_axes: List[Tuple[str, int]] = perm_to_axes[P_curr_name]

    obj = 0.0

    for params_name, axis in params_and_axes:

        # axis != 0 will be considered when P_curr will be P_prev for some next layer
        if axis == 0:

            Wa, Wb = params_a[params_name], params_b[params_name]
            assert Wa.shape == Wa.shape

            # permute B according to P_i
            Wb_perm = apply_perm(perm=perm_matrix_to_perm_indices(P_curr), x=Wb, axis=0)
            if len(Wb.shape) == 2:
                assert torch.all(Wb_perm == P_curr @ Wb)

            P_prev_name, P_prev = get_prev_permutation(perm_names, params_name, perm_to_axes, all_perm_indices)

            if P_prev is not None:
                # also permute B according to P_{i-1}^Ts
                # Wb_perm = Wb_perm @ P_prev.T
                Wb_perm = apply_perm(perm=perm_matrix_to_perm_indices(P_prev).T, x=Wb_perm, axis=1)
                if len(Wb.shape) == 2:
                    assert torch.all(Wb_perm == P_curr @ Wb @ P_prev.T)

            if len(Wa.shape) == 1:
                obj += Wa.T @ Wb_perm
            elif len(Wa.shape) == 2:
                obj += torch.trace(Wa.T @ Wb_perm).numpy()
            elif len(Wa.shape) == 3:
                # The einsum string 'ijk,ilk->ij' indicates the following operation:
                # - 'ijk' and 'ilk' are the dimensions of Wa and Wb respectively.
                # - The shared dimensions 'j' and 'k' indicate where the element-wise multiplication and summation will occur.
                # - 'ij' in the output string denotes the resulting dimensions after the operation.
                obj += torch.trace(torch.einsum("ijk,jnk->in", Wa.transpose(1, 0), Wb_perm)).numpy()
            else:
                obj += torch.trace(torch.einsum("ijkm,jnkm->in", Wa.transpose(1, 0), Wb_perm)).numpy()

    return obj


def compute_gradient_P_curr(Wa, Wb, P_prev):
    """
    (A P_{l-1} B^T)
    """
    assert Wa.shape == Wb.shape
    assert P_prev.shape[0] == Wb.shape[1]

    # P_{l-1} B^T
    B_perm = apply_perm(perm=perm_matrix_to_perm_indices(P_prev), x=Wb.transpose(0, 1), axis=0)
    if len(Wb.shape) == 2:
        assert torch.all(B_perm == P_prev @ Wb.T)

    # Using einsum to compute A * B^T
    # The einsum string 'ijkl,jmkl->im' indicates the following operation:
    # - 'ijkl' and 'jmkl' are the dimensions of A and B respectively.
    # - The shared dimension 'j' indicates where the summation will occur (like in matrix multiplication).
    # - 'im' in the output string denotes the resulting dimensions after the operation.
    if len(Wa.shape) == 2:
        grad_P_curr = Wa @ B_perm
    elif len(Wa.shape) == 3:
        grad_P_curr = torch.einsum("ijk,jmk->im", Wa, B_perm)
    else:
        grad_P_curr = torch.einsum("ijkl,jmkl->im", Wa, B_perm)

    return grad_P_curr


def compute_gradient_P_prev(Wa, Wb, P_curr):
    """
    (A^T P_l B)

    """
    assert P_curr.shape[0] == Wb.shape[0]

    grad_P_prev = None

    # (P_l B)
    Wb_perm = apply_perm(perm=perm_matrix_to_perm_indices(P_curr), x=Wb, axis=0)
    if len(Wb.shape) == 2:
        assert torch.all(Wb_perm == P_curr @ Wb)

    if len(Wa.shape) == 2:
        grad_P_prev = Wa.T @ Wb_perm
    elif len(Wa.shape) == 3:
        grad_P_prev = torch.einsum("ijk,jnk->in", Wa.transpose(1, 0), Wb_perm)
    else:
        grad_P_prev = torch.einsum("ijkm,jnkm->in", Wa.transpose(1, 0), Wb_perm)
        # ijkm,inkm->jn
        # ijkl, jmkl -> im (+ transpose of Wa)

    return grad_P_prev


def get_prev_permutation(perm_names, params_name, perm_to_axes, all_perm_indices):
    P_prev_name, P_prev = None, None
    for other_p in perm_names:

        params_perm_by_other_p = [tup[0] if tup[1] == 1 else None for tup in perm_to_axes[other_p]]
        if params_name in params_perm_by_other_p:
            P_prev_name = other_p
            P_prev = perm_indices_to_perm_matrix(all_perm_indices[P_prev_name])

    return P_prev_name, P_prev


# ij, jl -> il
# ij ij
# ji ij -> jj


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


def apply_perm(x, perm, axis):
    """
    Permute a tensor along a specified axis.

    Parameters:
    X (torch.Tensor): The input tensor, can be 1D, 2D, 3D, or 4D.
    P (list or torch.Tensor): The permutation to be applied.
    axis (int): The axis along which to permute.

    Returns:
    torch.Tensor: The permuted tensor.
    """
    # Ensure P is a torch.Tensor
    if not isinstance(perm, torch.Tensor):
        perm = torch.tensor(perm)

    # Check if the axis is valid for the tensor dimensions
    if axis < 0 or axis >= x.dim():
        raise ValueError("Axis is out of bounds for the tensor dimensions")

    # Permute the tensor
    # Generate indices for all dimensions
    idx = [slice(None)] * x.dim()
    # Set the indices for the specified axis to the permutation
    idx[axis] = perm

    return x[idx]


def frank_wolfe_with_sdp_penalty(W, X0, get_gradient_fn, get_objective_fn):
    """
    Maximise an objective f over block permutation matrices.

    Args:
        W: Data involved in the objective function.
        X0: Initialisation matrix.

    Returns:
        X: Optimised matrix.
        obj_vals: Objective values at each iteration.
    """
    n_max_fw_iters = 1000
    convergence_threshold = 1e-6
    X_old = X0

    obj_vals = [get_objective_fn(X_old, W)]

    for jj in range(n_max_fw_iters):

        grad_f = get_gradient_fn(X_old)  # Function that computes gradient of f w.r.t. X_old.

        grad_f_scaled = -grad_f  # Flip sign since we want to maximise.

        # Project gradient onto set of permutation matrices
        D = grad_f_scaled  # project_onto_partial_perm_blockwise(grad_f_scaled)

        # Line search to find step size (convex combination of X_old and D)
        D_minus_X_old = D - X_old

        def fun(t):
            return get_objective_fn(X_old + t * D_minus_X_old)

        eta = fminbound(fun, 0, 1)

        X = X_old + eta * D_minus_X_old

        # Check convergence
        obj_val = get_objective_fn(X, W)
        obj_vals.append(obj_val)

        if abs(obj_val / obj_vals[-2] - 1) < convergence_threshold:
            break

        X_old = X

    return X, obj_vals
