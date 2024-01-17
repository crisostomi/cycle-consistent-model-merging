import logging
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import fminbound
from tqdm import tqdm

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.utils import PermutationIndices, PermutationMatrix, perm_matrix_to_perm_indices
from ccmm.matching.weight_matching import solve_linear_assignment_problem
from ccmm.utils.utils import ModelParams

pylogger = logging.getLogger(__name__)


def frank_wolfe_weight_matching(
    ps: PermutationSpec,
    fixed: ModelParams,
    permutee: ModelParams,
    max_iter=100,
):
    """
    Find a permutation of params_b to make them match params_a.

    :param ps: PermutationSpec
    :param fixed: the parameters to match
    :param permutee: the parameters to permute
    """
    params_a, params_b = fixed, permutee

    # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
    perm_sizes = {}

    # FOR MLP
    # ps.perm_to_layers_and_axes["P_4"] = [("layer4.weight", 0)]

    # FOR RESNET
    ps.perm_to_layers_and_axes["P_final"] = [("linear.weight", 0)]

    for perm_name, params_and_axes in ps.perm_to_layers_and_axes.items():
        # params_and_axes is a list of tuples, e.g. [('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..]
        relevant_params, relevant_axis = params_and_axes[0]
        param_shape = params_a[relevant_params].shape
        perm_sizes[perm_name] = param_shape[relevant_axis]

    # initialize with identity permutation
    all_perm_indices: Dict[str, PermutationIndices] = {p: torch.arange(n) for p, n in perm_sizes.items()}
    perm_matrices: Dict[str, PermutationMatrix] = {p: torch.eye(n) for p, n in perm_sizes.items()}

    old_obj = 0.0
    patience_steps = 0

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        pylogger.info(f"Iteration {iteration}")

        # keep track if the params have been permuted in both axis 0 and 1
        # not_visited_params = {
        #     param_name: {0, 1} if ("bias" not in param_name and "bn" not in param_name) else {0}
        #     for param_name in set(params_a.keys())
        # }

        gradients = weight_matching_gradient_fn_layerwise(
            params_a, params_b, perm_matrices, ps.layer_and_axes_to_perm, perm_sizes
        )

        proj_grads = project_gradients(gradients)

        line_search_step_func = partial(
            line_search_step,
            proj_grads=proj_grads,
            perm_matrices=perm_matrices,
            params_a=params_a,
            params_b=params_b,
            layers_and_axes_to_perms=ps.layer_and_axes_to_perm,
        )

        step_size = fminbound(line_search_step_func, 0, 1)
        pylogger.info(f"Step size: {step_size}")

        perm_matrices = update_perm_matrices(perm_matrices, proj_grads, step_size)

        new_obj = get_global_obj_layerwise(params_a, params_b, perm_matrices, ps.layer_and_axes_to_perm)

        pylogger.info(f"Objective: {np.round(new_obj, 6)}")

        if (new_obj - old_obj) < 1e-6:
            patience_steps += 1
        else:
            patience_steps = 0
            old_obj = new_obj

        if patience_steps >= 15:
            break

    all_perm_indices = {p: perm_matrix_to_perm_indices(perm) for p, perm in perm_matrices.items()}

    return all_perm_indices


def project_gradients(gradients):
    proj_grads = {}

    for perm_name, grad in gradients.items():

        proj_grad = solve_linear_assignment_problem(grad, return_matrix=True)

        proj_grads[perm_name] = proj_grad

    return proj_grads


def line_search_step(
    t: float,
    params_a,
    params_b,
    proj_grads: Dict[str, torch.Tensor],
    perm_matrices: Dict[str, PermutationIndices],
    layers_and_axes_to_perms,
):

    interpolated_perms = {}

    for perm_name, perm in perm_matrices.items():
        proj_grad = proj_grads[perm_name]

        if perm_name in {"P_final", "P_4"}:
            interpolated_perms[perm_name] = perm
            continue

        interpolated_perms[perm_name] = (1 - t) * perm + t * proj_grad

    tot_obj = get_global_obj_layerwise(params_a, params_b, interpolated_perms, layers_and_axes_to_perms)

    return -tot_obj


def get_global_obj_layerwise(params_a, params_b, perm_matrices, layers_and_axes_to_perms):

    tot_obj = 0.0

    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        assert layer in params_a.keys()
        assert layer in params_b.keys()

        Wa, Wb = params_a[layer], params_b[layer]
        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)

        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices.keys()
        row_perm = perm_matrices[row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0])

        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices.keys()
        col_perm = perm_matrices[col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1])

        layer_similarity = compute_layer_similarity(Wa, Wb, row_perm, col_perm)

        tot_obj += layer_similarity

    return tot_obj


def weight_matching_gradient_fn_layerwise(params_a, params_b, perm_matrices, layers_and_axes_to_perms, perm_sizes):
    """
    Compute gradient of the weight matching objective function w.r.t. P_curr and P_prev.
    sim = <Wa_i, Pi Wb_i P_{i-1}^T>_f where f is the Frobenius norm, rewrite it as < A, xBy^T>_f where A = Wa_i, x = Pi, B = Wb_i, y = P_{i-1}

    Returns:
        grad_P_curr: Gradient of objective function w.r.t. P_curr.
        grad_P_prev: Gradient of objective function w.r.t. P_prev.
    """
    gradients = {p: torch.zeros((perm_sizes[p], perm_sizes[p])) for p in perm_matrices.keys()}

    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        assert layer in params_a.keys()
        assert layer in params_b.keys()

        Wa, Wb = params_a[layer], params_b[layer]
        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)

        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices.keys()
        row_perm = perm_matrices[row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0])

        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices.keys()
        col_perm = perm_matrices[col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1])

        grad_P_curr = compute_gradient_P_curr(Wa, Wb, col_perm)
        grad_P_prev = compute_gradient_P_prev(Wa, Wb, row_perm)

        if row_perm_id:
            gradients[row_perm_id] += grad_P_curr
        if col_perm_id:
            gradients[col_perm_id] += grad_P_prev

    return gradients


def weight_matching_gradient_fn(
    params_a, params_b, P_curr, P_curr_name, perm_to_axes, not_visited_params, perm_matrices, gradients
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

            Wa, Wb = params_a[params_name], params_b[params_name]
            assert Wa.shape == Wa.shape

            if Wa.dim() == 1:
                Wa = Wa.unsqueeze(1)
                Wb = Wb.unsqueeze(1)

            P_prev_name, P_prev = get_prev_permutation(params_name, perm_to_axes, perm_matrices)

            if not is_last_layer(params_and_axes):
                not_visited_params[params_name].remove(0)

                grad_P_curr = compute_gradient_P_curr(Wa, Wb, P_prev)

                gradients[P_curr_name] += grad_P_curr

            if P_prev_name is not None:
                grad_P_prev = compute_gradient_P_prev(Wa, Wb, P_curr)

                not_visited_params[params_name].remove(1)
                gradients[P_prev_name] += grad_P_prev


def is_last_layer(params_and_axes):
    return len(params_and_axes) == 1


def compute_single_perm_obj_function(params_a, params_b, P_curr, P_curr_name, perm_to_axes, perm_matrices, debug=True):
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

            P_prev_name, P_prev = get_prev_permutation(params_name, perm_to_axes, perm_matrices)

            layer_similarity = compute_layer_similarity(Wa, Wb, P_curr, P_prev, debug=debug)

            obj += layer_similarity

    return obj


def compute_layer_similarity(Wa, Wb, P_curr, P_prev, debug=True):
    # (P_i Wb_i) P_{i-1}^T

    # (P_i Wb_i)
    Wb_perm = perm_rows(perm=P_curr, x=Wb)
    if len(Wb.shape) == 2 and debug:
        assert torch.all(Wb_perm == P_curr @ Wb)

    if P_prev is not None:
        # (P_i Wb_i) P_{i-1}^T
        Wb_perm = perm_cols(x=Wb_perm, perm=P_prev.T)

        if len(Wb.shape) == 2 and debug:
            assert torch.all(Wb_perm == P_curr @ Wb @ P_prev.T)

    if len(Wa.shape) == 1:
        # vector case, result is the dot product of the vectors A^T B
        return Wa.T @ Wb_perm
    elif len(Wa.shape) == 2:
        # matrix case, result is the trace of the matrix product A^T B
        return torch.trace(Wa.T @ Wb_perm).numpy()
    elif len(Wa.shape) == 3:
        # tensor case, trace of a generalized inner product where the last dimensions are multiplied and summed
        return torch.trace(torch.einsum("ijk,jnk->in", Wa.transpose(1, 0), Wb_perm)).numpy()
    else:
        return torch.trace(torch.einsum("ijkm,jnkm->in", Wa.transpose(1, 0), Wb_perm)).numpy()


def compute_gradient_P_curr(Wa, Wb, P_prev):
    """
    (A P_{l-1} B^T)
    """

    if P_prev is None:
        P_prev = torch.eye(Wb.shape[1])

    assert Wa.shape == Wb.shape
    assert P_prev.shape[0] == Wb.shape[1]

    # P_{l-1} B^T
    Wb_perm = perm_rows(x=Wb.transpose(1, 0), perm=P_prev)
    if len(Wb.shape) == 2:
        assert torch.all(Wb_perm == P_prev @ Wb.T)

    if len(Wa.shape) == 2:
        grad_P_curr = Wa @ Wb_perm
    elif len(Wa.shape) == 3:
        grad_P_curr = torch.einsum("ijk,jnk->in", Wa, Wb_perm)
    else:
        grad_P_curr = torch.einsum("ijkm,jnkm->in", Wa, Wb_perm)

    return grad_P_curr


def compute_gradient_P_prev(Wa, Wb, P_curr):
    """
    (A^T P_l B)

    """
    assert P_curr.shape[0] == Wb.shape[0]

    grad_P_prev = None

    # (P_l B)
    Wb_perm = perm_rows(perm=P_curr, x=Wb)
    if len(Wb.shape) == 2:
        assert torch.all(Wb_perm == P_curr @ Wb)

    if len(Wa.shape) == 2:
        grad_P_prev = Wa.T @ Wb_perm
    elif len(Wa.shape) == 3:
        grad_P_prev = torch.einsum("ijk,jnk->in", Wa.transpose(1, 0), Wb_perm)
    else:
        grad_P_prev = torch.einsum("ijkm,jnkm->in", Wa.transpose(1, 0), Wb_perm)

    return grad_P_prev


def get_prev_permutation(params_name, perm_to_axes, perm_matrices):
    P_prev_name, P_prev = None, None

    for other_perm_name, other_perm in perm_matrices.items():

        # all the layers that are column-permuted by other_p
        params_perm_by_other_p = [tup[0] if tup[1] == 1 else None for tup in perm_to_axes[other_perm_name]]
        if params_name in params_perm_by_other_p:
            P_prev_name = other_perm_name
            P_prev = other_perm

    return P_prev_name, P_prev


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


def update_perm_matrices(perm_matrices, proj_grads, step_size):
    new_perm_matrices = {}

    for perm_name, perm in perm_matrices.items():

        if perm_name in {"P_final", "P_4"}:
            new_perm_matrices[perm_name] = perm
            continue

        proj_grad = proj_grads[perm_name]

        new_P_curr_interp = (1 - step_size) * perm + step_size * proj_grad
        new_perm_matrices[perm_name] = solve_linear_assignment_problem(new_P_curr_interp, return_matrix=True)

    return new_perm_matrices
