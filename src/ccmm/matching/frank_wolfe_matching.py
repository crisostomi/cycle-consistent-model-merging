import logging
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy  # NOQA
import torch
from pytorch_lightning import seed_everything
from scipy.optimize import fminbound
from tqdm import tqdm

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.utils import (
    PermutationIndices,
    PermutationMatrix,
    generalized_inner_product,
    perm_cols,
    perm_indices_to_perm_matrix,
    perm_rows,
)
from ccmm.matching.weight_matching import solve_linear_assignment_problem, weight_matching
from ccmm.utils.utils import ModelParams, to_np

pylogger = logging.getLogger(__name__)


def frank_wolfe_weight_matching(
    ps: PermutationSpec,
    fixed: ModelParams,
    permutee: ModelParams,
    initialization_method: str,
    max_iter=100,
    return_perm_history=False,
    num_trials=3,
    device="cuda",
    keep_soft_perms: bool = False,
):
    """
    Find a permutation of params_b to make them match params_a.

    :param ps: PermutationSpec
    :param fixed: the parameters to match
    :param permutee: the parameters to permute
    """
    params_a, params_b = fixed, permutee

    # FOR MLP
    # ps.perm_to_layers_and_axes["P_final"] = [("layer4.weight", 0)]

    # FOR RESNET
    # ps.perm_to_layers_and_axes["P_final"] = [("linear.weight", 0)]

    # FOR ViT
    # ps.perm_to_layers_and_axes["P_final"] = [("mlp_head.1.weight", 0)]

    # FOR VGG
    # ps.perm_to_layers_and_axes["P_final"] = [("classifier.4.weight", 0)]

    # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
    perm_sizes = collect_perm_sizes(ps, params_a)

    seeds = np.random.randint(0, 1000, num_trials)
    best_obj = 0.0

    for seed in tqdm(seeds, desc="Running multiple trials"):
        seed_everything(seed)

        perm_matrices, perm_matrices_history, trial_obj = frank_wolfe_weight_matching_trial(
            params_a, params_b, perm_sizes, initialization_method, ps, max_iter, device=device
        )
        pylogger.info(f"Trial objective: {trial_obj}")

        if trial_obj > best_obj:
            pylogger.info(f"New best objective! Previous was {best_obj}")
            best_obj = trial_obj
            best_perm_matrices = perm_matrices
            best_perm_matrices_history = perm_matrices_history

    all_perm_indices = {
        p: perm if keep_soft_perms else solve_linear_assignment_problem(perm) for p, perm in best_perm_matrices.items()
    }

    if return_perm_history:
        return all_perm_indices, best_perm_matrices_history
    else:
        return all_perm_indices


def collect_perm_sizes(perm_spec, ref_params):
    perm_sizes = {}

    for perm_name, params_and_axes in perm_spec.perm_to_layers_and_axes.items():
        relevant_params, relevant_axis = params_and_axes[0]
        param_shape = ref_params[relevant_params].shape
        perm_sizes[perm_name] = param_shape[relevant_axis]

    return perm_sizes


def frank_wolfe_weight_matching_trial(
    params_a, params_b, perm_sizes, initialization_method, perm_spec, max_iter=100, device="cuda", global_step_size=True
):

    perm_matrices: Dict[str, PermutationMatrix] = initialize_perm_matrices(
        perm_sizes, initialization_method, params_a, params_b, perm_spec, device=device
    )
    perm_matrices_history = [perm_matrices]

    old_obj = 0.0
    patience_steps = 0

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        pylogger.info(f"Iteration {iteration}")

        gradients = weight_matching_gradient_fn(
            params_a, params_b, perm_matrices, perm_spec.layer_and_axes_to_perm, perm_sizes
        )

        proj_grads = project_gradients(gradients, device)

        step_size = compute_step_size(proj_grads, perm_matrices, params_a, params_b, perm_spec, global_step_size)

        perm_matrices = update_perm_matrices(perm_matrices, proj_grads, step_size)

        new_obj = get_global_obj_layerwise(params_a, params_b, perm_matrices, perm_spec.layer_and_axes_to_perm)

        pylogger.info(f"Objective: {np.round(new_obj, 6)}")

        if (new_obj - old_obj) < 1e-4:
            patience_steps += 1
        else:
            patience_steps = 0
            old_obj = new_obj

        if patience_steps >= 5:
            break

        perm_matrices_history.append(perm_matrices)

    return perm_matrices, perm_matrices_history, new_obj


def initialize_perm_matrices(
    perm_sizes, initialization_method, fixed=None, permutee=None, perm_spec=None, device="cpu"
):
    if initialization_method == "identity":
        return {p: torch.eye(n).to(device) for p, n in perm_sizes.items()}
    elif initialization_method == "random":
        return {p: torch.rand(n, n).to(device) for p, n in perm_sizes.items()}
    elif initialization_method == "sinkhorn":
        return {p: sinkhorn_knopp(initialize_perturbed_uniform(n)) for p, n in perm_sizes.items()}
    elif initialization_method == "LAP":
        perm_indices = weight_matching(perm_spec, fixed, permutee)
        return {p: perm_indices_to_perm_matrix(perm_indices[p]).to(device) for p in perm_indices.keys()}
    else:
        raise ValueError(f"Unknown initialization method {initialization_method}")


def project_gradients(gradients, device):
    proj_grads = {}

    for perm_name, grad in gradients.items():

        proj_grad = solve_linear_assignment_problem(grad, return_matrix=True)

        proj_grads[perm_name] = proj_grad.to(device)

    return proj_grads


def compute_step_size(
    proj_grads, perm_matrices, params_a, params_b, perm_spec, global_step_size=True
) -> Union[float, np.array]:

    line_search_step_func = partial(
        line_search_step,
        proj_grads=proj_grads,
        perm_matrices=perm_matrices,
        params_a=params_a,
        params_b=params_b,
        layers_and_axes_to_perms=perm_spec.layer_and_axes_to_perm,
    )

    if global_step_size:
        step_size = fminbound(line_search_step_func, 0, 1)
        pylogger.info(f"Step size: {step_size}")

    else:
        x0 = np.array([0.0] * len(perm_matrices))
        bounds = [(0, 1)] * len(perm_matrices)
        res = scipy.optimize.minimize(line_search_step_func, x0, bounds=bounds, method="Nelder-Mead")
        step_size, success = res.x, res.success
        pylogger.info(f"Success: {success} Step size: {step_size}")

    return step_size


def line_search_step(
    step_size: Union[float, np.array],
    params_a,
    params_b,
    proj_grads: Dict[str, torch.Tensor],
    perm_matrices: Dict[str, PermutationIndices],
    layers_and_axes_to_perms,
):

    interpolated_perms = {}

    for ind, (perm_name, perm) in enumerate(perm_matrices.items()):
        proj_grad = proj_grads[perm_name]

        if perm_name == "P_final":
            interpolated_perms[perm_name] = perm
            continue

        alpha = step_size if isinstance(step_size, float) else step_size[ind]
        interpolated_perms[perm_name] = (1 - alpha) * perm + alpha * proj_grad

    tot_obj = get_global_obj_layerwise(params_a, params_b, interpolated_perms, layers_and_axes_to_perms)

    return -tot_obj


def get_global_obj_layerwise(params_a, params_b, perm_matrices, layers_and_axes_to_perms, device="cuda"):

    tot_obj = 0.0

    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        if (
            "num_batches_tracked" in layer
            or "running_mean" in layer
            or "running_var" in layer
            or "temperature" in layer
        ):
            continue

        assert layer in params_a.keys()
        assert layer in params_b.keys()

        Wa, Wb = params_a[layer], params_b[layer]
        Wa, Wb = Wa.to(device), Wb.to(device)
        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)

        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices.keys()
        row_perm = perm_matrices[row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0], device=device)

        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices.keys()
        col_perm = perm_matrices[col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1], device=device)

        layer_similarity = compute_layer_similarity(Wa, Wb, row_perm, col_perm)

        tot_obj += layer_similarity

    return tot_obj


def weight_matching_gradient_fn(params_a, params_b, perm_matrices, layers_and_axes_to_perms, perm_sizes, device="cuda"):
    """
    Compute gradient of the weight matching objective function w.r.t. P_curr and P_prev.
    sim = <Wa_i, Pi Wb_i P_{i-1}^T>_f where f is the Frobenius norm, rewrite it as < A, xBy^T>_f where A = Wa_i, x = Pi, B = Wb_i, y = P_{i-1}

    Returns:
        grad_P_curr: Gradient of objective function w.r.t. P_curr.
        grad_P_prev: Gradient of objective function w.r.t. P_prev.
    """
    gradients = {p: torch.zeros((perm_sizes[p], perm_sizes[p]), device=device) for p in perm_matrices.keys()}

    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        if (
            "num_batches_tracked" in layer
            or "running_mean" in layer
            or "running_var" in layer
            or "temperature" in layer
        ):
            continue

        assert layer in params_a.keys()
        assert layer in params_b.keys()

        Wa, Wb = params_a[layer], params_b[layer]
        Wa, Wb = Wa.to(device), Wb.to(device)
        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)

        # any permutation acting on the first axis is permuting rows
        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices.keys()
        row_perm = perm_matrices[row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0], device=device)

        # any permutation acting on the second axis is permuting columns
        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices.keys()
        col_perm = perm_matrices[col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1], device=device)

        grad_P_curr = compute_gradient_P_curr(Wa, Wb, col_perm)
        grad_P_prev = compute_gradient_P_prev(Wa, Wb, row_perm)

        if row_perm_id:
            gradients[row_perm_id] += grad_P_curr
        if col_perm_id:
            gradients[col_perm_id] += grad_P_prev

    return gradients


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
    """
    Compute (P_i Wb_i) P_{i-1}^T
    """

    # (P_i Wb_i)
    Wb_perm = perm_rows(perm=P_curr, x=Wb)

    if P_prev is not None:
        # (P_i Wb_i) P_{i-1}^T
        Wb_perm = perm_cols(x=Wb_perm, perm=P_prev.T)

    inner_product = generalized_inner_product(Wa.transpose(1, 0), Wb_perm)
    sim = torch.trace(inner_product)

    if debug and len(Wa.shape) == 2:
        assert torch.allclose(
            sim, torch.trace(Wa.T @ P_curr @ Wb @ P_prev.T), atol=1e-3
        ), f"{sim} != {torch.trace(Wa.T @ P_curr @ Wb @ P_prev.T)}"

    return to_np(sim)


def compute_gradient_P_curr(Wa, Wb, P_prev, debug=True):
    """
    (A P_{l-1} B^T)
    """

    if P_prev is None:
        P_prev = torch.eye(Wb.shape[1])

    assert Wa.shape == Wb.shape
    assert P_prev.shape[0] == Wb.shape[1]

    # P_{l-1} B^T
    Wb_perm = perm_rows(x=Wb.transpose(1, 0), perm=P_prev)

    grad_P_curr = generalized_inner_product(Wa, Wb_perm)

    if debug and len(Wa.shape) == 2:
        assert torch.allclose(grad_P_curr, Wa @ P_prev @ Wb.T, atol=1e-5)

    return grad_P_curr


def compute_gradient_P_prev(Wa, Wb, P_curr, debug=True):
    """
    (A^T P_l B)

    """
    assert P_curr.shape[0] == Wb.shape[0]

    grad_P_prev = None

    # (P_l B)
    Wb_perm = perm_rows(perm=P_curr, x=Wb)

    grad_P_prev = generalized_inner_product(Wa.transpose(1, 0), Wb_perm)

    if debug and len(Wa.shape) == 2:
        assert torch.allclose(grad_P_prev, Wa.T @ P_curr @ Wb, atol=1e-3)

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


def update_perm_matrices(perm_matrices, proj_grads, step_size: Union[float, np.array]):
    new_perm_matrices = {}

    for i, (perm_name, perm) in enumerate(perm_matrices.items()):

        if perm_name == "P_final":
            new_perm_matrices[perm_name] = perm
            continue

        proj_grad = proj_grads[perm_name]

        alpha = step_size if isinstance(step_size, float) else step_size[i]

        new_P_curr_interp = (1 - alpha) * perm + alpha * proj_grad
        new_perm_matrices[perm_name] = new_P_curr_interp

    return new_perm_matrices


def sinkhorn_knopp(matrix, tol=1e-8, max_iterations=10000, device="cuda"):
    """
    Applies the Sinkhorn-Knopp algorithm to make a non-negative matrix doubly stochastic.

    Parameters:
    matrix (2D torch tensor): A non-negative matrix.
    tol (float): Tolerance for the stopping condition.
    max_iterations (int): Maximum number of iterations.

    Returns:
    2D torch tensor: Doubly stochastic matrix.
    """
    if not torch.all(matrix >= 0):
        raise ValueError("Matrix contains negative values.")

    R, C = matrix.size()

    if R != C:
        raise ValueError("Matrix must be square.")

    matrix += 1e-6

    for iter in range(max_iterations):
        matrix /= matrix.sum(dim=1, keepdims=True)

        matrix /= matrix.sum(dim=0, keepdims=True)

        # Check if matrix is close enough to doubly stochastic
        if torch.all(torch.abs(matrix.sum(dim=0) - 1) < tol) and torch.all(torch.abs(matrix.sum(dim=1) - 1) < tol):
            pylogger.debug(f"Sinkhorn-Knopp algorithm converged after {iter} iterations.")

            return matrix

    row_diff = torch.abs(matrix.sum(dim=0) - 1).sum()
    col_diff = torch.abs(matrix.sum(dim=1) - 1).sum()
    pylogger.debug(
        f"Sinkhorn-Knopp algorithm did not converge, row_diff: {row_diff.item()}, col_diff: {col_diff.item()}"
    )

    return matrix


def initialize_perturbed_uniform(n, device="cuda"):
    # Start with a uniform matrix
    A = torch.ones((n, n), device=device) / n

    # Apply a small random perturbation
    perturbation = torch.rand((n, n), device=device) * 0.01 / n
    A += perturbation

    # Ensure A remains non-negative
    A = torch.clip(A, min=0, max=None)

    return A


# def sinkhorn_knopp(matrix, tol=1e-8, max_iterations=100000, device="cuda"):
#     matrix = matrix.to(device)

#     log_A = torch.log(matrix)

#     for _ in range(max_iterations):
#         # Row normalization in log space
#         log_A -= log_A.exp().sum(dim=1, keepdims=True).log()

#         # Column normalization in log space
#         log_A -= log_A.exp().sum(dim=0, keepdims=True).log()

#         # Convergence check (optional)
#         if (
#             torch.max(torch.abs(log_A.exp().sum(axis=1) - 1)) < tol
#             and torch.max(torch.abs(log_A.exp().sum(axis=0) - 1)) < tol
#         ):
#             pylogger.info(f"Sinkhorn-Knopp algorithm converged after {_} iterations.")
#             return log_A.exp()

#     pylogger.info("Sinkhorn-Knopp algorithm did not converge.")
#     return log_A.exp()


# TODO:
def projected_grad_descent_weight_matching_trial(
    params_a, params_b, perm_sizes, initialization_method, perm_spec, max_iter=100
):
    pass  # TODO
    perm_matrices: Dict[str, PermutationMatrix] = initialize_perm_matrices(
        perm_sizes, initialization_method, params_a, params_b, perm_spec
    )
    perm_matrices_history = [perm_matrices]

    old_obj = 0.0
    patience_steps = 0

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        pylogger.debug(f"Iteration {iteration}")

        gradients = weight_matching_gradient_fn(
            params_a, params_b, perm_matrices, perm_spec.layer_and_axes_to_perm, perm_sizes
        )

        proj_grads = project_gradients(gradients)

        line_search_step_func = partial(
            line_search_step,
            proj_grads=proj_grads,
            perm_matrices=perm_matrices,
            params_a=params_a,
            params_b=params_b,
            layers_and_axes_to_perms=perm_spec.layer_and_axes_to_perm,
        )

        step_size = fminbound(line_search_step_func, 0, 1)
        pylogger.info(f"Step size: {step_size}")

        perm_matrices = update_perm_matrices(perm_matrices, proj_grads, step_size)

        new_obj = get_global_obj_layerwise(params_a, params_b, perm_matrices, perm_spec.layer_and_axes_to_perm)

        pylogger.info(f"Objective: {np.round(new_obj, 6)}")

        if (new_obj - old_obj) < 1e-4:
            patience_steps += 1
        else:
            patience_steps = 0
            old_obj = new_obj

        if patience_steps >= 5:
            break

        perm_matrices_history.append(perm_matrices)

    return perm_matrices, perm_matrices_history, new_obj
