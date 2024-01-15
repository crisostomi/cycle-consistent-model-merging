import logging

import numpy as np
import torch
from scipy.optimize import fminbound
from tqdm import tqdm

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.utils import compute_obj_function, perm_indices_to_perm_matrix, weight_matching_gradient_fn
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
    # ps.perm_to_axes["P_4"] = [("layer4.weight", 0)]

    # FOR RESNET
    ps.perm_to_axes["P_final"] = [("linear.weight", 0)]

    for perm_name, params_and_axes in ps.perm_to_axes.items():
        # params_and_axes is a list of tuples, e.g. [('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..]
        relevant_params, relevant_axis = params_and_axes[0]
        param_shape = params_a[relevant_params].shape
        perm_sizes[perm_name] = param_shape[relevant_axis]

    # initialize with identity permutation if none given
    all_perm_indices = {p: torch.arange(n) for p, n in perm_sizes.items()}
    # e.g. P0, P1, ..
    perm_names = list(all_perm_indices.keys())
    num_perms = len(perm_names)

    old_obj = 0.0
    patience_steps = 0

    for iteration in tqdm(range(max_iter), desc="Weight matching"):

        perm_order = torch.arange(num_perms)
        gradients = {p: torch.zeros((perm_sizes[p], perm_sizes[p])) for p in perm_names}

        for p_ix in perm_order:
            P_curr_name = perm_names[p_ix]
            P_prev_name = perm_names[p_ix - 1] if p_ix > 0 else None

            P_curr = perm_indices_to_perm_matrix(all_perm_indices[P_curr_name])
            P_prev = perm_indices_to_perm_matrix(all_perm_indices[P_prev_name]) if P_prev_name else None

            perm_to_axes = ps.perm_to_axes

            grad_P_curr, grad_P_prev = weight_matching_gradient_fn(
                params_a, params_b, P_curr, P_prev, P_curr_name, P_prev_name, perm_to_axes
            )

            if p_ix < num_perms:
                gradients[P_curr_name] += grad_P_curr
            if p_ix > 0:
                gradients[P_prev_name] += grad_P_prev

        pylogger.info(f"Iteration {iteration}")

        new_obj = 0.0

        for p_ix in perm_order[:-1]:

            P_curr_name = perm_names[p_ix]
            P_prev_name = perm_names[p_ix - 1] if p_ix > 0 else None

            P_curr = perm_indices_to_perm_matrix(all_perm_indices[P_curr_name])
            P_prev = perm_indices_to_perm_matrix(all_perm_indices[P_prev_name]) if P_prev_name else None

            perm_to_axes = ps.perm_to_axes

            # (num_neurons, num_neurons)
            # TODO: check wtf is going on, using the gradient or its transpose doesn't make any difference
            grad = gradients[P_curr_name]

            projected_grad_indices = solve_linear_assignment_problem(grad)

            proj_grad = perm_indices_to_perm_matrix(projected_grad_indices)

            def fun(t):
                p_curr_opt = (1 - t) * P_curr + t * proj_grad
                # TODO: understand if the projection is necessary
                p_curr_opt = perm_indices_to_perm_matrix(solve_linear_assignment_problem(p_curr_opt))
                return -compute_obj_function(
                    params_a, params_b, p_curr_opt, P_prev, P_curr_name, P_prev_name, perm_to_axes
                )

            step_size = fminbound(fun, 0, 1)

            new_P_curr_interp = (1 - step_size) * P_curr + step_size * (proj_grad)
            new_P_curr = solve_linear_assignment_problem(new_P_curr_interp)

            obj = compute_obj_function(
                params_a,
                params_b,
                perm_indices_to_perm_matrix(new_P_curr),
                P_prev,
                P_curr_name,
                P_prev_name,
                perm_to_axes,
            )

            new_obj += obj

            pylogger.info(f"{P_curr_name} step_size: {np.round(step_size, 4)}, obj {np.round(obj, 4)}")

            # TODO: check that it is correct to use the just computed perm or if we should assign a new all_perm_indices
            all_perm_indices[P_curr_name] = new_P_curr

        pylogger.info(f"Objective: {np.round(new_obj)}")
        if abs(old_obj - new_obj) < 1e-3:
            patience_steps += 1
        else:
            patience_steps = 0

        old_obj = new_obj

        if patience_steps >= 10:
            break

    return all_perm_indices
