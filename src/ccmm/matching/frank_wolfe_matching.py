import logging
from functools import partial

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
    ps.perm_to_axes["P_4"] = [("layer4.weight", 0)]

    # FOR RESNET
    # ps.perm_to_axes["P_final"] = [("linear.weight", 0)]

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
        # keep track if the params have been permuted in both axis 0 and 1
        not_visited_params = {
            param_name: {0, 1} if ("bias" not in param_name and "bn" not in param_name) else {0}
            for param_name in set(params_a.keys())
        }

        perm_order = torch.arange(num_perms)
        gradients = {p: torch.zeros((perm_sizes[p], perm_sizes[p])) for p in perm_names}

        for p_ix in perm_order:

            P_curr_name = perm_names[p_ix]
            P_curr = perm_indices_to_perm_matrix(all_perm_indices[P_curr_name])

            weight_matching_gradient_fn(
                params_a,
                params_b,
                P_curr,
                P_curr_name,
                ps.perm_to_axes,
                not_visited_params,
                perm_names,
                all_perm_indices,
                gradients,
            )

        pylogger.info(f"Iteration {iteration}")
        # pylogger.info(not_visited_params)

        new_obj = 0.0

        for p_ix in perm_order[:-1]:

            P_curr_name = perm_names[p_ix]
            P_curr = perm_indices_to_perm_matrix(all_perm_indices[P_curr_name])

            # (num_neurons, num_neurons)
            grad = gradients[P_curr_name]

            projected_grad_indices = solve_linear_assignment_problem(grad)
            proj_grad = perm_indices_to_perm_matrix(projected_grad_indices)

            obj_func = partial(
                compute_obj_function,
                params_a=params_a,
                params_b=params_b,
                perm_to_axes=ps.perm_to_axes,
                P_curr_name=P_curr_name,
                perm_names=perm_names,
                all_perm_indices=all_perm_indices,
            )

            line_search_step_func = partial(line_search_step, P_curr=P_curr, proj_grad=proj_grad, obj_func=obj_func)
            step_size = fminbound(line_search_step_func, 0, 1)

            new_P_curr_interp = (1 - step_size) * P_curr + step_size * (proj_grad)
            new_P_curr = solve_linear_assignment_problem(new_P_curr_interp)

            obj = obj_func(
                P_curr=perm_indices_to_perm_matrix(new_P_curr),
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


def line_search_step(t, P_curr, proj_grad, obj_func):
    p_curr_opt = (1 - t) * P_curr + t * proj_grad
    # TODO: understand if the projection is necessary
    # (in theory it shouldn't be, but then we have to permute stuff with a soft permutation)
    p_curr_opt = perm_indices_to_perm_matrix(solve_linear_assignment_problem(p_curr_opt))

    local_obj = -obj_func(
        P_curr=p_curr_opt,
    )
    return local_obj
