import logging
from enum import auto
from typing import List, Tuple

import numpy as np
import pygmtools as pygm
import scipy
import torch
from backports.strenum import StrEnum
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from ccmm.matching.utils import get_permuted_param, perm_indices_to_perm_matrix, perm_matrix_to_perm_indices
from ccmm.matching.weight_matching import PermutationSpec, solve_linear_assignment_problem
from ccmm.utils.utils import ModelParams


class DiagContent(StrEnum):
    """Enum for diagonal content of affinity matrix"""

    ONES = auto()
    SIMILARITIES = auto()


pylogger = logging.getLogger(__name__)


def quadratic_weight_matching(
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

            dist_aa = torch.zeros((num_neurons, num_neurons))
            dist_bb = torch.zeros((num_neurons, num_neurons))
            dist_ab = torch.zeros((num_neurons, num_neurons))

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

                dist_ab += w_a @ w_b.T  # / (torch.norm(w_a, dim=1) * torch.norm(w_b, dim=1))
                dist_aa += w_a @ w_a.T  # / (torch.norm(w_a, dim=1) * torch.norm(w_a, dim=1))
                dist_bb += w_b @ w_b.T  # / (torch.norm(w_b, dim=1) * torch.norm(w_b, dim=1))
                # dist_ab += w_a @ w_b.T #  torch.cdist(w_a, w_b) #
                #  torch.cdist(w_a, w_b) #
                # dist_aa += w_a @ w_a.T #  torch.cdist(w_a, w_a) #
                # dist_bb += w_b @ w_b.T #  torch.cdist(w_b, w_b) #

            # var_frac = 1e-2
            # # pylogger.info(f"dist_aa max: {dist_aa.max()}, dist_aa mean: {dist_aa.mean()}, dist_aa var: {dist_aa.var()} Variance: {var}")
            # # pylogger.info(f"dist_bb max: {dist_aa.max()}, dist_bb mean: {dist_aa.mean()}, dist_bb var: {dist_aa.var()}")

            # affinity = build_affinity_matrix(dist_aa, dist_bb, diag=DiagContent.SIMILARITIES, var_frac=var_frac, dist_ab=dist_ab, sparsify=True)

            # principal_eigenvector = get_principal_eigenvector(torch.tensor(affinity).cuda())

            # P_AB = extract_matching_leordeanu(principal_eigenvector)
            # P_AB = extract_matching_lap(principal_eigenvector)

            x0 = solve_linear_assignment_problem(dist_ab)
            init_perm = perm_indices_to_perm_matrix(x0).unsqueeze(0).numpy()

            dist_aa_batched = np.expand_dims(dist_aa, axis=0)
            dist_bb_batched = np.expand_dims(dist_bb, axis=0)
            num_neurons_batched = np.expand_dims(num_neurons, axis=0)

            conn1, edge1, ne1 = pygm.utils.dense_to_sparse(dist_aa_batched)
            conn2, edge2, ne2 = pygm.utils.dense_to_sparse(dist_bb_batched)

            # gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=dist_aa.max().numpy() * 1e-2)

            K = pygm.utils.build_aff_mat(
                node_feat1=None,
                edge_feat1=edge1,
                connectivity1=conn1,
                node_feat2=None,
                edge_feat2=edge2,
                connectivity2=conn2,
                n1=num_neurons_batched,
                ne1=ne1,
                n2=num_neurons_batched,
                ne2=ne2,
                edge_aff_fn=pygm.utils.inner_prod_aff_fn,
            )  # ) #
            X = pygm.ipfp(
                K, num_neurons_batched, num_neurons_batched, x0=init_perm, max_iter=200
            ).squeeze()  # ipfp, sm, , max_iter=200

            P_AB = pygm.hungarian(X)
            P_AB = torch.tensor(P_AB)

            perm_indices = perm_matrix_to_perm_indices(P_AB)

            # old_similarity = compute_weights_similarity(dist_ab, all_perm_indices[p])
            old_cost = compute_weights_similarity_metric(
                dist_aa, dist_bb, perm_indices_to_perm_matrix(all_perm_indices[p])
            )

            all_perm_indices[p] = perm_indices

            # new_similarity = compute_weights_similarity(dist_ab, all_perm_indices[p])
            new_cost = compute_weights_similarity_metric(
                dist_aa, dist_bb, perm_indices_to_perm_matrix(all_perm_indices[p])
            )

            pylogger.info(f"Iteration {iteration}, Permutation {p}: {old_cost - new_cost}")
            progress = progress or new_cost < old_cost - 1e-12

            # pylogger.info(f"Iteration {iteration}, Permutation {p}: {new_similarity - old_similarity}")
            # progress = progress or new_similarity > old_similarity + 1e-12  # 1e-12

        if not progress:
            break

    return all_perm_indices


def sparsify_similarities(dist_aa, dist_bb, dist_ab, K=2):
    mean_dist_aa, std_dist_aa = dist_aa.mean(), dist_aa.std()
    mean_dist_bb, std_dist_bb = dist_bb.mean(), dist_bb.std()
    mean_dist_ab, std_dist_ab = dist_ab.mean(), dist_ab.std()

    dist_aa[dist_aa > mean_dist_aa + K * std_dist_aa] = 1e2
    dist_bb[dist_bb > mean_dist_bb + K * std_dist_bb] = 1e2
    dist_ab[dist_ab > mean_dist_ab + K * std_dist_ab] = 1e2

    return dist_aa, dist_bb, dist_ab


def compute_weights_similarity_metric(dist_aa, dist_bb, perm_matrix):
    # M(A,B) = \| D_A - P * D_B * P' \|_2
    return torch.norm(dist_aa - perm_matrix @ dist_bb @ perm_matrix.T, p=2)


def build_affinity_matrix(dist_aa, dist_bb, diag: DiagContent, var_frac=1e-3, dist_ab=None, sparsify=True):

    num_neurons = dist_aa.size(0)
    num_matchings = num_neurons**2

    # Prepare the distance matrices for broadcasting
    dist_aa_broadcast = dist_aa.view(num_neurons, 1, num_neurons, 1).expand(-1, num_neurons, -1, num_neurons)
    dist_bb_broadcast = dist_bb.view(1, num_neurons, 1, num_neurons).expand(num_neurons, -1, num_neurons, -1)

    var = dist_aa.max() * var_frac
    S = torch.exp(-torch.abs(dist_aa_broadcast - dist_bb_broadcast) / var)

    S = S.reshape(num_matchings, num_matchings)

    if diag == DiagContent.ONES:
        diag_matrix = torch.eye(num_matchings)

    elif diag == DiagContent.SIMILARITIES:
        var = dist_ab.max() * var_frac
        sim_ab = torch.exp(-torch.abs(dist_ab) / var)

        sim_ab = sim_ab.reshape(num_matchings, 1)

        diag_matrix = torch.diag(sim_ab.squeeze())

    mask = torch.eye(num_matchings, dtype=torch.bool)
    S[mask] = 0

    S = S + diag_matrix

    if sparsify:
        mean_sim, std_sim = S.mean(), S.std()
        S[S < mean_sim + 2 * std_sim] = 0

    return S.numpy()


def normalize_zero_one(matrix):
    min_val = torch.min(matrix)
    max_val = torch.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def get_principal_eigenvector(M):

    num_neurons = torch.sqrt(torch.tensor(M.size(0))).int()

    values, vectors = torch.linalg.eigh(M)
    principal_eigenvector = vectors[:, torch.argmax(values)]

    principal_eigenvector = principal_eigenvector.reshape(num_neurons, num_neurons)
    principal_eigenvector = torch.abs(principal_eigenvector)

    return principal_eigenvector


def extract_matching_leordeanu(principal_eigenvector: Tensor):
    """
    principal_eigenvector: shape (num_neurons, num_neurons)
    """

    num_neurons = principal_eigenvector.shape[0]

    # Initialize the solution vector
    x = torch.zeros((num_neurons, num_neurons)).type_as(principal_eigenvector).long()

    # Initialize masks for rows and columns
    row_mask = torch.ones(num_neurons, dtype=torch.bool)
    col_mask = torch.ones(num_neurons, dtype=torch.bool)

    while True:
        # Apply masks to principal eigenvector
        masked_principal_eigenvector = principal_eigenvector.clone()
        masked_principal_eigenvector[~row_mask, :] = 0
        masked_principal_eigenvector[:, ~col_mask] = 0

        # Find the maximum value and its index
        flat_index = masked_principal_eigenvector.argmax()

        i, j = np.unravel_index(flat_index.item(), (num_neurons, num_neurons))

        assignment_value = masked_principal_eigenvector[(i, j)]
        if assignment_value == 0:
            break

        # Update the solution vector
        x[i, j] = 1

        # Update the masks to exclude row i and column j
        row_mask[i] = False
        col_mask[j] = False

    return x.cpu()


def extract_matching_lap(principal_eigenvector):
    num_neurons = principal_eigenvector.shape[0]

    principal_eigenvector = principal_eigenvector.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(principal_eigenvector.max() - principal_eigenvector)
    P_AB = scipy.sparse.coo_matrix(
        (np.ones(num_neurons), (row_ind, col_ind)), shape=(num_neurons, num_neurons)
    ).toarray()

    return torch.tensor(P_AB)


# def build_affinity_matrix(Wa, Wb):
#     n_points = len(Wa)

#     S = np.zeros((n_points**2, n_points**2))
#     for xi in range(n_points):
#         p_xi = Wa[xi,:]
#         for yi in range(n_points):
#             p_yi = Wb[yi,:]
#             for xj in range(n_points):
#                 p_xj = Wa[xj,:]
#                 dx = np.linalg.norm(p_xi - p_xj, ord=2)
#                 for yj in range(n_points):
#                     p_yj = Wb[yj,:]
#                     dy = np.linalg.norm(p_yi - p_yj, ord=2)
#                     S[xi * n_points + yi, xj * n_points + yj] = np.exp(-np.abs(dx - dy)/1e-2)

#     return S
