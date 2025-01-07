from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.optimize import fmin_l_bfgs_b, minimize
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ccmm.matching.frank_wolfe_matching import initialize_perturbed_uniform, sinkhorn_knopp
from ccmm.matching.weight_matching import solve_linear_assignment_problem


def compute_func_map(
    X, Y, P, radius=None, num_neighbors=None, mode="distance", normalize_lap=True, num_eigenvectors=50
):

    Xevecs, Yevecs, Xevals, Yevals = compute_eigenvectors(X, Y, radius, num_neighbors, mode, normalize_lap)

    C = Xevecs[:, :num_eigenvectors].T @ P @ Yevecs[:, :num_eigenvectors]

    return C, Xevecs, Yevecs


def compute_eigenvectors(X, Y, radius=None, num_neighbors=None, mode="distance", normalize_lap=True):

    X_adj_sym = build_knn_graph(X, radius, num_neighbors, mode)
    Y_adj_sym = build_knn_graph(Y, radius, num_neighbors, mode)

    if X_adj_sym.sum() == 0 or Y_adj_sym.sum() == 0:
        print("No edges in the graph")
        return np.zeros((X_adj_sym.shape[0], Y_adj_sym.shape[0]))

    XL, Xevals, Xevecs = build_laplacian(X_adj_sym, normalize_lap)
    YL, Yevals, Yevecs = build_laplacian(Y_adj_sym, normalize_lap)

    return Xevecs, Yevecs, Xevals, Yevals


def build_knn_graph(X, radius=None, num_neighbors=None, mode="distance"):
    assert radius is not None or num_neighbors is not None

    # print("input shape:", x.shape)

    if radius is not None:
        Xneigh = NearestNeighbors(radius=radius)

    elif num_neighbors is not None:
        Xneigh = NearestNeighbors(n_neighbors=num_neighbors)

    else:
        raise ValueError("Either radius or num_neighbors must be provided")

    Xneigh.fit(X)

    # (num_neurons, num_neurons)
    X_knn_graph = Xneigh.kneighbors_graph(X, mode=mode)

    X_adj = X_knn_graph.toarray()

    np.fill_diagonal(X_adj, 0)

    X_adj_sym = (X_adj + X_adj.T) / 2

    assert np.allclose(X_adj_sym, X_adj_sym.T), "Adjacences are not symmetric"

    return X_adj_sym


def build_laplacian(A, normalized=True):

    D = np.diag(np.sum(A, axis=1))

    assert not np.any(np.diag(D) <= 0)

    L = D - A

    if normalized:
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        L = D_inv_sqrt @ L @ D_inv_sqrt
        L = (L + L.T) / 2

    assert not np.any(np.isnan(L))

    # evecs are returned in columns
    evals, evecs = np.linalg.eigh(L)

    # idxs that sort the array in increasing order
    idx = evals.argsort()

    evals = evals[idx]
    evecs = evecs[:, idx]

    return L, evals, evecs


def plot_func_maps(func_maps, fig_name, vmin, vmax, cmap_name, layer_idx):
    fig, axs = plt.subplots(7, 7, figsize=(20, 20))

    k = range(3, 100, 2)

    for i in range(7):
        for j in range(7):
            ax = axs[i, j]
            fmap = func_maps[i * 7 + j][1:, 1:]  # Display without the first line and column
            ax.imshow(fmap, cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title(f"k={k[i * 7 + j]}")
    plt.savefig(f"figures/fmaps/{layer_idx}/{fig_name}.png")


"""Constraints for the optimization problem"""

# Descriptor preservation
def descr_preservation(C, descr1_proj, descr2_proj):
    """
    Compute the descriptor preservation constraint

    Parameters
    ---------------------
    C      :
        (K2,K1) Functional map
    descr1_proj :
        (K1,p) descriptors on first basis
    descr2_proj :
        (K2,p) descriptros on second basis

    Returns
    ---------------------
    float: gradient of the descriptor preservation constraint
    """

    # L2 squared
    return 0.5 * np.square(C @ descr1_proj - descr2_proj).sum()


# Laplace-Beltrami commutativity
def LB_Commutation(C, ev_sq_diff):
    """
    Compute the Laplace-Beltrami commutativity constraint

    Parameters
    ---------------------
    C      :
        (K2,K1) Functional map
    ev_sqdiff :
        (K2,K1) [normalized] matrix of squared eigenvalue differences

    Returns
    ---------------------
    energy : float
        (float) LB commutativity squared norm
    """
    return 0.5 * (np.square(C) * ev_sq_diff).sum()


# Operator commutativity
def op_commutation(C, op1, op2):
    """
    Compute the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op1 :
        (K1,K1) operator on first basis
    op2 :
        (K2,K2) descriptros on second basis

    Returns
    ---------------------
    energy : float
        (float) operator commutativity squared norm
    """
    return 0.5 * np.square(C @ op1 - op2 @ C).sum()


def oplist_commutation(C, op_list):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op_list :
        list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Returns
    ---------------------
    energy : float
        (float) sum of operators commutativity squared norm
    """
    energy = 0
    for op1, op2 in op_list:
        energy += op_commutation(C, op1, op2)

    return energy


"""Gradient computation for the optimization problem"""


def descr_preservation_grad(C, descr1_proj, descr2_proj):
    """
    Compute the gradient of the descriptor preservation constraint

    Parameters
    ---------------------
    C     :
        (K2,K1) Functional map
    descr1_proj :
        (K1,p) descriptors on first basis
    descr2_proj :
        (K2,p) descriptros on second basis

    Returns
    ---------------------
    gradient : np.ndarray
        gradient of the descriptor preservation squared norm
    """
    return (C @ descr1_proj - descr2_proj) @ descr1_proj.T


def LB_Commutation_grad(C, ev_sq_diff):
    """
    Compute the gradient of the LB commutativity constraint

    Parameters
    ---------------------
    C         :
        (K2,K1) Functional map
    ev_sqdiff :
        (K2,K1) [normalized] matrix of squared eigenvalue differences

    Returns
    ---------------------
    gradient : np.ndarray
        (K2,K1) gradient of the LB commutativity squared norm
    """
    return C * ev_sq_diff


def op_commutation_grad(C, op1, op2):
    """
    Compute the gradient of the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op1 :
        (K1,K1) operator on first basis
    op2 :
        (K2,K2) descriptros on second basis

    Returns
    ---------------------
    gardient : np.ndarray
        (K2,K1) gradient of the operator commutativity squared norm
    """
    return op2.T @ (op2 @ C - C @ op1) - (op2 @ C - C @ op1) @ op1.T


def oplist_commutation_grad(C, op_list):
    """
    Compute the gradient of the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op_list :
        list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Returns
    ---------------------
    gradient : np.ndarray
        (K2,K1) gradient of the sum of operators commutativity squared norm
    """
    gradient = 0
    for op1, op2 in op_list:
        gradient += op_commutation_grad(C, op1, op2)
    return gradient


def compute_descr_op(descr1, descr2, Xevecs, Yevecs, k1, k2):
    """
    Compute the multiplication operators associated with the descriptors

    Returns
    ---------------------------
    operators : list
        n_descr long list of ((k1,k1),(k2,k2)) operators.
    """

    list_descr = [
        (
            Xevecs[:, :k1].T @ (descr1[:, i, None] * Xevecs[:, :k1]),
            Yevecs[:, :k2].T @ (descr2[:, i, None] * Yevecs[:, :k2]),
        )
        for i in range(descr1.shape[1])
    ]

    return list_descr


def init_func_map(K1, K2, mode="zeros"):
    """
    Initialize the functional map for the optimization

    Returns
    ---------------------
    C_init : np.ndarray
    """

    if mode == "random":
        C_init = np.random.random((K2, K1))
    elif mode == "identity":
        C_init = np.eye(K2, K1)
    else:
        C_init = np.zeros((K2, K1))

    # C_init[:, 0] = np.zeros(
    #     K2
    # )  # In any case, the first column of the functional map is computed by hand and not modified during optimization

    return C_init


def loss_fn(C, w_descr, w_lap, w_dcomm, X_proj, Y_proj, list_descr=None, ev_sq_diff=None):
    K1 = X_proj.shape[0]
    K2 = Y_proj.shape[0]
    C = C.reshape(K1, K2)

    loss = 0

    loss += w_descr * descr_preservation(C, X_proj, Y_proj) + w_lap * LB_Commutation(C, ev_sq_diff)
    # + w_dcomm * oplist_commutation(C, list_descr)

    return loss


def grad_fn(C, w_descr, w_lap, w_dcomm, X_proj, Y_proj, list_descr, ev_sq_diff):

    K1 = X_proj.shape[0]
    K2 = Y_proj.shape[0]
    C = C.reshape(K1, K2)

    gradient = np.zeros_like(C)

    gradient += w_descr * descr_preservation_grad(C, X_proj, Y_proj) + w_lap * LB_Commutation_grad(C, ev_sq_diff)
    # + w_dcomm * oplist_commutation_grad(C, list_descr)
    # gradient +=  w_lap * LB_Commutation_grad(C, ev_sq_diff)

    # gradient[:, 0] = 0

    return gradient.reshape(-1)


def fit_func_map(
    X: List,
    Y: List,
    Xevecs: List,
    Yevecs: List,
    Xevals: List,
    Yevals: List,
    k1,
    k2,
    InitFM_mode,
    w_descr,
    w_lap,
    w_dcomm,
    iprint=-1,
    method="linsolve",
):

    # only keep the first k1 and k2 eigenvectors
    Phi = Xevecs[:, :k1]
    Psi = Yevecs[:, :k2]

    # project the descriptors onto the eigenvectors
    X_projs = Phi.T @ X  # (n_ev1, n_descr)
    Y_projs = Psi.T @ Y  # (n_ev2, n_descr)

    # Compute multiplicative operators associated to each descriptor
    # list_descr = []
    # if w_dcomm > 0:
    # print('Computing commutativity operators:', w_dcomm)
    # list_descr = compute_descr_op(X, Y, Xevecs, Yevecs, k1, k2)  # (n_descr, ((k1,k1), (k2,k2)) )

    # Compute the squared differences between eigenvalues for LB commutativity
    ev_sq_diff = np.square(Xevals[None, :k1] - Yevals[:k2, None])
    ev_sq_diff = ev_sq_diff / ev_sq_diff.sum()  # Scaling

    # Initialization
    C_init = init_func_map(k1, k2, mode=InitFM_mode)

    res = minimize(
        fun=loss_fn,
        x0=C_init,
        args=(w_descr, w_lap, w_dcomm, X_projs, Y_projs, None, ev_sq_diff),
        method="CG",
        jac=grad_fn,
        options={"maxiter": 1000, "disp": True},
    )
    FM = res.x.reshape((k2, k1))
    loss = res.fun

    return FM, loss


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    n_jobs          : number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def p2p_to_FM(p2p_21, evects1, evects2, A2=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.

    Parameters
    ------------------------------
    p2p_21    : (n2,) vertex to vertex map from target to source.
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                Can also be presented as a (n2,n1) sparse matrix.
    eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
    eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
    A2        : (n2,n2) area matrix of the target mesh. If specified, the eigenvectors can't be subsampled

    Outputs
    -------------------------------
    FM_12       : (k2,k1) functional map corresponding to the p2p map given.
                  Solved with pseudo inverse if A2 is given, else using least square.
    """
    # Pulled back eigenvectors
    evects1_pb = evects1[p2p_21, :] if np.asarray(p2p_21).ndim == 1 else p2p_21 @ evects1

    if A2 is not None:
        if A2.shape[0] != evects2.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")

        if A2.ndim == 1:
            return evects2.T @ (A2[:, None] * evects1_pb)  # (k2,k1)

        return evects2.T @ (A2 @ evects1_pb)  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(evects2, evects1_pb)[0]  # (k2,k1)


def FM_to_p2p(FM_12, evects1, evects2, use_adj=False, n_jobs=1):
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T

    Either one can transport the first diracs with the functional map or the second ones with
    the adjoint, which leads to different results (adjoint is the mathematically correct way)

    Parameters
    --------------------------
    FM_12     : (k2,k1) functional map from mesh1 to mesh2 in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_adj   : use the adjoint method
    n_jobs    : number of parallel jobs. Use -1 to use all processes


    Outputs:
    --------------------------
    P: permutation matrix mapping
    """
    k2, k1 = FM_12.shape

    assert k1 <= evects1.shape[1], f"At least {k1} should be provided, here only {evects1.shape[1]} are given"
    assert k2 <= evects2.shape[1], f"At least {k2} should be provided, here only {evects2.shape[1]} are given"

    if use_adj:
        emb1 = evects1[:, :k1]
        emb2 = evects2[:, :k2] @ FM_12

    else:
        emb1 = evects1[:, :k1] @ FM_12.T
        emb2 = evects2[:, :k2]

    S = emb1 @ emb2.T
    P = solve_linear_assignment_problem(S.T, return_matrix=True)

    return P
    # p2p_21 = knn_query(emb1, emb2,  k=1, n_jobs=n_jobs)
    # return p2p_21  # (n2,)


def zoomout_iteration(FM_12, evects1, evects2, step=1, A2=None, n_jobs=1):
    """
    Performs an iteration of ZoomOut.

    Parameters
    --------------------
    FM_12    : (k2,k1) Functional map from evects1[:,:k1] to evects2[:,:k2]
    evects1  : (n1,k1') eigenvectors on source shape with k1' >= k1 + step.
                 Can be a subsample of the original ones on the first dimension.
    evects2  : (n2,k2') eigenvectors on target shape with k2' >= k2 + step.
                 Can be a subsample of the original ones on the first dimension.
    step     : int - step of increase of dimension.
    A2       : (n2,n2) sparse area matrix on target mesh, for vertex to vertex computation.
                 If specified, the eigenvectors can't be subsampled !

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    """
    k2, k1 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p_21 = FM_to_p2p(FM_12, evects1, evects2, n_jobs=n_jobs)  # (n2,)
    # Compute the (k2+step, k1+step) FM
    FM_zo = p2p_to_FM(p2p_21, evects1[:, :new_k1], evects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(
    FM_AB,
    eigvecs_A,
    eigvecs_B,
    num_iters=10,
    step=1,
    A2=None,
    subsample=None,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    FM_AB      : (K,K) Functional map from from B to shape A
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : tuple or iterable of size 2. Each gives indices of vertices to sample
                 for faster optimization. If not specified, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    FM_12_zo  : zoomout-refined functional map from basis 1 to 2
    p2p_21_zo : only if return_p2p is set to True - the refined pointwise map from basis 2 to basis 1
    """
    k2_0, k1_0 = FM_AB.shape

    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert (
        k1_0 + num_iters * step1 <= eigvecs_A.shape[1]
    ), f"Not enough eigenvectors on source : \
        {k1_0 + num_iters*step1} are needed when {eigvecs_A.shape[1]} are provided"
    assert (
        k2_0 + num_iters * step2 <= eigvecs_B.shape[1]
    ), f"Not enough eigenvectors on target : \
        {k2_0 + num_iters*step2} are needed when {eigvecs_B.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_12_zo = FM_AB.copy()

    iterable = range(num_iters) if not verbose else tqdm(range(num_iters))
    for it in iterable:
        if use_subsample:
            FM_12_zo = zoomout_iteration(FM_12_zo, eigvecs_A[sub1], eigvecs_B[sub2], A2=None, step=step, n_jobs=n_jobs)

        else:
            FM_12_zo = zoomout_iteration(FM_12_zo, eigvecs_A, eigvecs_B, A2=A2, step=step, n_jobs=n_jobs)

    if return_p2p:
        p2p_21_zo = FM_to_p2p(FM_12_zo, eigvecs_A, eigvecs_B, n_jobs=n_jobs)  # (n2,)
        return FM_12_zo, p2p_21_zo

    return FM_12_zo


def graph_zoomout_refine(
    FM_AB, eigvecs_A, eigvecs_B, G1=None, G2=None, num_iters=10, step=1, subsample=None, verbose=False
):
    """
    Refines the functional map using ZoomOut and saves the result

    Parameters
    -------------------
    FM_AB      : (K, K) Functional map from from B to A
    eigvects1  : (n1, k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2, k2) eigenvectors on target shape with k2 >= K + nit
    nit       : int - number of iterations to do
    step      : increase in dimension at each Zoomout Iteration
    subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.
    overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                will be the zoomout refined FM (larger than the other 2)
    """
    if subsample is None or subsample == 0 or G1 is None or G2 is None:
        sub = None
    else:
        sub1 = G1.extract_fps(subsample)
        sub2 = G2.extract_fps(subsample)
        sub = (sub1, sub2)

    _FM_zo = zoomout_refine(
        FM_AB, eigvecs_A, eigvecs_B, num_iters, step=step, subsample=sub, return_p2p=False, n_jobs=1, verbose=verbose
    )

    return _FM_zo
