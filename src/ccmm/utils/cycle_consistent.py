import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import eye as speye
from scipy.sparse.linalg import svds

pylogger = logging.getLogger(__name__)


def gerror_sparse(N, n, clstack, truth):
    # sparse identity
    sparse_identity = speye(n).toarray()

    # setting up objective matrix using clstack information
    clrow = []
    for i in range(N):
        for j in range(N):
            foo = sparse_identity[:, clstack[i][j]]
            clrow.append(foo)
    clrow = np.array(clrow)

    S = []  # objective matrix
    ss = []
    for j in range(0, N * n, N * n):
        for i in range(0, n * N, n):
            c = clrow[j + i : j + i + n, :]
            ss.append(c)
        S.append(ss)
        ss = []
    S = np.array(S)

    # Sparse singular value decomposition
    u, s, v = svds(S, n + 10)
    ev = np.diag(s)
    print("svd S")

    # Collecting top-n left singular vectors
    u = u[:, :n]

    # Setting up Permutations for each instance in dataset
    B = {}
    for i in range(0, N * n, n):
        B[(i + (n - 1)) // n] = u[i : i + n, :]

    recPM = {}
    for i in range(N):
        tmp = np.dot(B[i], B[0].T)
        tmpU, tmpS, tmpV = np.linalg.svd(tmp)
        recPM[i] = np.dot(tmpU, tmpV.T)

    recPerm = []
    for i in range(N):
        row_ind, col_ind = linear_sum_assignment(-recPM[i].T)
        recPerm.append(col_ind)

    # Error calculation - only correct if truth is provided
    recPerm = np.array(recPerm)
    grossError = np.zeros_like(truth)
    ind = np.where(truth - recPerm)
    grossError[ind] = 1
    grossError = np.sum(grossError) / (N * n)

    return ev, grossError, recPM, recPerm
