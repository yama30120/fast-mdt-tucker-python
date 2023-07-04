import numpy as np
from scipy.linalg import qr
from tmult import tmult
from all_prod import all_prod
from all_prod_exec_n import all_prod_exec_n
from unfold import unfold


# X = G x U
# min_{G, U} ||Y - X||_F^2
# return [G, U]
def als(Y, R):
    # d is the order of the tensor Y
    d = Y.ndim

    # U is factor matrices
    U = [];
    for n in range(0, d):
        Q, _ = qr(np.random.randn(Y.shape[n], R[n]))
        U.append(np.reshape(Q[:, 0:R[n]], [Y.shape[n], R[n]]))

    # ALS(alternating least squares)
    for i in range(0, 100):
        for n in range(0, d):
            U[n], _, _ = np.linalg.svd(unfold(all_prod_exec_n(Y, U, n, True), n))
            U[n] = U[n][:, 0:R[n]]

    # calulating the core tensor G
    G = all_prod(Y, U, True)
    return G, U
