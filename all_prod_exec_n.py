from tmult import tmult

def all_prod_exec_n(G, U, n, transpose=False):
    d = G.ndim
    X = G
    for k in range(0, d):
        if k == n:
            continue
        if transpose:
            X = tmult(X, U[k].T, k)
        else:
            X = tmult(X, U[k], k)
    return X

