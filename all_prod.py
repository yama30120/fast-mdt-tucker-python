from tmult import tmult

def all_prod(G, U, transpose=False):
    d = G.ndim
    X = G
    for n in range(0, d):
        if transpose:
            X = tmult(X, U[n].T, n)
        else:
            X = tmult(X, U[n], n)
    return X
