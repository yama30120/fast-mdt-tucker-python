from fold import fold
from unfold import unfold

def tmult(X, A, k, order='C'):
    assert(X.shape[k] == A.shape[1])
    dim = list(X.shape)
    dim[k] = A.shape[0]
    return fold(A @ unfold(X, k, order=order), k, dim, order=order)
